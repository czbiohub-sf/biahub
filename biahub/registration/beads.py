"""
Beads-based registration module.

Provides functions for registering volumetric imaging data by detecting fluorescent
bead landmarks in moving and reference channels, matching them using graph-based
algorithms, and estimating affine transformations.

Pipeline overview
-----------------
1. **Peak detection** (`peaks_from_beads`): Detect bead positions in both channels.
2. **Matching** (`matches_from_beads`): Find bead correspondences via graph matching
   (Hungarian or descriptor-based) with geometric consistency filtering.
3. **Transform estimation** (`transform_from_matches`): Fit an affine/euclidean/similarity
   transform from matched bead pairs.
4. **Iterative refinement** (`optimize_transform`, `estimate`): Compose the approximate
   transform with bead-based corrections, re-detect peaks, and score until convergence.
5. **Parameter tuning** (`optimize_matches`): Grid search over matching settings to find
   the combination that maximizes registration quality.

Key conventions
---------------
- Coordinates are in ZYX order for 3D data.
- "mov" / "moving" refers to the source channel being aligned.
- "ref" / "reference" refers to the fixed target channel.
- Transforms map from moving space to reference space (forward direction).
"""

import json

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Literal

import ants
import click
import dask.array as da
import numpy as np
import submitit

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform

from biahub.characterize_psf import detect_peaks
from biahub.cli.parsing import (
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import (
    _check_nan_n_zeros,
    estimate_resources,
)
from biahub.core.graph_matching import Graph, GraphMatcher
from biahub.core.transform import Transform
from biahub.registration.qc import flag_timepoints, write_qc_report
from biahub.registration.utils import (
    get_aprox_transform,
    load_quality_scores,
    load_transforms,
    plot_quality_scores,
    save_quality_score,
)
from biahub.settings import AffineTransformSettings, BeadsMatchSettings, DetectPeaksSettings


# Default grid for the score-gated sweep fallback, distilled from a 432-combination
# hungarian search and a 200-combination spectral search over the same 15 flagged timepoints
# on two real datasets.
#
# A LIST of sub-grids, not one dict, because the two matchers are separate regimes: the
# hungarian cost/graph parameters are inert when spectral does the matching and vice versa,
# so a single cross product would spend most of its trials re-measuring identical results.
# The union is 16 + 12 = 28 trials; the equivalent cross product would be 192.
#
# Hungarian axes, kept only where one actually decided a winner:
#   cost_threshold  0.05 won at all 15 timepoints, against a global default of 0.10.
#   weights_dist    0.25 won at 9 of 15 -- de-weighting position distance is what helps when
#                   the initial transform is off by more than a bead spacing.
#   method          "full" won at 3, and was only discoverable once EdgeGraphSettings.method
#                   stopped being ignored; the kNN graph can omit the correct edge outright.
#   max_distance_quantile  the loosest useful filter axis on sparse match sets.
# Excluded: k (never decided a winner once method was free) and angle_threshold /
# weights_edge_angle (both 2D-only, hence flat on ZYX data).
#
# What this grid delivers, measured on the 15 flagged timepoints, taking max against the
# incumbent: 8 of 15 improved, mean gain +0.041 over all 15 and +0.077 over those it helped.
# That is 83% of what an exhaustive 632-trial search over the same axes achieves (11 of 15,
# +0.049) for 4% of the trials, which is why the axes are trimmed this hard.
#
# The spectral sub-grid is what makes the fallback worth running at all: it supplies 8 of the
# 9 rescues, and a hungarian-only sweep gets only 5 of 15.
#
# Its ranges are bounded by a SEPARATE stratified sweep over 24 evenly-spaced timepoints,
# not by what won at the weak ones, because those two searches disagree and the stratified
# one is the trustworthy guide to whether a value is safe. Per-timepoint argmax on the weak
# set picked rel_cut 0.2 at 8 of 15 timepoints, yet across the stratified set rel_cut 0.2 is
# the single worst choice available (mean 0.694, worst case 0.000 -- outright failure at some
# timepoints), and sigma 8.0 likewise collapses (mean 0.624, worst 0.000). Those apparent
# wins were selection noise: with ~20 beads the overlap metric quantises at ~0.045, so
# picking the max of 200 trials reliably finds a lucky flat-metric tie.
#
# Restricting to the values that never collapse costs almost nothing at the weak timepoints
# it is meant to rescue -- 10 of 15 improved instead of 11, mean gain +0.0466 against
# +0.0490, a difference of one timepoint well inside one quantisation step -- so the safe
# ranges are strictly the better trade.
DEFAULT_SWEEP_GRID = [
    {
        "cost_threshold": [0.05, 0.10],
        "weights_dist": [0.25, 1.0],
        "method": ["knn", "full"],
        "max_distance_quantile": [0.95, 0.99],
    },
    {
        "algorithm": ["spectral"],
        "spectral_sigma": [1.0, 3.0, 5.0],
        "spectral_rel_cut": [0.35, 0.65, 0.80],
    },
]


def _set_graph_method(edge_graph_settings, method: str) -> None:
    """Switch graph mode, supplying the parameter that mode needs.

    EdgeGraphSettings' validator nulls whichever of k/radius the chosen method does not use,
    so setting method alone can leave the other unset and the matcher then fails. Defaults
    here match that validator's.
    """
    edge_graph_settings.method = method
    if method == "knn" and edge_graph_settings.k is None:
        edge_graph_settings.k = 10
    elif method == "radius" and edge_graph_settings.radius is None:
        edge_graph_settings.radius = 60.0


def optimize_matches(
    mov: ArrayLike,
    ref: ArrayLike,
    approx_transform: Transform,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    param_grid: dict = None,
    verbose: bool = False,
) -> tuple[BeadsMatchSettings, Transform | None, float]:
    """
    Optimize BeadsMatchSettings by grid search over matching and filter parameters.

    For each parameter combination: detects peaks in approximately registered space,
    matches them, estimates a correction transform, composes it with the approx transform,
    applies to the full volume via ANTs, re-detects peaks, and scores the overlap.

    Peaks are detected once and reused: they depend only on approx_transform, not on the
    matching parameters. Trials whose filtered match set is identical to one already
    evaluated reuse that score instead of repeating the ANTs warp and peak re-detection,
    which is the dominant cost -- many filter combinations collapse to the same match set.

    Parameters
    ----------
    mov : ArrayLike
        Original (unregistered) moving volume (Z, Y, X).
    ref : ArrayLike
        Reference volume (Z, Y, X).
    approx_transform : Transform
        Initial approximate transform to compose with.
    beads_match_settings : BeadsMatchSettings
        Initial matching settings to use as baseline.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform estimation.
    param_grid : dict | list[dict], optional
        Parameter names to lists of values. A list of such dicts is searched as the UNION of
        their cross products, which is how mutually inert parameter sets -- the hungarian
        cost parameters and the spectral ones -- are swept without paying for their cross
        product. Defaults to DEFAULT_SWEEP_GRID. Unsupported keys raise ValueError rather
        than being ignored, so a typo cannot silently produce a flat axis. See the setter
        table in the body for what is supported.
    verbose : bool
        If True, prints logs for each trial.

    Returns
    -------
    tuple[BeadsMatchSettings, Transform | None, float]
        The settings that produced the best overlap score, the transform they produced
        (already composed with approx_transform, so directly comparable with
        optimize_transform's return), and that score. The transform is None and the score
        -1 if no trial produced a scoreable transform, in which case the returned settings
        are the unmodified input.
    """
    if param_grid is None:
        param_grid = DEFAULT_SWEEP_GRID

    score_radius = beads_match_settings.qc_settings.score_centroid_mask_radius

    # Convert volumes to ANTs images once (reused across all trials)
    mov_ants = ants.from_numpy(mov.astype(np.float32))
    ref_ants = ants.from_numpy(ref.astype(np.float32))

    # Apply approximate transform to moving volume and detect peaks once.
    # These peaks are reused for all parameter combinations in the grid search.
    click.echo("Detecting peaks in approximately registered space for grid search...")
    mov_reg_approx = (
        approx_transform.to_ants().apply_to_image(mov_ants, reference=ref_ants).numpy()
    )
    mov_peaks, ref_peaks = peaks_from_beads(
        mov=mov_reg_approx,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=False,
    )
    if mov_peaks is None or ref_peaks is None or len(mov_peaks) < 2 or len(ref_peaks) < 2:
        click.echo("Not enough peaks detected for optimization, returning original settings.")
        return beads_match_settings, None, -1.0

    # A bare dict is the single-sub-grid case; normalising here keeps one loop below.
    sub_grids = [param_grid] if isinstance(param_grid, dict) else list(param_grid)
    trial_params_list = [
        dict(zip(sub, combo, strict=True))
        for sub in sub_grids
        for combo in product(*(sub[k] for k in sub))
    ]

    click.echo(
        f"Starting grid search: {len(mov_peaks)} mov peaks, {len(ref_peaks)} ref peaks, "
        f"{len(trial_params_list)} parameter combinations"
        + (f" over {len(sub_grids)} sub-grids." if len(sub_grids) > 1 else ".")
    )

    ndim = mov_peaks.shape[1]
    best_score = -1.0
    best_settings = beads_match_settings
    best_transform = None

    def apply_trial_params(trial_settings, trial_params):
        """Apply parameter values from a grid search trial to a BeadsMatchSettings copy."""
        fm = trial_settings.filter_matches_settings
        hm = trial_settings.hungarian_match_settings
        eg = hm.edge_graph_settings
        sm = trial_settings.spectral_match_settings
        w = hm.cost_matrix_settings.weights
        param_map = {
            "min_distance_quantile": lambda v: setattr(fm, "min_distance_quantile", v),
            "max_distance_quantile": lambda v: setattr(fm, "max_distance_quantile", v),
            "direction_threshold": lambda v: setattr(fm, "direction_threshold", v),
            "angle_threshold": lambda v: setattr(fm, "angle_threshold", v),
            "cost_threshold": lambda v: setattr(hm, "cost_threshold", v),
            "max_ratio": lambda v: setattr(hm, "max_ratio", v),
            "k": lambda v: setattr(eg, "k", v),
            # method needs k/radius set consistently or EdgeGraphSettings' validator
            # nulls whichever one the new mode requires.
            "method": lambda v: _set_graph_method(eg, v),
            "radius": lambda v: setattr(eg, "radius", v),
            "algorithm": lambda v: setattr(trial_settings, "algorithm", v),
            "spectral_sigma": lambda v: setattr(sm, "sigma", v),
            "spectral_rel_cut": lambda v: setattr(sm, "rel_cut", v),
            "weights_dist": lambda v: w.__setitem__("dist", v),
            "weights_edge_angle": lambda v: w.__setitem__("edge_angle", v),
            "weights_edge_length": lambda v: w.__setitem__("edge_length", v),
            "weights_pca_dir": lambda v: w.__setitem__("pca_dir", v),
            "weights_pca_aniso": lambda v: w.__setitem__("pca_aniso", v),
            "weights_edge_descriptor": lambda v: w.__setitem__("edge_descriptor", v),
        }
        unsupported = set(trial_params) - set(param_map)
        if unsupported:
            raise ValueError(
                f"Unsupported param_grid key(s) {sorted(unsupported)}. "
                f"Supported: {sorted(param_map)}"
            )
        for key, val in trial_params.items():
            param_map[key](val)

    # Score keyed on (algorithm, match set), so a trial can only reuse a score computed
    # from an identical match set under the same matcher.
    score_cache: dict[tuple, float] = {}
    transform_cache: dict[tuple, Transform] = {}

    for trial_params in trial_params_list:
        trial_settings = beads_match_settings.model_copy(deep=True)
        apply_trial_params(trial_settings, trial_params)

        try:
            matches = matches_from_beads(
                mov_peaks=mov_peaks,
                ref_peaks=ref_peaks,
                beads_match_settings=trial_settings,
                verbose=False,
            )

            if len(matches) < 3:
                continue

            cache_key = (
                trial_settings.algorithm,
                tuple(map(tuple, np.asarray(matches))),
            )
            if cache_key in score_cache:
                if score_cache[cache_key] > best_score:
                    best_score = score_cache[cache_key]
                    best_settings = trial_settings
                    best_transform = transform_cache[cache_key]
                if verbose:
                    click.echo(
                        f"  {trial_params} -> matches={len(matches)}, "
                        f"score={score_cache[cache_key]:.4f} (cached)"
                    )
                continue

            fwd_transform, inv_transform = transform_from_matches(
                matches=matches,
                mov_peaks=mov_peaks,
                ref_peaks=ref_peaks,
                affine_transform_settings=affine_transform_settings,
                ndim=ndim,
                verbose=False,
            )

            # Compose approx_transform with correction and apply to full volume
            composed_transform = approx_transform @ inv_transform
            mov_reg_optimized = (
                composed_transform.to_ants()
                .apply_to_image(mov_ants, reference=ref_ants)
                .numpy()
            )

            # Re-detect peaks and score
            mov_peaks_opt, ref_peaks_opt = peaks_from_beads(
                mov=mov_reg_optimized,
                ref=ref,
                mov_peaks_settings=beads_match_settings.source_peaks_settings,
                ref_peaks_settings=beads_match_settings.target_peaks_settings,
                verbose=False,
            )
            if mov_peaks_opt is None or ref_peaks_opt is None:
                continue

            score = overlap_score(
                mov_peaks=mov_peaks_opt,
                ref_peaks=ref_peaks_opt,
                radius=score_radius,
                verbose=False,
            )

            if np.isnan(score):
                continue

            score_cache[cache_key] = score
            transform_cache[cache_key] = composed_transform

            if verbose:
                click.echo(f"  {trial_params} -> matches={len(matches)}, score={score:.4f}")

            if score > best_score:
                best_score = score
                best_settings = trial_settings
                best_transform = composed_transform

        except Exception as e:
            if verbose:
                click.echo(f"  {trial_params} -> failed: {e}")
            continue

    click.echo(
        f"Grid search best score: {best_score:.4f} "
        f"({len(score_cache)} distinct match sets scored)"
    )
    if verbose:
        click.echo(f"Best settings: {best_settings}")

    return best_settings, best_transform, best_score


def overlap_score(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    radius: int = 6,
    verbose: bool = False,
) -> float:
    """
    Compute the overlap fraction between two sets of bead peaks.

    For each reference peak, checks whether any moving peak falls within a
    spherical neighborhood of the given radius (using a KDTree). The score is
    the fraction of reference peaks that have at least one nearby moving peak,
    normalized by the smaller peak set size.

    Parameters
    ----------
    mov_peaks : ArrayLike
        (N_mov, D) array of moving bead coordinates (z, y, x).
    ref_peaks : ArrayLike
        (N_ref, D) array of reference bead coordinates (z, y, x).
    radius : int
        Spherical neighborhood radius in voxels for overlap counting.
    verbose : bool
        If True, prints peak counts and overlap statistics.

    Returns
    -------
    float
        Overlap fraction in [0, 1]. Returns np.nan if either peak set is empty.
    """
    if len(mov_peaks) == 0 or len(ref_peaks) == 0:
        click.echo("No peaks found, returning nan metrics")
        return np.nan

    # ---- Overlap counting using KDTree ----
    mov_tree = cKDTree(mov_peaks)

    ref_peaks_mask = np.zeros(len(ref_peaks), dtype=bool)
    mov_peaks_mask = np.zeros(len(mov_peaks), dtype=bool)

    for i, p in enumerate(ref_peaks):
        idx = mov_tree.query_ball_point(p, r=radius)
        if idx:
            ref_peaks_mask[i] = True
            mov_peaks_mask[idx] = True

    peaks_overlap_count = int(ref_peaks_mask.sum())

    # ---- Overlap fraction ----
    peaks_overlap_fraction = peaks_overlap_count / max(min(len(mov_peaks), len(ref_peaks)), 1)

    if verbose:
        click.echo(f"Mov peaks: {len(mov_peaks)}")
        click.echo(f"Ref peaks: {len(ref_peaks)}")
        click.echo(f"Peaks overlap count: {peaks_overlap_count}")
        click.echo(f"Peaks overlap fraction: {peaks_overlap_fraction}")

    return peaks_overlap_fraction


def score_transform(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
) -> float:
    """Overlap score of a transform, by the same path that produced the run's own scores.

    Deliberately duplicates the warp / re-detect / overlap_score sequence from
    optimize_transform's step 3 rather than calling optimize_transform, because that function
    also refines the transform -- which would score something other than what was passed in.
    Returns nan when the warp leaves too few detectable beads.
    """
    warped = (
        transform.to_ants()
        .apply_to_image(ants.from_numpy(np.asarray(mov)), reference=ants.from_numpy(np.asarray(ref)))
        .numpy()
    )
    mov_peaks, ref_peaks = peaks_from_beads(
        mov=warped,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=False,
    )
    if mov_peaks is None or ref_peaks is None:
        return float("nan")
    return overlap_score(
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        radius=beads_match_settings.qc_settings.score_centroid_mask_radius,
        verbose=False,
    )


def repair_flagged_timepoints(
    mov_tzyx: da.Array,
    ref_tzyx: da.Array,
    transforms: list,
    scores: "pd.DataFrame",
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    output_transforms_path: Path,
    mode: str = "registration",
    verbose: bool = False,
) -> tuple[list, "pd.DataFrame"]:
    """Re-estimate flagged timepoints from their neighbours' transforms.

    This is the second half of the fallback, and it has to live here rather than inside
    estimate() for a structural reason: it seeds from t-1 AND t+1, and t+1 does not exist
    yet while t is being estimated -- in independent mode the timepoints are running in
    parallel shards. So it can only run once the whole series is known.

    That ordering is also what makes it strictly stronger than propagation's own fallback,
    which can only ever reach backwards to t-1. A timepoint that failed because the sample
    jumped between t-1 and t is often perfectly reachable from t+1.

    It covers a different failure class from the sweep. The sweep re-tunes the matching
    parameters and helps where the transform is already in the right basin but the
    correspondence is suboptimal -- measured yield +0.058 on timepoints scoring 0.72-0.78,
    and +0.006 on those below 0.72. Reseeding is what addresses a transform in the WRONG
    basin, which no amount of re-weighting the cost matrix fixes: at one real timepoint that
    had no usable transform at all, propagation gave 0.429 while a reseed reached 0.778.

    Each candidate seed is run through the full estimate() rather than a bare
    optimize_transform, so a reseed also gets the spectral cascade and the sweep fallback.
    Candidates are scored and accepted only if they strictly beat what is already there, so
    this pass cannot make a run worse.

    Which timepoints to repair comes from the run's own adaptive threshold, so the gate is
    the same median-2*MAD line the QC report flags on rather than a second fixed number.

    Returns the possibly-updated transforms and scores; the repaired .npy files are rewritten
    in place so the on-disk record matches what is returned.
    """
    settings = beads_match_settings.repair_pass_settings
    # .copy(): to_numpy can hand back a read-only view onto the DataFrame's own buffer when
    # the dtype already matches, and this array is written to as repairs are accepted.
    score_col = scores["quality_score"].to_numpy(dtype=float).copy()
    n_t = len(score_col)

    try:
        flags = flag_timepoints(score_col)
    except ValueError:
        click.echo("Repair pass skipped: no finite scores to flag against.")
        return transforms, scores
    flagged = [int(t) for t in flags.loc[flags["flagged"], "t"]]
    if not flagged:
        click.echo("Repair pass: nothing flagged, nothing to repair.")
        return transforms, scores

    click.echo(
        f"Repair pass: {len(flagged)} of {n_t} timepoints flagged "
        f"(adaptive line {flags.attrs['adaptive_line']:.3f}) -> {flagged}"
    )
    if settings.max_timepoints is not None and len(flagged) > settings.max_timepoints:
        # Announced rather than silently truncated: a capped run that says nothing reads as
        # "everything was repaired".
        worst = sorted(flagged, key=lambda t: np.nan_to_num(score_col[t], nan=-1.0))
        dropped = sorted(worst[settings.max_timepoints :])
        flagged = sorted(worst[: settings.max_timepoints])
        click.echo(
            f"  capped at max_timepoints={settings.max_timepoints}; repairing the worst "
            f"{len(flagged)} and LEAVING {len(dropped)} unrepaired: {dropped}"
        )

    seed_matrix = np.asarray(affine_transform_settings.approx_transform, dtype=float)
    flagged_set = set(flagged)
    good_median = float(np.nanmedian(score_col))
    log = []

    for t in flagged:
        before = score_col[t] if np.isfinite(score_col[t]) else -1.0

        # Only non-flagged neighbours are worth seeding from; a flagged neighbour is by
        # definition not a transform we trust.
        candidates = []
        for name, idx in (("t-1", t - 1), ("t+1", t + 1)):
            if 0 <= idx < n_t and idx not in flagged_set and transforms[idx] is not None:
                candidates.append((name, np.asarray(transforms[idx], dtype=float)))
        if settings.try_config_seed:
            candidates.append(("config_seed", seed_matrix))

        mov_t = np.asarray(mov_tzyx[t])
        ref_t = np.asarray(ref_tzyx[t])

        best_name, best_matrix, best_score = "unchanged", None, before
        for name, seed in candidates:
            trial_ats = affine_transform_settings.model_copy(deep=True)
            trial_ats.approx_transform = seed.tolist()
            # Reseeding is the point, so propagation must be off for the trial or estimate()
            # would reintroduce the very previous-timepoint transform being replaced.
            trial_ats.use_prev_t_transform = False
            try:
                candidate_transform = estimate(
                    mov=mov_t,
                    ref=ref_t,
                    beads_match_settings=beads_match_settings,
                    affine_transform_settings=trial_ats,
                    verbose=False,
                )
                if candidate_transform is None:
                    continue
                # estimate() persists its score as a sidecar rather than returning it, so
                # score here -- and scoring the returned transform directly is what makes
                # this comparable to `before` in the first place.
                candidate_score = score_transform(
                    candidate_transform, mov_t, ref_t, beads_match_settings
                )
            except Exception as e:  # noqa: BLE001
                click.echo(f"  t={t} seed {name} failed: {type(e).__name__}: {e}")
                continue
            if np.isfinite(candidate_score) and candidate_score > best_score:
                best_name, best_matrix, best_score = name, candidate_transform, candidate_score
            # Short-circuit: once a reseed is as good as a typical timepoint in this run,
            # further candidates are unlikely to matter and each costs a full estimate().
            if best_score >= good_median:
                break

        if best_matrix is not None:
            matrix = np.asarray(
                best_matrix.to_list() if hasattr(best_matrix, "to_list") else best_matrix,
                dtype=float,
            )
            transforms[t] = matrix
            np.save(output_transforms_path / f"{t}.npy", matrix)
            score_col[t] = best_score
            scores.loc[scores["t"] == t, "quality_score"] = best_score
            if "fell_back_to_seed" in scores:
                scores.loc[scores["t"] == t, "fell_back_to_seed"] = False
        click.echo(
            f"  t={t:4d} {before:.3f} -> {best_score:.3f} via {best_name}"
            + ("" if best_matrix is not None else "  (kept original)")
        )
        log.append(
            {
                "t": t,
                "before": float(before),
                "after": float(best_score),
                "source": best_name,
                "candidates_tried": [n for n, _ in candidates],
            }
        )

    improved = sum(1 for r in log if r["after"] > r["before"] + 1e-9)
    click.echo(f"Repair pass: {improved} of {len(log)} flagged timepoints improved")
    (output_transforms_path.parent / "repair_log.json").write_text(
        json.dumps(
            {
                "adaptive_line": flags.attrs["adaptive_line"],
                "median": flags.attrs["median"],
                "mad": flags.attrs["mad"],
                "n_flagged": len(log),
                "n_improved": improved,
                "repairs": log,
            },
            indent=2,
        )
    )
    return transforms, scores


def estimate_tczyx(
    mov_tczyx: da.Array,
    ref_tczyx: da.Array,
    mov_channel_index: int,
    ref_channel_index: int = None,
    beads_match_settings: BeadsMatchSettings = None,
    affine_transform_settings: AffineTransformSettings = None,
    verbose: bool = False,
    cluster: bool = False,
    sbatch_filepath: Path = None,
    output_folder_path: Path = None,
    ref_voxel_size: tuple[float, float, float] = (0.174, 0.1494, 0.1494),
    mov_voxel_size: tuple[float, float, float] = (0.174, 0.1494, 0.1494),
    mode: Literal["registration", "stabilization"] = "registration",
) -> list[Transform]:
    """
    Estimate beads-based registration transforms for all timepoints.

    Orchestrates the full registration pipeline: computes the approximate transform
    (if needed), then estimates per-timepoint transforms either sequentially with
    propagation or independently via SLURM, depending on settings.

    Parameters
    ----------
    mov_tczyx : da.Array
        Moving data (T, C, Z, Y, X).
    ref_tczyx : da.Array
        Reference data (T, C, Z, Y, X).
    mov_channel_index : int
        Channel index in the moving data containing beads.
    ref_channel_index : int, optional
        Channel index in the reference data. Ignored in stabilization mode.
    beads_match_settings : BeadsMatchSettings
        Settings for bead detection, matching, filtering, and QC.
    affine_transform_settings : AffineTransformSettings
        Settings for transform type, initial approx transform, and propagation.
    verbose : bool
        If True, prints detailed logs.
    cluster : bool
        If True, submits jobs to SLURM; otherwise runs locally.
    sbatch_filepath : Path, optional
        Path to sbatch file for custom SLURM parameters.
    output_folder_path : Path
        Directory to save per-timepoint transforms and logs.
    ref_voxel_size : tuple[float, float, float]
        Reference voxel size (Z, Y, X) in microns.
    mov_voxel_size : tuple[float, float, float]
        Moving voxel size (Z, Y, X) in microns.
    mode : {"registration", "stabilization"}
        "registration": align two different channels.
        "stabilization": align one channel to itself over time.

    Returns
    -------
    list[Transform]
        One 4x4 affine transform per timepoint.
    """
    mov_tzyx = mov_tczyx[:, mov_channel_index]
    if mode == "stabilization":
        ref_tzyx = mov_tzyx
    elif mode == "registration":
        ref_tzyx = ref_tczyx[:, ref_channel_index]

    output_transforms_path = output_folder_path / "xyz_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)

    if affine_transform_settings.compute_approx_transform:
        approx_transform = get_aprox_transform(
            mov_shape=mov_tzyx.shape[-3:],
            ref_shape=ref_tzyx.shape[-3:],
            pre_affine_90degree_rotation=-1,
            pre_affine_fliplr=False,
            verbose=verbose,
            ref_voxel_size=ref_voxel_size,
            mov_voxel_size=mov_voxel_size,
        )
        click.echo("Computed approx transform: ", approx_transform)
        affine_transform_settings.approx_transform = approx_transform.to_list()

    if affine_transform_settings.use_prev_t_transform:
        estimate_with_propagation(
            mov_tzyx=mov_tzyx,
            ref_tzyx=ref_tzyx,
            beads_match_settings=beads_match_settings,
            affine_transform_settings=affine_transform_settings,
            verbose=verbose,
            output_folder_path=output_transforms_path,
            mode=mode,
        )
    else:
        estimate_independently(
            mov_tzyx=mov_tzyx,
            ref_tzyx=ref_tzyx,
            beads_match_settings=beads_match_settings,
            affine_transform_settings=affine_transform_settings,
            verbose=verbose,
            output_folder_path=output_transforms_path,
            cluster=cluster,
            sbatch_filepath=sbatch_filepath,
            mode=mode,
        )

    transforms = load_transforms(output_transforms_path, mov_tzyx.shape[0], verbose)

    # Surface the per-timepoint quality score. It was previously only echoed as the run
    # went, so there was no way to see where a run degraded without grepping the log --
    # and no way at all for the independent arm, whose timepoints each log to their own
    # submitit file.
    scores = load_quality_scores(output_transforms_path, mov_tzyx.shape[0])

    # Second half of the fallback. Runs here, after the whole series is known, because it
    # seeds from both neighbours and t+1 does not exist while t is being estimated. The
    # sweep half already ran per-timepoint inside estimate(); between them they cover the
    # two distinct failure modes, and each accepts its result only on a strict score win.
    if beads_match_settings.repair_pass_settings.mode == "on_flagged":
        transforms, scores = repair_flagged_timepoints(
            mov_tzyx=mov_tzyx,
            ref_tzyx=ref_tzyx,
            transforms=transforms,
            scores=scores,
            beads_match_settings=beads_match_settings,
            affine_transform_settings=affine_transform_settings,
            output_transforms_path=output_transforms_path,
            mode=mode,
            verbose=verbose,
        )

    scores.to_csv(output_folder_path / "quality_scores.csv", index=False)
    plot_quality_scores(
        scores,
        output_folder_path / "translation_plots" / "beads_quality_score.png",
        score_threshold=beads_match_settings.qc_settings.score_threshold,
    )
    # Full QC report: adaptive flagging, transform plausibility, smoothness. Reports
    # only -- nothing is modified, because interpolating weak timepoints was measured to
    # make them worse.
    try:
        write_qc_report(
            output_dir=output_folder_path,
            scores=scores["quality_score"].to_numpy(),
            transforms=[t for t in transforms if t is not None],
        )
    except Exception as e:  # noqa: BLE001
        click.echo(f"QC report skipped: {type(e).__name__}: {e}")

    scored = scores.dropna(subset=["quality_score"])
    if len(scored):
        n_low = int(
            (scored["quality_score"] < beads_match_settings.qc_settings.score_threshold).sum()
        )
        click.echo(
            f"Quality score: median {scored['quality_score'].median():.3f}, "
            f"{n_low} of {len(scored)} timepoints below "
            f"{beads_match_settings.qc_settings.score_threshold}, "
            f"{int(scores['fell_back_to_seed'].sum())} fell back to the seed"
        )

    return transforms


def estimate_with_propagation(
    mov_tzyx: da.Array,
    ref_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    mode: Literal["registration", "stabilization"] = "registration",
) -> None:
    """
    Estimate transforms sequentially, propagating each result to the next timepoint.

    Processes timepoints in order (t=0, 1, 2, ...). After each timepoint, the
    estimated transform is used as the approximate transform for the next timepoint.
    This is useful when drift is gradual and cumulative, as each timepoint starts
    from a better initial guess.

    Parameters
    ----------
    mov_tzyx : da.Array
        Moving volume (T, Z, Y, X).
    ref_tzyx : da.Array
        Reference volume (T, Z, Y, X).
    beads_match_settings : BeadsMatchSettings
        Settings for bead detection, matching, and filtering.
    affine_transform_settings : AffineTransformSettings
        Settings for transform type and initial approximate transform.
        Modified in-place: approx_transform is updated after each timepoint.
    verbose : bool
        If True, prints progress for each timepoint.
    output_folder_path : Path
        Directory to save per-timepoint transform .npy files.
    mode : {"registration", "stabilization"}
        "registration": align moving to reference channel.
        "stabilization": align moving channel to itself over time.
    """
    # The static seed from the config. Kept as the `user_transform` competition arm for
    # every timepoint, so a run that has drifted still gets a chance to snap back to it.
    config_seed = affine_transform_settings.approx_transform
    # The most recent transform that actually succeeded. This -- not the config seed --
    # is what a failed timepoint falls back to. Resetting to the config seed makes a
    # single failure self-sustaining: the seed is a deliberately coarse initialisation
    # (measured at overlap score 0.000 on real bead data), so the next timepoint starts
    # from somewhere the matcher cannot recover from, fails in turn, and the failure
    # walks forward as a cluster.
    last_good_transform = config_seed

    T, _, _, _ = mov_tzyx.shape
    for t in range(T):
        if mode == "stabilization" and t == 0:
            continue

        # Resume: a timepoint already on disk is reloaded rather than recomputed, and its
        # transform still propagates forward so the chain stays identical to an
        # uninterrupted run. Without this, any interruption restarts at t=0 -- a 24 h
        # walltime cut two 800-timepoint estimates at ~75% complete and would have
        # discarded 23 h of finished work, even though every finished timepoint was
        # already saved as {t}.npy.
        existing = output_folder_path / f"{t}.npy" if output_folder_path else None
        if existing is not None and existing.exists():
            try:
                last_good_transform = np.load(existing).tolist()
                affine_transform_settings.approx_transform = last_good_transform
                if verbose:
                    click.echo(f"Timepoint {t} already estimated, reusing {existing.name}")
                continue
            except (OSError, ValueError):
                # A truncated file from a job killed mid-write: recompute it.
                click.echo(f"Timepoint {t} transform unreadable, recomputing")

        if np.sum(mov_tzyx[t]) == 0 or np.sum(ref_tzyx[t]) == 0:
            click.echo(f"Timepoint {t} has no data, skipping")
            # approx_transform stays on the last good one, so a blank frame costs only
            # itself rather than derailing every timepoint after it.
        else:
            approx_transform = estimate_tzyx(
                t_idx=t,
                mov_tzyx=mov_tzyx,
                ref_tzyx=ref_tzyx,
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
                verbose=verbose,
                output_folder_path=output_folder_path,
                mode=mode,
                user_transform=config_seed,
            )

            if approx_transform is not None:
                last_good_transform = approx_transform.to_list()
            elif verbose:
                click.echo(
                    f"Timepoint {t} produced no transform; propagating the last "
                    "successful transform rather than the config seed."
                )
            affine_transform_settings.approx_transform = last_good_transform


def estimate_independently(
    mov_tzyx: da.Array,
    ref_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    cluster: str = "local",
    sbatch_filepath: Path = None,
    mode: Literal["registration", "stabilization"] = "registration",
) -> None:
    """
    Estimate transforms for all timepoints independently via SLURM.

    Each timepoint is submitted as an independent job using submitit. All jobs
    use the same approximate transform as their starting point (no propagation).
    Suitable for large datasets where timepoints can be processed in parallel.

    Parameters
    ----------
    mov_tzyx : da.Array
        Moving volume (T, Z, Y, X).
    ref_tzyx : da.Array
        Reference volume (T, Z, Y, X).
    beads_match_settings : BeadsMatchSettings
        Settings for bead detection, matching, and filtering.
    affine_transform_settings : AffineTransformSettings
        Settings for transform type and initial approximate transform.
    verbose : bool
        If True, prints progress for each timepoint.
    output_folder_path : Path
        Directory to save per-timepoint transform .npy files.
    cluster : str
        Submitit cluster backend ('local', 'slurm', etc.).
    sbatch_filepath : Path, optional
        Path to sbatch file for custom SLURM parameters.
    mode : {"registration", "stabilization"}
        "registration": align moving to reference channel.
        "stabilization": align moving channel to itself over time.
    """
    T, Z, Y, X = mov_tzyx.shape
    _, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, 2, Z, Y, X), ram_multiplier=5, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_registration",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 30,
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    slurm_out_path = output_folder_path.parent / "slurm_output"
    slurm_out_path.mkdir(exist_ok=True)

    # Submitit executor
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)
    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")

    # Submit jobs
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for t in range(T):
            job = executor.submit(
                estimate_tzyx,
                t_idx=t,
                mov_tzyx=mov_tzyx,
                ref_tzyx=ref_tzyx,
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
                verbose=verbose,
                output_folder_path=output_folder_path,
                mode=mode,
            )
            jobs.append(job)

    # Save job IDs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = slurm_out_path / f"job_ids_{timestamp}.log"
    with open(log_path, "w") as log_file:
        for job in jobs:
            log_file.write(f"{job.job_id}\n")

    wait_for_jobs_to_finish(jobs)


def peaks_from_beads(
    mov: da.Array,
    ref: da.Array,
    mov_peaks_settings: DetectPeaksSettings,
    ref_peaks_settings: DetectPeaksSettings,
    verbose: bool = False,
    mask_path: Path = None,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Detect peaks in moving and reference channels using the detect_peaks function.

    Parameters
    ----------
    mov : da.Array
        (Z, Y, X) array of the moving channel (Dask array).
    ref : da.Array
        (Z, Y, X) array of the reference channel (Dask array).
    mov_peaks_settings : DetectPeaksSettings
        Settings for the moving peaks.
    ref_peaks_settings : DetectPeaksSettings
        Settings for the reference peaks.
    verbose : bool
        If True, prints detailed logs during the process.
    mask_path : Path
        Path to the mask file.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        Tuple of (mov_peaks, ref_peaks).
    """
    if verbose:
        click.echo("Detecting beads in moving dataset")
    # TODO: detecte peaks in the zyx space, use skimage.feature.peak_local_max for 2D
    mov_peaks = detect_peaks(
        mov,
        block_size=mov_peaks_settings.block_size,
        threshold_abs=mov_peaks_settings.threshold_abs,
        nms_distance=mov_peaks_settings.nms_distance,
        min_distance=mov_peaks_settings.min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo("Detecting beads in reference dataset")
    # TODO: detecte peaks in the zyx space, use skimage.feature.peak_local_max for 2D
    ref_peaks = detect_peaks(
        ref,
        block_size=ref_peaks_settings.block_size,
        threshold_abs=ref_peaks_settings.threshold_abs,
        nms_distance=ref_peaks_settings.nms_distance,
        min_distance=ref_peaks_settings.min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo(f"Total of peaks in moving dataset: {len(mov_peaks)}")
        click.echo(f"Total of peaks in reference dataset: {len(ref_peaks)}")

    if len(mov_peaks) < 2 or len(ref_peaks) < 2:
        click.echo("Not enough beads detected")
        # Return a pair, not a bare None. The annotation promises a 2-tuple and every
        # call site unpacks into two names, so a bare `return` raised
        # "cannot unpack non-iterable NoneType object" and killed the whole run on the
        # first timepoint with too few beads -- it took out a 3 h estimate at t=92 of
        # 288. The callers already test `if mov_peaks is None`, so they were written
        # expecting this contract; the unpacking simply crashed before those guards ran.
        return None, None
    if mask_path is not None:
        click.echo("Filtering peaks with mask")
        with open_ome_zarr(mask_path) as mask_ds:
            mask_load = np.asarray(mask_ds.data[0, 0])

        # filter the peaks with the mask
        # Keep only peaks whose (y, x) column is clean across all Z slices
        ref_peaks_filtered = []
        for peak in ref_peaks:
            z, y, x = peak.astype(int)
            if (
                0 <= y < mask_load.shape[1]
                and 0 <= x < mask_load.shape[2]
                and not mask_load[:, y, x].any()  # True if all Z are clean at (y, x)
            ):
                ref_peaks_filtered.append(peak)
        ref_peaks = np.array(ref_peaks_filtered)
    return mov_peaks, ref_peaks


def matches_from_beads(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
    verbose: bool = False,
) -> ArrayLike:
    """
    Find bead correspondences between moving and reference peak sets.

    Supports two matching algorithms:
    - "hungarian": Builds k-NN graphs for both peak sets, computes a cost matrix
      based on position distance and edge consistency, then solves the assignment
      problem with the Hungarian algorithm.
    - "match_descriptor": Uses scikit-image's descriptor matching on peak positions.

    After matching, applies geometric consistency filters (distance quantiles,
    direction threshold, angle threshold) to remove outliers.

    Parameters
    ----------
    mov_peaks : ArrayLike
        (N, D) array of moving peak coordinates (D = 2 or 3).
    ref_peaks : ArrayLike
        (M, D) array of reference peak coordinates.
    beads_match_settings : BeadsMatchSettings
        Settings controlling the matching algorithm, graph construction,
        cost matrix weights, and post-match filtering.
    verbose : bool
        If True, prints matching settings and match count.

    Returns
    -------
    ArrayLike
        (K, 2) array of matched index pairs [mov_idx, ref_idx].
    """
    if verbose:
        click.echo(f"Getting matches from beads with settings: {beads_match_settings}")

    if beads_match_settings.algorithm == "match_descriptor":
        mov_graph = Graph.from_nodes(mov_peaks)
        ref_graph = Graph.from_nodes(ref_peaks)

        match_descriptor_settings = beads_match_settings.match_descriptor_settings
        matcher = GraphMatcher(
            algorithm="descriptor",
            cross_check=match_descriptor_settings.cross_check,
            max_ratio=match_descriptor_settings.max_ratio,
            metric=match_descriptor_settings.distance_metric,
            verbose=verbose,
        )

        matches = matcher.match(mov_graph, ref_graph)

    elif beads_match_settings.algorithm == "hungarian":
        hungarian_match_settings = beads_match_settings.hungarian_match_settings
        edge_graph_settings = hungarian_match_settings.edge_graph_settings
        # Honour the configured graph mode. This was hardcoded to "knn", so `method` was
        # silently ignored and "radius" / "full" did nothing. That matters here rather
        # than being cosmetic: the two clouds have very different densities (~20 moving
        # against ~54 reference peaks in the same volume), so a fixed k spans a 1.6x
        # larger physical neighbourhood on the sparse side at every k tested. The
        # edge-length cost then compares descriptors measured at different scales. A
        # radius graph uses the same physical scale on both sides -- measured on synthetic
        # clouds that took the edge-scale ratio from 1.46 to 0.99 and recall from 0.85 to
        # 0.91. EdgeGraphSettings already validates and defaults k / radius per method.
        graph_kwargs = {"mode": edge_graph_settings.method}
        if edge_graph_settings.method == "knn":
            graph_kwargs["k"] = edge_graph_settings.k
        elif edge_graph_settings.method == "radius":
            graph_kwargs["radius"] = edge_graph_settings.radius
        mov_graph = Graph.from_nodes(mov_peaks, **graph_kwargs)
        ref_graph = Graph.from_nodes(ref_peaks, **graph_kwargs)

        matcher = GraphMatcher(
            algorithm="hungarian",
            weights=hungarian_match_settings.cost_matrix_settings.weights,
            cost_threshold=hungarian_match_settings.cost_threshold,
            cross_check=hungarian_match_settings.cross_check,
            max_ratio=hungarian_match_settings.max_ratio,
            verbose=verbose,
        )

        matches = matcher.match(mov_graph, ref_graph)

    elif beads_match_settings.algorithm == "spectral":
        spectral_match_settings = beads_match_settings.spectral_match_settings
        # No graph features are used: spectral matching works from the raw point
        # coordinates via pairwise distances, so the graph is just a node container and
        # k is irrelevant here.
        mov_graph = Graph.from_nodes(mov_peaks)
        ref_graph = Graph.from_nodes(ref_peaks)

        matcher = GraphMatcher(
            algorithm="spectral",
            spectral_sigma=spectral_match_settings.sigma,
            spectral_rel_cut=spectral_match_settings.rel_cut,
            spectral_max_iter=spectral_match_settings.max_iter,
            verbose=verbose,
        )

        matches = matcher.match(mov_graph, ref_graph)

    else:
        raise ValueError(f"Unknown matching algorithm: {beads_match_settings.algorithm}")

    # Filter as part of the pipeline
    matches = matcher.filter_matches(
        matches,
        mov_graph,
        ref_graph,
        angle_threshold=beads_match_settings.filter_matches_settings.angle_threshold,
        min_distance_quantile=beads_match_settings.filter_matches_settings.min_distance_quantile,
        max_distance_quantile=beads_match_settings.filter_matches_settings.max_distance_quantile,
        direction_threshold=beads_match_settings.filter_matches_settings.direction_threshold,
    )

    if verbose:
        click.echo(f"Total of matches: {len(matches)}")

    return matches


def transform_from_matches(
    matches: ArrayLike,
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    affine_transform_settings: AffineTransformSettings,
    ndim: int = 3,
    verbose: bool = False,
) -> tuple[Transform, Transform]:
    """
    Estimate the affine transformation matrix between source and target channels.

    Based on detected bead matches at a specific timepoint.

    Parameters
    ----------
    matches : ArrayLike
        (n, 2) array of matches.
    mov_peaks : ArrayLike
        (n, 2) array of moving peaks.
    ref_peaks : ArrayLike
        (n, 2) array of reference peaks.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    ndim: int
        Number of dimensions.
    verbose : bool
        If True, prints detailed logs during the process.

    Returns
    -------
    tuple[Transform, Transform]
        Tuple of forward and inverse transforms.
    """
    if verbose:
        click.echo(f"Estimating transform with settings: {affine_transform_settings}")
    # Detect dimensionality from peaks
    if ndim not in (2, 3):
        raise ValueError(f"Peaks must be 2D or 3D, got {ndim}D")

    # Create appropriate transform
    if affine_transform_settings.transform_type == "affine":
        transform = AffineTransform(dimensionality=ndim)
    elif affine_transform_settings.transform_type == "euclidean":
        transform = EuclideanTransform(dimensionality=ndim)
    elif affine_transform_settings.transform_type == "similarity":
        transform = SimilarityTransform(dimensionality=ndim)
    else:
        raise ValueError(f"Unknown transform type: {affine_transform_settings.transform_type}")

    # Fit transform
    transform.estimate(mov_peaks[matches[:, 0]], ref_peaks[matches[:, 1]])

    inv_transform = Transform(matrix=transform.inverse.params)
    fwd_transform = Transform(matrix=transform.params)

    return fwd_transform, inv_transform


def estimate_tzyx(
    t_idx: int,
    mov_tzyx: da.Array,
    ref_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    mode: Literal["registration", "stabilization"] = "registration",
    user_transform: Transform = None,
) -> Transform:
    """
    Estimate the affine transform for a single timepoint.

    Extracts the 3D volumes for the given timepoint, sets up the reference
    depending on the mode (registration vs stabilization), and delegates to
    `estimate()` for iterative bead-based transform estimation.

    Parameters
    ----------
    t_idx : int
        Timepoint index to process.
    mov_tzyx : da.Array
        Moving volume (T, Z, Y, X).
    ref_tzyx : da.Array
        Reference volume (T, Z, Y, X). Ignored in stabilization mode.
    beads_match_settings : BeadsMatchSettings
        Settings for bead detection, matching, and filtering.
    affine_transform_settings : AffineTransformSettings
        Settings for transform type and initial approximate transform.
    verbose : bool
        If True, prints detailed logs during the process.
    output_folder_path : Path, optional
        Directory to save the transform as ``{t_idx}.npy``.
    mode : {"registration", "stabilization"}
        "registration": align moving to reference (different channels).
        "stabilization": align moving channel to itself over time,
        using t_reference setting ("first" or "previous").
    user_transform : Transform, optional
        Alternative initial transform to compete with the default on iteration 0.

    Returns
    -------
    Transform or None
        The estimated 4x4 affine transform, or None if estimation failed.
    """
    click.echo("........................................................................")
    click.echo(f"Processing timepoint: {t_idx}")

    (T, Z, Y, X) = mov_tzyx.shape

    if mode == "stabilization":
        click.echo("Performing stabilization, aka registration over time in the same file.")
        if affine_transform_settings.t_reference == "first":
            ref_tzyx = np.broadcast_to(mov_tzyx[0], (T, Z, Y, X)).copy()
        elif affine_transform_settings.t_reference == "previous":
            ref_tzyx = np.roll(mov_tzyx, shift=-1, axis=0)
            ref_tzyx[0] = mov_tzyx[0]
        else:
            raise ValueError(
                "Invalid reference. Please use 'first' or 'previous' as reference."
            )
    elif mode == "registration":
        click.echo("Performing registration between different files")
    mov_zyx = np.asarray(mov_tzyx[t_idx]).astype(np.float32)
    ref_zyx = np.asarray(ref_tzyx[t_idx]).astype(np.float32)

    if output_folder_path:
        output_folder_path.mkdir(parents=True, exist_ok=True)
        output_filepath = output_folder_path / f"{t_idx}.npy"
    else:
        output_filepath = None

    transform = estimate(
        mov=mov_zyx,
        ref=ref_zyx,
        beads_match_settings=beads_match_settings,
        affine_transform_settings=affine_transform_settings,
        verbose=verbose,
        output_filepath=output_filepath,
        user_transform=user_transform,
    )
    return transform


def optimize_transform(
    transform: Transform,
    mov: da.Array,
    ref: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    debug: bool = False,
) -> tuple[Transform, float]:
    """
    Refine a transform by bead matching and evaluate registration quality.

    Applies the current transform to the moving volume, detects beads in both
    the registered moving and reference volumes, matches them, estimates a
    correction transform, and composes it with the input transform. Returns
    the better of the two (original vs corrected) based on overlap score.

    Parameters
    ----------
    transform : Transform
        Current transform to refine (maps moving -> reference space).
    mov : ArrayLike
        Original (unregistered) moving volume (Z, Y, X).
    ref : ArrayLike
        Reference volume (Z, Y, X).
    beads_match_settings : BeadsMatchSettings
        Settings controlling peak detection, matching, and filtering.
    affine_transform_settings : AffineTransformSettings
        Settings for the transform type (affine/euclidean/similarity).
    verbose : bool
        If True, prints quality scores before and after optimization.
    debug : bool
        If True, prints detailed intermediate results (peaks, matches, transforms).

    Returns
    -------
    tuple[Transform, float]
        The best transform and its overlap score.
        Returns (None, -1) if not enough peaks or matches are found.
    """
    mov_ants = ants.from_numpy(mov)
    ref_ants = ants.from_numpy(ref)

    # Step 1: Score the current transform by applying it and measuring peak overlap
    if debug:
        click.echo("Step 1: Scoring current transform (before bead matching)...")
    mov_reg_approx = transform.to_ants().apply_to_image(mov_ants, reference=ref_ants).numpy()
    mov_peaks, ref_peaks = peaks_from_beads(
        mov=mov_reg_approx,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=debug,
    )
    if mov_peaks is None or ref_peaks is None:
        return None, -1

    quality_score_approx = overlap_score(
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        radius=beads_match_settings.qc_settings.score_centroid_mask_radius,
        verbose=debug,
    )

    # Step 2: Match beads and estimate a correction transform
    if debug:
        click.echo("Step 2: Matching beads to estimate correction transform...")
    matches = matches_from_beads(
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        beads_match_settings=beads_match_settings,
        verbose=debug,
    )

    if len(matches) < 3:
        click.echo("Not enough matches found, returning the current transform")
        return None, -1

    fwd_transform, inv_transform = transform_from_matches(
        matches=matches,
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        affine_transform_settings=affine_transform_settings,
        ndim=mov.ndim,
        verbose=debug,
    )
    composed_transform = transform @ inv_transform

    # Step 3: Score the composed (corrected) transform
    if debug:
        click.echo("Step 3: Scoring composed transform (after bead matching)...")
    mov_reg_optimized = (
        composed_transform.to_ants().apply_to_image(mov_ants, reference=ref_ants).numpy()
    )
    mov_peaks_optimized, ref_peaks_optimized = peaks_from_beads(
        mov=mov_reg_optimized,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=debug,
    )
    if mov_peaks_optimized is None or ref_peaks_optimized is None:
        # The correction warped the beads out of detectability, so the composed transform
        # cannot be scored and must not be accepted -- returning the pre-correction
        # transform and score leaves the caller exactly where it started, which is the
        # same outcome as a refinement that failed to improve.
        click.echo(
            "Composed transform left too few detectable beads; keeping the input transform."
        )
        return transform, quality_score_approx

    quality_score_optimized = overlap_score(
        mov_peaks=mov_peaks_optimized,
        ref_peaks=ref_peaks_optimized,
        radius=beads_match_settings.qc_settings.score_centroid_mask_radius,
        verbose=debug,
    )
    if debug:
        click.echo(f"Bead matches: {matches}")
        click.echo(f"Forward transform: {fwd_transform}")
        click.echo(f"Inverse transform: {inv_transform}")
        click.echo(f"Composed transform: {composed_transform}")

    if verbose:
        click.echo(f"Quality score before beads matching: {quality_score_approx}")
        click.echo(f"Quality score after beads matching: {quality_score_optimized}")

    if quality_score_optimized >= quality_score_approx:
        return composed_transform, quality_score_optimized
    else:
        return transform, quality_score_approx


def estimate(
    mov: da.Array,
    ref: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_filepath: Path = None,
    user_transform: Transform = None,
    debug: bool = False,
) -> Transform:
    """
    Estimate the best affine transformation between moving and reference volumes.

    Iteratively refines the transform by detecting beads, matching them, estimating
    a correction, and scoring the result. Supports an optional user-provided
    transform that competes with the computed one on the first iteration.

    Works for both 2D (Y, X) and 3D (Z, Y, X) arrays.

    Parameters
    ----------
    mov : ArrayLike
        Moving channel volume (Z, Y, X) or (Y, X).
    ref : ArrayLike
        Reference channel volume (Z, Y, X) or (Y, X).
    beads_match_settings : BeadsMatchSettings
        Settings for bead detection, matching, filtering, and QC iterations.
    affine_transform_settings : AffineTransformSettings
        Settings for transform type and initial approximate transform.
    verbose : bool
        If True, prints the best transform and score at the end.
    output_filepath : Path, optional
        If provided, saves the best transform matrix as a .npy file.
    user_transform : Transform, optional
        An alternative initial transform (e.g. from a previous timepoint).
        Tested on the first iteration; used if it scores better.
    debug : bool
        If True, passes debug flag to optimize_transform for detailed logging.

    Returns
    -------
    Transform
        The best transform found across all iterations. Falls back to the
        initial approximate transform if no valid optimization was found.
    """
    if _check_nan_n_zeros(mov) or _check_nan_n_zeros(ref):
        click.echo("Skipping: moving or reference data contains only NaN/zeros.")
        return

    initial_transform = Transform(
        matrix=np.asarray(affine_transform_settings.approx_transform)
    )
    transform = initial_transform

    current_iterations = 0
    qc_iterations = beads_match_settings.qc_settings.iterations
    transform_iter_dict = {}

    while current_iterations < qc_iterations:
        click.echo(
            f"Iteration {current_iterations + 1}/{qc_iterations}: "
            "optimizing transform via bead matching..."
        )
        optimized_transform, quality_score_optimized = optimize_transform(
            transform=transform,
            mov=mov,
            ref=ref,
            beads_match_settings=beads_match_settings,
            affine_transform_settings=affine_transform_settings,
            verbose=verbose,
            debug=debug,
        )
        transform_iter_dict[current_iterations] = {
            "transform": optimized_transform,
            "quality_score": quality_score_optimized,
        }
        if quality_score_optimized == 1:
            break
        transform = optimized_transform

        if user_transform is not None and current_iterations == 0:
            click.echo("Optimizing user transform:")
            user_transform = Transform(matrix=np.asarray(user_transform))
            optimized_transform_user, quality_score_optimized_user = optimize_transform(
                transform=user_transform,
                mov=mov,
                ref=ref,
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
                verbose=verbose,
                debug=debug,
            )

            if quality_score_optimized < quality_score_optimized_user:
                transform_iter_dict[current_iterations] = {
                    "transform": optimized_transform_user,
                    "quality_score": quality_score_optimized_user,
                }
                if quality_score_optimized_user == 1:
                    break
                transform = optimized_transform_user

        # Third arm: acquire the correspondence with spectral matching, then refine.
        # Opt-in, and gated on the other arms having done badly, because it costs a full
        # optimize_transform call and is only useful when the initial transform is wrong by
        # more than about one bead spacing -- exactly when the position-distance cost in
        # the Hungarian matcher starts pairing a bead with its neighbour instead of itself.
        # Spectral matching uses only relative distances, so it is unaffected by how wrong
        # the initial transform is.
        #
        # Whichever arm scores highest wins, so enabling this can never make the result
        # worse than leaving it off.
        best_so_far = transform_iter_dict[current_iterations]["quality_score"]
        spectral_mode = beads_match_settings.spectral_arm
        run_spectral = current_iterations == 0 and (
            spectral_mode == "always"
            or (
                spectral_mode == "on_low_score"
                and best_so_far < beads_match_settings.qc_settings.score_threshold
            )
        )
        if run_spectral:
            click.echo(f"Spectral arm ({spectral_mode}), current best {best_so_far:.3f}:")
            # Stage 1 -- ACQUIRE with spectral matching. This does not need the initial
            # transform to be close, because only relative distances are used.
            spectral_settings = beads_match_settings.model_copy(deep=True)
            spectral_settings.algorithm = "spectral"
            transform_spec, score_spec = optimize_transform(
                transform=initial_transform,
                mov=mov,
                ref=ref,
                beads_match_settings=spectral_settings,
                affine_transform_settings=affine_transform_settings,
                verbose=verbose,
                debug=debug,
            )

            # Stage 2 -- REFINE from spectral's transform with the CONFIGURED matcher.
            # This is the point of the cascade and not an optional extra: spectral
            # acquires the correspondence but is the less precise of the two once the
            # transform is already close, so handing its result back to the configured
            # matcher is what recovers the last part. Measured at one real failing
            # timepoint: seed 0.000 -> spectral 0.778 -> refined 0.882, which equals the
            # best transform any variant found there. Stopping after stage 1 would have
            # left 0.778 on the table.
            if transform_spec is not None:
                transform_refined, score_refined = optimize_transform(
                    transform=transform_spec,
                    mov=mov,
                    ref=ref,
                    beads_match_settings=beads_match_settings,
                    affine_transform_settings=affine_transform_settings,
                    verbose=verbose,
                    debug=debug,
                )
                # optimize_transform only accepts a step that improves its own score, so
                # a None here means refinement found nothing better -- keep stage 1.
                if transform_refined is not None and score_refined > score_spec:
                    transform_spec, score_spec = transform_refined, score_refined

            if transform_spec is not None and score_spec > best_so_far:
                click.echo(f"Spectral cascade wins: {best_so_far:.3f} -> {score_spec:.3f}")
                transform_iter_dict[current_iterations] = {
                    "transform": transform_spec,
                    "quality_score": score_spec,
                }
                transform = transform_spec

        if transform is None:
            break
        current_iterations += 1

    # get highest quality score
    best_quality_score = max(transform_iter_dict.values(), key=lambda x: x["quality_score"])
    best_transform = best_quality_score["transform"]

    # Fourth arm, last resort: re-tune the matching parameters for THIS timepoint. Every arm
    # above shares one parameter set across the whole timelapse, and a few timepoints are
    # simply not well served by it. Gated on score because it is the most expensive arm --
    # one ANTs warp plus peak re-detection per distinct match set.
    #
    # Runs after the loop, on the best of all arms, so it only ever fires where everything
    # else has already failed, and its result is accepted only if it strictly wins. It
    # therefore cannot make any timepoint worse than leaving it off.
    sweep = beads_match_settings.sweep_fallback_settings
    incumbent_score = best_quality_score["quality_score"]
    if sweep.mode == "on_low_score" and incumbent_score < sweep.score_threshold:
        click.echo(
            f"Sweep fallback: best arm scored {incumbent_score:.3f} < "
            f"{sweep.score_threshold}, grid-searching matching parameters for this timepoint"
        )
        try:
            _, swept_transform, swept_score = optimize_matches(
                mov=mov,
                ref=ref,
                approx_transform=initial_transform,
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
                param_grid=sweep.grid,
                verbose=verbose,
            )
        except Exception as e:  # noqa: BLE001
            # A fallback that can take down a whole timelapse is worse than no fallback.
            click.echo(f"Sweep fallback failed ({type(e).__name__}: {e}); keeping incumbent.")
            swept_transform, swept_score = None, -1.0

        if swept_transform is not None and swept_score > incumbent_score:
            click.echo(f"Sweep fallback wins: {incumbent_score:.3f} -> {swept_score:.3f}")
            best_transform = swept_transform
            best_quality_score = {
                "transform": swept_transform,
                "quality_score": swept_score,
            }
            transform_iter_dict["sweep_fallback"] = best_quality_score
        else:
            click.echo(
                f"Sweep fallback found nothing better than {incumbent_score:.3f}; "
                "keeping incumbent."
            )

    # Every optimisation attempt failed, so the coarse initial transform is all we have.
    # Recorded explicitly: the saved .npy is otherwise indistinguishable from a real fit,
    # and such a transform can still pass validate_transforms, which only checks
    # consistency against neighbouring timepoints -- and a propagated seed is perfectly
    # self-consistent.
    fell_back_to_seed = best_transform is None
    if fell_back_to_seed:
        best_transform = initial_transform
    if verbose:
        click.echo(f"Best transform: {best_transform}")
        click.echo(f"Best quality score: {best_quality_score['quality_score']}")
    if output_filepath:
        click.echo(f"Saving transform to {output_filepath}")
        np.save(output_filepath, best_transform.to_list())
        # Persist the score as a sidecar rather than returning it, so that estimate()
        # keeps the same signature across every registration method. Written to disk
        # because the independent arm runs each timepoint in its own submitit process,
        # so an in-memory accumulator would not survive.
        save_quality_score(
            Path(output_filepath).with_suffix(".score"),
            score=best_quality_score["quality_score"],
            fell_back_to_seed=fell_back_to_seed,
        )

    return best_transform
