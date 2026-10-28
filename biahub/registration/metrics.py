"""Alignment metrics: every measure of how well a transform brings `mov` onto `ref`.

The engine decides with some of them (a score per transform) and the evaluation
analyses with all of them (residuals in voxels or µm); both call them from here.

Two families, because the pipeline registers across modalities:

- Bead metrics (`bead_alignment_metrics`, `residual_score`) need beads visible in both
  channels. They complement `beads.overlap_score`, which is a bead-count ratio quantized
  at ~1/N per bead, with continuous residuals in voxels.
- Intensity metrics work on any pair of channels. Plain correlation is meaningless
  between, say, phase and fluorescence (the correct transform can score ~0.1), so
  `normalized_mutual_information` -- the cross-modal similarity ANTs' Mattes metric
  approximates -- is the choice when no beads are available, and
  `gradient_correlation` is a cheaper edge-based alternative.

Every function takes the engine's forward transform and warps with `Transform.apply`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from numpy.typing import ArrayLike
from scipy.ndimage import binary_erosion
from scipy.ndimage import shift as ndi_shift
from scipy.spatial import cKDTree
from skimage import filters

from biahub.core.nodes.solvers import mutual_nearest
from biahub.core.transform import Transform, polar, rotation_angles
from biahub.registration.estimators import ScoreFn
from biahub.registration.methods.intensity.phase_cross_corr import phase_cross_corr
from biahub.registration.methods.landmark.bead_matching import overlap_score, peaks_from_beads
from biahub.settings import BeadsMatchSettings


@dataclass
class BeadAlignmentMetrics:
    n_mov_peaks: int
    n_ref_peaks: int
    n_matched: int  # reference peaks with a warped moving peak within `radius`
    overlap: float  # beads.overlap_score
    median_residual: float  # voxels, over matched pairs; nan when none
    rms_residual: float

    def to_dict(self) -> dict:
        return asdict(self)


def bead_alignment_metrics(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
) -> BeadAlignmentMetrics | None:
    """Warp, re-detect beads, and measure how the two peak sets line up.

    Returns None when peaks cannot be detected on either side.
    """
    radius = beads_match_settings.qc_settings.score_centroid_mask_radius
    ref = np.asarray(ref)
    warped = transform.apply(np.asarray(mov), reference=ref)
    peaks = peaks_from_beads(
        mov=warped,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=False,
    )
    if peaks is None:
        return None
    mov_peaks, ref_peaks = (np.asarray(p, dtype=float) for p in peaks)
    if len(mov_peaks) == 0 or len(ref_peaks) == 0:
        return None
    distances, _ = cKDTree(mov_peaks).query(ref_peaks, distance_upper_bound=radius)
    matched = distances[np.isfinite(distances)]
    return BeadAlignmentMetrics(
        n_mov_peaks=int(len(mov_peaks)),
        n_ref_peaks=int(len(ref_peaks)),
        n_matched=int(len(matched)),
        overlap=float(overlap_score(mov_peaks, ref_peaks, radius=radius, verbose=False)),
        median_residual=float(np.median(matched)) if len(matched) else float("nan"),
        rms_residual=float(np.sqrt(np.mean(matched**2))) if len(matched) else float("nan"),
    )


def residual_score(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
) -> float:
    """Continuous bead score in [0, 1]: overlap weighted by how tightly matched beads sit.

    `overlap * (1 - median_residual / radius)`: the same quantity `overlap_score`
    measures, but a transform whose matched beads land 0.5 voxel off outscores one whose
    beads land 4 voxels off instead of tying with it. nan when no beads are detected.
    """
    metrics = bead_alignment_metrics(transform, mov, ref, beads_match_settings)
    if metrics is None or metrics.n_matched == 0:
        return float("nan") if metrics is None else 0.0
    radius = beads_match_settings.qc_settings.score_centroid_mask_radius
    return float(metrics.overlap * max(0.0, 1.0 - metrics.median_residual / radius))


def _overlap_pair(transform: Transform, mov: ArrayLike, ref: ArrayLike):
    ref = np.asarray(ref, dtype=np.float32)
    warped = transform.apply(np.asarray(mov, dtype=np.float32), reference=ref)
    mask = (warped != 0) & (ref != 0)
    return warped, ref, mask


class Prewarped:
    """Stands in for a transform when the moving volume is already warped onto the reference.

    Lets several metrics share one warp: each calls `transform.apply`, which hands the
    volume back unchanged.
    """

    def apply(self, moving, reference=None, **kwargs):
        return np.asarray(moving)


PREWARPED = Prewarped()


# Voxels a 3x3x3 filter (Sobel) reaches: one on each side.
SOBEL_FOOTPRINT = 1


def _away_from_padding(mask: np.ndarray, footprint: int) -> np.ndarray:
    """Return `mask` without the voxels a filter of `footprint` reaches from outside it.

    A filter run on the warped volume sees the edge of the empty padding (or of empty data)
    as a strong edge; those voxels compare an artefact, not content. The volume's own
    borders are not padding and are kept.
    """
    if footprint <= 0:
        return mask
    # every neighbour (26 in 3D, 8 in 2D): a 3x3x3 filter reaches diagonally too
    return binary_erosion(
        mask,
        structure=np.ones((3,) * mask.ndim, dtype=bool),
        iterations=footprint,
        border_value=1,
    )


def _footprint(prepare, *, sobel: bool = False) -> int:
    """Return how far the score's blocks (and any built-in Sobel) reach, in voxels."""
    return getattr(prepare, "footprint", 0) + (SOBEL_FOOTPRINT if sobel else 0)


def normalized_mutual_information(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    bins: int = 64,
    percentiles: tuple[float, float] = (0.5, 99.5),
    sobel: bool = False,
    prepare=None,
) -> float:
    """Compute normalized mutual information of the warped moving volume and the reference.

    2 * I(A; B) / (H(A) + H(B)) over the overlapping voxels, in [0, 1]; intensities are
    clipped to `percentiles` before binning so a few hot voxels do not collapse the
    histogram. Independent of the intensity relation between the two channels, which is
    what makes it usable across modalities; `sobel` compares edge magnitudes instead of
    intensities. nan when the volumes do not overlap. `prepare(warped, ref, overlap)`
    (the score's blocks) runs after the overlap is found, before comparing.
    """
    warped, ref, mask = _overlap_pair(transform, mov, ref)
    if mask.sum() < bins:
        return float("nan")
    if prepare is not None:
        warped, ref = prepare(warped, ref, mask)
    if sobel:
        # MI of edge magnitudes: what the fault detectors in registration-eval found most
        # reliable between fluorescence and virtual staining.
        warped, ref = filters.sobel(warped), filters.sobel(ref)
    mask = _away_from_padding(mask, _footprint(prepare, sobel=sobel))
    if mask.sum() < bins:
        return float("nan")
    a, b = warped[mask], ref[mask]
    a_lo, a_hi = np.percentile(a, percentiles)
    b_lo, b_hi = np.percentile(b, percentiles)
    if a_hi <= a_lo or b_hi <= b_lo:
        return float("nan")
    joint, _, _ = np.histogram2d(
        np.clip(a, a_lo, a_hi),
        np.clip(b, b_lo, b_hi),
        bins=bins,
        range=[[a_lo, a_hi], [b_lo, b_hi]],
    )
    p = joint / joint.sum()
    pa, pb = p.sum(axis=1), p.sum(axis=0)

    def entropy(q):
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())

    h_a, h_b, h_ab = entropy(pa), entropy(pb), entropy(p.ravel())
    if h_a + h_b == 0:
        return float("nan")
    return float(2.0 * (h_a + h_b - h_ab) / (h_a + h_b))


def gradient_correlation(
    transform: Transform, mov: ArrayLike, ref: ArrayLike, prepare=None
) -> float:
    """Pearson correlation of Sobel gradient magnitudes over the overlap, in [-1, 1].

    Edges exist in every modality even when intensities do not correlate; cheaper than
    mutual information and adequate when both channels show the same structures.
    """
    warped, ref, mask = _overlap_pair(transform, mov, ref)
    if mask.sum() < 2:
        return float("nan")
    if prepare is not None:
        warped, ref = prepare(warped, ref, mask)
    mask = _away_from_padding(mask, _footprint(prepare, sobel=True))
    if mask.sum() < 2:
        return float("nan")
    a = filters.sobel(warped)[mask]
    b = filters.sobel(ref)[mask]
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def residual_shift(transform: Transform, mov: ArrayLike, ref: ArrayLike) -> float:
    """Remaining misalignment after `transform`, in voxels, as a displacement not a similarity.

    Phase cross-correlation between the Sobel edges of the warped moving volume and the
    reference; the norm of the recovered shift. A displacement stays interpretable when
    similarity metrics are depressed by defocus or bleaching (registration-eval's
    `mch_eshift`), and 0 means aligned regardless of modality.
    """
    warped, ref, mask = _overlap_pair(transform, mov, ref)
    inside = _away_from_padding(mask, SOBEL_FOOTPRINT)
    if inside.sum() < 8:
        return float("nan")
    # edges of the content only: the padding's edge would dominate the correlation
    ref_edges = np.where(inside, filters.sobel(ref), 0).astype(np.float32)
    warped_edges = np.where(inside, filters.sobel(warped), 0).astype(np.float32)
    shift, _ = phase_cross_corr(ref_edges, warped_edges)
    return float(np.linalg.norm(shift))


def score_transform(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
) -> float:
    """Overlap score of a candidate transform, independent of any estimator's own scoring.

    Warps `mov` with `transform`, re-detects beads, and scores their overlap against
    `ref` -- the same sequence `optimize_transform` uses internally, duplicated here
    (rather than calling `optimize_transform`) because that function also refines the
    transform, which would score something other than what was passed in. Used as the
    `score_fn` a fallback pass (e.g. `registration.engine.repair`) needs to compare
    candidate seeds against each other and against the current transform.
    """
    warped = transform.apply(np.asarray(mov), reference=np.asarray(ref))
    peaks = peaks_from_beads(
        mov=warped,
        ref=np.asarray(ref),
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=False,
    )
    if peaks is None:
        return float("nan")
    mov_peaks, ref_peaks = peaks
    return overlap_score(
        mov_peaks=mov_peaks,
        ref_peaks=ref_peaks,
        radius=beads_match_settings.qc_settings.score_centroid_mask_radius,
        verbose=False,
    )


def correlation_score(
    transform: Transform,
    mov: np.ndarray,
    ref: np.ndarray,
    sobel_filter: bool = False,
    prepare=None,
) -> float:
    """Pearson correlation between `mov` warped by `transform` and `ref`, over their overlap.

    An intensity analogue of the bead overlap score: continuous in [-1, 1], nan when
    the warped volume and the reference do not overlap. Optionally compares Sobel
    magnitudes instead, for cross-modality pairs registered that way. `prepare(warped,
    ref, overlap)` (the score's blocks) runs after the overlap is found, before comparing.
    """
    ref = np.asarray(ref, dtype=np.float32)
    warped = transform.apply(np.asarray(mov, dtype=np.float32), reference=ref)
    mask = (warped != 0) & (ref != 0)
    if mask.sum() < 2:
        return float("nan")
    if prepare is not None:
        warped, ref = prepare(warped, ref, mask)
    if sobel_filter:
        warped, ref = filters.sobel(warped), filters.sobel(ref)
    mask = _away_from_padding(mask, _footprint(prepare, sobel=sobel_filter))
    if mask.sum() < 2:
        return float("nan")
    a, b = warped[mask], ref[mask]
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    if denominator == 0:
        return float("nan")
    return float((a * b).sum() / denominator)


def beads_score_fn(
    beads_match_settings: BeadsMatchSettings, metric: str = "overlap"
) -> ScoreFn:
    """Build a bead score function: overlap (production's ratio), residual, or mutual information."""
    if metric == "overlap":
        return lambda transform, mov, ref: score_transform(
            transform, mov, ref, beads_match_settings
        )
    if metric == "residual":
        return lambda transform, mov, ref: residual_score(
            transform, mov, ref, beads_match_settings
        )
    if metric == "mutual_information":
        return normalized_mutual_information
    raise ValueError(f"unknown score_metric {metric!r}")


# --- residuals: matched landmarks and image patches, in µm (used by the evaluation) ---

# Fewer matched points than this do not constrain an affine residual (12 parameters,
# fitted to noisy centres: DragonFly nuclei gave a scale of 0.86 from 8 points).
MIN_POINTS_FOR_AFFINE = 10


def displacement_stats(diff_vox: ArrayLike, voxel_um: ArrayLike) -> dict:
    """Return statistics of matched residual displacements (N x 3, voxels, ZYX) in µm."""
    diff = np.asarray(diff_vox, dtype=float).reshape(-1, 3) * np.asarray(voxel_um, float)
    if len(diff) == 0:
        return {"n": 0}
    dist = np.linalg.norm(diff, axis=1)
    median_zyx = np.median(diff, axis=0)
    return {
        "n": int(len(diff)),
        "median_um": float(np.median(dist)),
        "p90_um": float(np.percentile(dist, 90)),
        "median_zyx_um": median_zyx.tolist(),
        "mad_zyx_um": np.median(np.abs(diff - median_zyx), axis=0).tolist(),
        "rms_zyx_um": np.sqrt((diff**2).mean(axis=0)).tolist(),
    }


def residual_affine(points_vox: ArrayLike, diff_vox: ArrayLike, voxel_um: ArrayLike) -> dict:
    """Fit residual displacement = a + B @ mapped position over matched points, in µm.

    `points_vox` are the reference positions and `diff_vox` reference minus mapped moving,
    so what is left is a map acting on the mapped positions (`points - diff`): fitted
    there, a moving set at 0.8 x the reference reads a scale of 1.25.

    A residual that grows across the field is a scale or rotation error of the transform,
    which median distances alone hide; a constant residual is a shift. Returns the residual
    map's per-axis scale and rotation (1 and 0 for no such error), its shift in µm, and the
    displacement it accumulates across the extent of the points (`trend_um`, per axis).
    """
    diff = np.asarray(diff_vox, dtype=float).reshape(-1, 3) * np.asarray(voxel_um, float)
    ref = np.asarray(points_vox, dtype=float).reshape(-1, 3) * np.asarray(voxel_um, float)
    pts = ref - diff  # the mapped positions, where the residual map acts
    n = int(len(pts))
    if n < MIN_POINTS_FOR_AFFINE:
        return {"n": n, "scale_zyx": None, "rotation_yx_deg": None}
    # positions about their centre: points at one depth (tiles) make a constant column,
    # which would otherwise share the shift with a false z gradient
    centre = pts.mean(axis=0)
    design = np.c_[np.ones(n), pts - centre]
    coef, *_ = np.linalg.lstsq(design, diff, rcond=None)
    gradient = coef[1:].T  # diff ~ shift + gradient @ pos
    shift = coef[0] - gradient @ centre
    rotation, stretch = polar(np.eye(3) + gradient)
    _total, yx = rotation_angles(rotation)
    extent = pts.max(axis=0) - pts.min(axis=0)
    return {
        "n": n,
        "scale_zyx": np.diag(stretch).tolist(),
        "rotation_yx_deg": yx,
        "shift_um": shift.tolist(),
        "trend_um": (gradient @ extent).tolist(),
    }


def _residuals(points_vox: ArrayLike, diff_vox: ArrayLike, voxel_um: ArrayLike) -> dict:
    """Return each residual with where it was measured, in µm (for the overlays)."""
    um = np.asarray(voxel_um, float)
    return {
        "position_um": (np.asarray(points_vox, float).reshape(-1, 3) * um).tolist(),
        "diff_um": (np.asarray(diff_vox, float).reshape(-1, 3) * um).tolist(),
    }


# --- layer 2: point-based residuals -----------------------------------------------------
# Points seen in both channels (beads, nuclei, or points a person clicked), mapped through
# the transform without warping any image, and compared where they pair up.


def point_residuals(
    transform: Transform,
    mov_points: np.ndarray,
    ref_points: np.ndarray,
    ref_um: tuple,
    radius_um: float = 1.0,
    axes: str = "zyx",
    paired: bool = False,
) -> dict:
    """Return how far `transform` leaves each point from its partner, in µm.

    `mov_points` and `ref_points` are N x 3 voxel coordinates (z, y, x) in their own
    volumes; the moving ones are mapped onto the reference (no warping), whose voxel size
    is `ref_um`. `paired` points correspond row by row (points a person clicked in both
    channels); otherwise they pair by mutual nearest neighbour within `radius_um`, measured
    on `axes` ("zyx", or "yx" for objects long in z such as nuclei). Displacements are
    reference minus mapped moving, in reference voxels, summarised by `displacement_stats`
    and `residual_affine`.
    """
    mov_points = np.asarray(mov_points, dtype=float).reshape(-1, 3)
    ref_points = np.asarray(ref_points, dtype=float).reshape(-1, 3)
    mapped = transform.apply_points(mov_points) if len(mov_points) else mov_points
    if paired:
        if len(mov_points) != len(ref_points):
            raise ValueError(
                f"paired points need one partner each: {len(mov_points)} moving, "
                f"{len(ref_points)} reference"
            )
        i = j = np.arange(len(mov_points))
        matched_by = "given pairs"
    else:
        cols = slice(1, 3) if axes == "yx" else slice(0, 3)
        um = np.asarray(ref_um, float)
        i, j = mutual_nearest(
            mapped[:, cols] * um[cols], ref_points[:, cols] * um[cols], radius_um
        )
        matched_by = f"mutual nearest within {radius_um} um ({axes})"
    diff = ref_points[j] - mapped[i]
    return {
        "n_mov": int(len(mov_points)),
        "n_ref": int(len(ref_points)),
        "n_matched": int(len(i)),
        "matched_by": matched_by,
        "stats": displacement_stats(diff, ref_um),
        "affine": residual_affine(ref_points[j], diff, ref_um),
        "residuals": _residuals(ref_points[j], diff, ref_um),
    }


# --- layer 3: image-based residuals -----------------------------------------------------
# For data without landmarks: local shifts between the warped moving volume and the
# reference, measured by phase correlation and summarised like matched points.


@dataclass
class ComparablePair:
    """The warped moving volume and the reference, prepared once for layer 3's measures."""

    warped: np.ndarray
    ref: np.ndarray
    has_data: np.ndarray  # where the warped moving volume has data
    compare: str


def prepare_pair(transform: Transform, mov: ArrayLike, ref: ArrayLike, compare: str = "edges"):
    """Return the pair `tile_residuals` and `z_residuals` compare, to share between them.

    `edges` takes the Sobel magnitude in each volume's own grid before warping, so the
    empty padding carries no edges and cross-modal pairs compare structure; `intensity`
    compares the volumes as they are.
    """
    if compare not in ("edges", "intensity"):
        raise ValueError(f"compare must be 'edges' or 'intensity', got {compare!r}")
    mov = np.nan_to_num(np.asarray(mov, dtype=np.float32))
    ref = np.nan_to_num(np.asarray(ref, dtype=np.float32))
    if compare == "edges":
        mov, ref = filters.sobel(mov).astype(np.float32), filters.sobel(ref).astype(np.float32)
    if transform.is_identity and mov.shape == ref.shape:
        # the unregistered pair: an identity warp returns the volume unchanged
        return ComparablePair(mov, ref, np.ones(ref.shape, dtype=bool), compare)
    warped = transform.apply(mov, reference=ref)
    return ComparablePair(warped, ref, transform.coverage(mov.shape, ref.shape), compare)


def _pair(transform, mov, ref, compare: str, pair: ComparablePair | None) -> ComparablePair:
    if pair is None:
        return prepare_pair(transform, mov, ref, compare)
    if pair.compare != compare:
        raise ValueError(f"the pair was prepared for {pair.compare!r}, not {compare!r}")
    return pair


def _zscore(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


def tile_residuals(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    voxel_um: tuple,
    compare: str = "edges",
    tile_px: int = 256,
    grid: int = 7,
    min_ncc: float = 0.5,
    max_shift_px: float | None = None,
    min_coverage: float = 0.95,
    pair: ComparablePair | None = None,
) -> dict:
    """Return local yx residuals over a grid of tiles of the max projections, in µm.

    Each tile of the warped moving projection is phase-correlated with the reference's; a
    tile counts when it is mostly real data, its shift is plausible (under `max_shift_px`,
    default a quarter tile) and its shifted copy correlates above `min_ncc`. Displacements
    follow `point_residuals`' convention (reference minus mapped moving) and are summarised
    by `displacement_stats` and `residual_affine` (a radial pattern = a scale error).
    `pair` (from `prepare_pair`) skips preparing the volumes again.
    """
    pair = _pair(transform, mov, ref, compare, pair)
    warped, ref_p, has_data = pair.warped, pair.ref, pair.has_data
    w_mip, r_mip = warped.max(axis=0), ref_p.max(axis=0)
    covered = has_data.mean(axis=0) >= 0.5
    _, ny, nx = ref_p.shape
    size = min(tile_px, ny, nx)  # a tile fits the image, and is exactly the window's size
    max_shift = size / 4 if max_shift_px is None else max_shift_px
    h = size // 2
    window = np.outer(np.hanning(size), np.hanning(size))
    points, diffs = [], []
    # distinct centres only: a tile that fits the image once is one measurement
    for cy in np.unique(np.linspace(h, ny - (size - h), grid).astype(int)):
        for cx in np.unique(np.linspace(h, nx - (size - h), grid).astype(int)):
            tile = (slice(cy - h, cy - h + size), slice(cx - h, cx - h + size))
            if covered[tile].mean() < min_coverage:
                continue
            r, w = r_mip[tile], w_mip[tile]
            if r.std() == 0 or w.std() == 0:
                continue
            r, w = _zscore(r), _zscore(w)
            shift, _ = phase_cross_corr(r * window, w * window, upsample_factor=10)
            if np.abs(shift).max() > max_shift:
                continue
            # the shared window pulls the peak toward zero shift (~5% short on nuclei):
            # measure the remainder once more, where that pull is a share of almost nothing
            moved = ndi_shift(w, shift, order=1, mode="nearest")
            remainder, _ = phase_cross_corr(r * window, moved * window, upsample_factor=10)
            shift = shift + remainder
            if np.abs(shift).max() > max_shift:
                continue
            moved = ndi_shift(w, shift, order=1, mode="nearest")
            if np.corrcoef(r.ravel(), moved.ravel())[0, 1] < min_ncc:
                continue
            points.append((ref_p.shape[0] / 2, cy, cx))
            diffs.append((0.0, *shift))  # moving shifted by `shift` matches: ref - mapped
    points, diffs = np.array(points).reshape(-1, 3), np.array(diffs).reshape(-1, 3)
    return {
        "compare": compare,
        "n_tiles": int(len(points)),
        "stats": displacement_stats(diffs, voxel_um),
        "affine": residual_affine(points, diffs, voxel_um),
        "residuals": _residuals(points, diffs, voxel_um),
    }


def z_residuals(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    voxel_um: tuple,
    compare: str = "edges",
    crop_px: int = 768,
    min_coverage: float = 0.5,
    pair: ComparablePair | None = None,
) -> dict:
    """Return the residual shift in the centre and 4 corner crops (3D phase correlation), µm.

    Different corners show a tilt (`tilt_dz_um`: the spread of their z residuals). A crop
    that is mostly padding is skipped; in intensity mode the padding is filled with the
    crop's median so its edge does not dominate the correlation. `pair` (from
    `prepare_pair`) skips preparing the volumes again.
    """
    pair = _pair(transform, mov, ref, compare, pair)
    warped, ref_p, has_data = pair.warped, pair.ref, pair.has_data
    _, ny, nx = ref_p.shape
    cy, cx = min(crop_px, ny), min(crop_px, nx)
    starts = {
        "centre": ((ny - cy) // 2, (nx - cx) // 2),
        "top_left": (0, 0),
        "top_right": (0, nx - cx),
        "bottom_left": (ny - cy, 0),
        "bottom_right": (ny - cy, nx - cx),
    }
    um = np.asarray(voxel_um, float)
    crops = {}
    for name, (y0, x0) in starts.items():
        box = (slice(None), slice(y0, y0 + cy), slice(x0, x0 + cx))
        w, r, cov = warped[box], ref_p[box], has_data[box]
        if cov.mean() < min_coverage or r.std() == 0 or w[cov].std() == 0:
            crops[name] = None
            continue
        if compare == "intensity":
            w = np.where(cov, w, np.median(w[cov]))
        shift, _ = phase_cross_corr(_zscore(r), _zscore(w), upsample_factor=10)
        crops[name] = (np.asarray(shift) * um).tolist()
    # one value per distinct place: corners of a crop as big as the image are one place
    corner_dz = list(
        {starts[k]: v[0] for k, v in crops.items() if k != "centre" and v is not None}.values()
    )
    return {
        "compare": compare,
        "crops_um": crops,
        "tilt_dz_um": float(max(corner_dz) - min(corner_dz)) if len(corner_dz) > 1 else None,
    }
