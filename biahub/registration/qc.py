"""Quality control for beads-registration transforms.

Four measures, because no single one is sufficient and each catches something the others
miss:

**overlap score** -- fraction of reference beads with a moving bead within a radius. This
is what the estimator optimises, but it is *piecewise-constant*: measured on real data it
does not move at all across a +/-5 voxel misregistration and only then falls off a cliff.
A good pass/fail signal and a poor ranking signal.

**mean matched residual** -- mean distance from each warped bead to its nearest reference
bead. Continuous and monotone over exactly the range where overlap is blind (2.66 -> 10.95
voxels over a 0 -> 14 voxel perturbation), so this is what separates two transforms that
tie on score. Biased low in absolute terms when the reference cloud is much denser than the
moving one, so use it to compare transforms at the same timepoint, not as an absolute
accuracy figure.

**transform plausibility** -- shear and scale anisotropy from the decomposed linear part.
Needs no image data, and catches geometrically impossible transforms: failed timepoints in
an older run showed 57 degrees of shear and 3.8 anisotropy against normal values of ~6
degrees and ~1.29. The expected values are instrument properties, not identity -- an
oblique light-sheet plus deskew produces real z-y shear, and differing voxel sizes between
the arms produce real anisotropy.

**temporal smoothness** -- largest step in translation between consecutive timepoints.
Independent per-timepoint estimation can be more accurate per frame yet produce a jittery
series, which shows up as visible jitter in the applied movie.

Flagging is adaptive: median - k*MAD of the run's own scores, with an absolute floor. MAD
rather than standard deviation because std is inflated by the very outliers being hunted --
on a run with 8 near-zero scores, mean-2*std put the line at 0.48 and masked them, while
median-2*MAD was unmoved. The floor prevents a uniformly good run from having timepoints
flagged merely for sitting below its own median.

Flagged timepoints are reported, never modified. Interpolating them was measured to make
results worse: a transform scoring 0.43 was replaced by one scoring 0.00.
"""

import json

from pathlib import Path

import click
import numpy as np
import pandas as pd

from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

# Absolute floor below which a timepoint is unusable rather than merely unusual.
HARD_FAIL_SCORE = 0.40
# Timepoints must be below BOTH the adaptive line and this floor to be flagged, so a
# healthy run is not flagged for internal spread alone.
FLAG_FLOOR_SCORE = 0.70
FLAG_K_MAD = 2.0


def matched_residual(mov_peaks: ArrayLike, ref_peaks: ArrayLike) -> dict:
    """Mean and p95 distance from each moving bead to its nearest reference bead."""
    if mov_peaks is None or ref_peaks is None or len(mov_peaks) == 0 or len(ref_peaks) == 0:
        return {"resid_mean": np.nan, "resid_p95": np.nan}
    d, _ = cKDTree(np.asarray(ref_peaks)).query(np.asarray(mov_peaks), k=1)
    return {"resid_mean": float(d.mean()), "resid_p95": float(np.percentile(d, 95))}


def decompose_transform(matrix: ArrayLike) -> dict:
    """Shear and scale of the linear part, for plausibility checks.

    Shear is reported per axis pair as the deviation of the transformed basis vectors from
    orthogonal. On mantis data the y-x pair is clean (~0.5 deg) and serves as a noise
    floor, while z-y carries real shear (~6 deg zebrafish, ~10 deg cells).
    """
    linear = np.asarray(matrix, dtype=float)[:3, :3]

    def angle(u, v):
        c = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        return abs(float(np.degrees(np.arccos(np.clip(c, -1, 1)))) - 90.0)

    sv = np.linalg.svd(linear, compute_uv=False)
    return {
        "shear_zy_deg": angle(linear[:, 0], linear[:, 1]),
        "shear_zx_deg": angle(linear[:, 0], linear[:, 2]),
        "shear_yx_deg": angle(linear[:, 1], linear[:, 2]),
        "scale_max": float(sv.max()),
        "scale_min": float(sv.min()),
        "scale_anisotropy": float(sv.max() / max(sv.min(), 1e-12)),
    }


def translation_smoothness(transforms: list[ArrayLike]) -> dict:
    """Largest and mean step in translation between consecutive timepoints."""
    tr = np.asarray([np.asarray(m, dtype=float)[:3, 3] for m in transforms])
    if len(tr) < 2:
        return {}
    d = np.abs(np.diff(tr, axis=0))
    return {
        f"{stat}_abs_d{axis}": float(fn(d[:, i]))
        for i, axis in enumerate("zyx")
        for stat, fn in (("max", np.max), ("mean", np.mean))
    }


def flag_timepoints(
    scores: ArrayLike,
    k_mad: float = FLAG_K_MAD,
    floor: float = FLAG_FLOOR_SCORE,
    hard_fail: float = HARD_FAIL_SCORE,
) -> pd.DataFrame:
    """Adaptive per-run flagging. One row per timepoint, with reasons.

    Flagged when below the run's own median - k*MAD AND below the absolute floor, or below
    hard_fail, or missing a score entirely.
    """
    s = np.asarray(scores, dtype=float)
    finite = s[np.isfinite(s)]
    if not len(finite):
        raise ValueError("no finite scores to flag against")
    median = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - median)))
    line = median - k_mad * mad

    rows = []
    for t, score in enumerate(s):
        reasons = []
        if not np.isfinite(score):
            reasons.append("no_score")
        else:
            if score < line and score < floor:
                reasons.append("below_adaptive_line")
            if score < hard_fail:
                reasons.append("below_hard_fail")
        rows.append(
            {
                "t": t,
                "quality_score": score,
                "flagged": bool(reasons),
                "reasons": ";".join(reasons),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs.update(
        {
            "median": median,
            "mad": mad,
            "adaptive_line": line,
            "floor": floor,
            "hard_fail": hard_fail,
        }
    )
    return out


def write_qc_report(
    output_dir: Path,
    scores: ArrayLike,
    transforms: list[ArrayLike] | None = None,
    residuals: dict[int, dict] | None = None,
    extra: dict | None = None,
) -> pd.DataFrame:
    """Write qc_flags.csv and qc_summary.json alongside a registration run.

    Reports only -- no transform is altered here.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qc = flag_timepoints(scores)
    # pd.concat does not carry .attrs through, so keep the flagging stats before adding
    # any columns.
    stats = dict(qc.attrs)

    if transforms is not None:
        decomp = pd.DataFrame([decompose_transform(m) for m in transforms[: len(qc)]])
        qc = pd.concat([qc, decomp], axis=1)
        qc.attrs.update(stats)
    if residuals:
        for col in ("resid_mean", "resid_p95"):
            qc[col] = [residuals.get(t, {}).get(col, np.nan) for t in qc["t"]]

    qc.to_csv(output_dir / "qc_flags.csv", index=False)

    summary = {
        "n_timepoints": int(len(qc)),
        "median_score": stats["median"],
        "mad": stats["mad"],
        "adaptive_line": stats["adaptive_line"],
        "floor": stats["floor"],
        "hard_fail": stats["hard_fail"],
        "n_flagged": int(qc["flagged"].sum()),
        "flagged_timepoints": [int(t) for t in qc.loc[qc["flagged"], "t"]],
        "note": (
            "Flagged timepoints are reported, not modified. Interpolating them was "
            "measured to make results worse (0.43 -> 0.00 at one real timepoint)."
        ),
    }
    if transforms is not None:
        summary["smoothness"] = translation_smoothness(transforms)
    if "resid_mean" in qc:
        r = qc["resid_mean"].dropna()
        if len(r):
            summary["residual_median"] = float(r.median())
            summary["residual_max"] = float(r.max())
    if extra:
        summary.update(extra)

    (output_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2))
    click.echo(
        f"QC: median={summary['median_score']:.3f} MAD={summary['mad']:.3f} "
        f"adaptive line={summary['adaptive_line']:.3f}\n"
        f"    {summary['n_flagged']} of {summary['n_timepoints']} timepoints flagged"
        + (f" -> {summary['flagged_timepoints']}" if summary["n_flagged"] else "")
        + f"\n    wrote qc_flags.csv and qc_summary.json in {output_dir}"
    )
    return qc
