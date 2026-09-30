"""Adaptive quality-score flagging, shared by every fallback pass.

A fixed score threshold does not survive contact with a second dataset: a run whose
median score is 0.87 needs a very different bar than one whose median sits at 0.75,
where a fixed 0.75 would flag roughly half the run. Flagging against the run's own
median - k*MAD line lets a fallback pass (repair, sweep, ...) stay a fallback regardless
of how the run as a whole scored. MAD rather than standard deviation because std is
itself inflated by the outliers being hunted for.

Flagging only reports; it never modifies a transform.
"""

from __future__ import annotations

import click
import numpy as np
import pandas as pd

from numpy.typing import ArrayLike

# Absolute floor below which a timepoint is unusable rather than merely unusual
# relative to an otherwise-good run.
HARD_FAIL_SCORE = 0.40
FLAG_FLOOR_SCORE = 0.80
FLAG_K_MAD = 2.0


def flag_timepoints(
    scores: ArrayLike,
    k_mad: float = FLAG_K_MAD,
    floor: float = FLAG_FLOOR_SCORE,
    hard_fail: float = HARD_FAIL_SCORE,
) -> pd.DataFrame:
    """One row per timepoint, flagged against the run's own median - k*MAD line.

    Flagged when below the adaptive line AND below the absolute floor, or below
    `hard_fail`, or missing a score entirely.
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


def select_flagged(
    score_col: ArrayLike,
    label: str,
    max_timepoints: int | None = None,
) -> tuple[list[int], dict]:
    """Timepoints for a fallback pass to act on, from the adaptive median-2*MAD line.

    Shared by every fallback pass so they can't drift apart in how they choose work, and
    so no pass carries its own fixed score threshold.

    Returns (flagged timepoints, run statistics) -- the statistics are returned rather
    than left for callers to recompute, so whatever gets logged is provably the same
    numbers the selection used.

    Returns the worst `max_timepoints` when a cap is set, and always says which ones it
    dropped: a capped pass that logs nothing about the drop would read as full coverage.
    """
    score_col = np.asarray(score_col, dtype=float)
    n_t = len(score_col)
    try:
        flags = flag_timepoints(score_col)
    except ValueError:
        click.echo(f"{label}: no finite scores to flag against; skipping.")
        return [], {}
    flagged = [int(t) for t in flags.loc[flags["flagged"], "t"]]
    if not flagged:
        click.echo(f"{label}: nothing flagged, nothing to do.")
        return [], dict(flags.attrs)

    click.echo(
        f"{label}: {len(flagged)} of {n_t} timepoints flagged "
        f"(adaptive line {flags.attrs['adaptive_line']:.3f}, median "
        f"{flags.attrs['median']:.3f}) -> {flagged}"
    )
    if max_timepoints is not None and len(flagged) > max_timepoints:
        worst = sorted(flagged, key=lambda t: np.nan_to_num(score_col[t], nan=-1.0))
        dropped = sorted(worst[max_timepoints:])
        flagged = sorted(worst[:max_timepoints])
        click.echo(
            f"  capped at max_timepoints={max_timepoints}; taking the worst {len(flagged)} "
            f"and LEAVING {len(dropped)} untouched: {dropped}"
        )
    return flagged, dict(flags.attrs)
