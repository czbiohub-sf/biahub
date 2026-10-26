"""Helpers for choosing a registration pipeline from a benchmark.

The registration skill runs a few candidate pipelines (estimate configs, e.g. stacks of
methods) through `nextflow/registration_benchmark_wf.nf` on a sample of positions and picks
one with `pick_pipeline`, so the choice is a rule on the held-out evaluation, not a guess.
"""

from __future__ import annotations

import csv

import numpy as np


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pick_pipeline(
    benchmark_csv,
    blocks: dict[str, int] | None = None,
    tie_um: float = 0.05,
    must_beat: float = 0.5,
) -> dict:
    """Return the pipeline to register with, and why, from a benchmark table.

    Each pipeline's final transform ("transforms" rows; its blocks are not choices) is
    scored by the median over held-out positions of `points_median_um` when the table has
    landmarks, else `tiles_median_um`. The lowest wins; within `tie_um` of it, the pipeline
    with fewer `blocks` (simpler) does. No winner unless it is below `must_beat` times the
    unregistered pair's median (registration must clearly help).
    """
    with open(benchmark_csv) as f:
        rows = list(csv.DictReader(f))
    finals = [r for r in rows if r["candidate"] == "transforms"]
    metric = (
        "points_median_um"
        if any(_float(r.get("points_median_um")) is not None for r in finals)
        else "tiles_median_um"
    )

    def median(selected):
        values = [_float(r.get(metric)) for r in selected]
        values = [v for v in values if v is not None]
        return float(np.median(values)) if values else None

    scores = {}
    for pipeline in dict.fromkeys(r["pipeline"] for r in finals):
        score = median([r for r in finals if r["pipeline"] == pipeline])
        if score is not None:
            scores[pipeline] = score
    unregistered = median([r for r in rows if r["candidate"] == "unregistered"])
    if not scores:
        return {
            "pipeline": None,
            "metric": metric,
            "median_um": None,
            "unregistered_um": unregistered,
            "scores": {},
            "reason": "no pipeline was evaluated",
        }
    best = min(scores.values())
    close = [p for p, s in scores.items() if s - best <= tie_um]
    blocks = blocks or {}
    winner = min(close, key=lambda p: (blocks.get(p, 1), scores[p]))
    if unregistered is not None and scores[winner] >= must_beat * unregistered:
        return {
            "pipeline": None,
            "metric": metric,
            "median_um": scores[winner],
            "unregistered_um": unregistered,
            "scores": scores,
            "reason": f"no pipeline clearly beats unregistered ({unregistered:.2f} µm): "
            f"the best, {winner}, reaches {scores[winner]:.2f} µm",
        }
    reason = f"{winner}: {metric} median {scores[winner]:.2f} µm on held-out positions"
    if unregistered is not None:
        reason += f" (unregistered {unregistered:.2f} µm)"
    if len(close) > 1 and blocks.get(winner, 1) < max(blocks.get(p, 1) for p in close):
        reason += f"; within {tie_um} µm of the best, the simpler pipeline"
    return {
        "pipeline": winner,
        "metric": metric,
        "median_um": scores[winner],
        "unregistered_um": unregistered,
        "scores": scores,
        "reason": reason,
    }
