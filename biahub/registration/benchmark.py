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


def choose_positions(survey_rows: list[dict], n_held_out: int = 3) -> tuple[str, list[str]]:
    """Return (the position to estimate on, held-out positions) from `survey_positions`.

    Usable positions are written and under half NaN. The estimate position has the most
    spread-out structure (what constrains scale and rotation); held-out ones are the next
    most spread, from other wells first, so the evaluation sees how the transform carries
    across the plate.
    """
    usable = [
        r for r in survey_rows if not r.get("empty") and (r.get("nan_fraction") or 0) < 0.5
    ]
    if not usable:
        raise ValueError("no usable position: every one is empty or mostly NaN")
    ranked = sorted(usable, key=lambda r: -(r.get("spread") or 0))
    estimate = ranked[0]["position"]

    def well(position):
        return position.rsplit("/", 1)[0]

    others = [r["position"] for r in ranked[1:] if well(r["position"]) != well(estimate)]
    same = [r["position"] for r in ranked[1:] if well(r["position"]) == well(estimate)]
    return estimate, (others + same)[:n_held_out]


def benchmark_spec(
    path,
    case_name: str,
    moving,
    reference,
    moving_channel: str,
    reference_channel: str,
    estimate_positions,
    evaluate_positions: list[str],
    pipelines: dict,
    landmarks: str | None = None,
    tile_px: int | None = None,
):
    """Write the benchmark file `registration_benchmark_wf.nf` reads, checked first.

    One case (its stores, channels, estimate and held-out positions) and `pipelines`
    (name -> estimate config). Every config must load, every position must exist in the
    stores, and no position may be both estimated on and held out.
    """
    from pathlib import Path

    import yaml

    from biahub.settings import EstimateTransformSettings

    estimate_positions = (
        [estimate_positions]
        if isinstance(estimate_positions, str)
        else list(estimate_positions)
    )
    for store in [moving] + ([reference] if reference else []):
        for position in estimate_positions + list(evaluate_positions):
            if not (Path(store) / position).is_dir():
                raise ValueError(f"position {position} is not in {store}")
    both = set(estimate_positions) & set(evaluate_positions)
    if both:
        raise ValueError(f"{sorted(both)} are estimated on, so cannot be held out")
    for name, config in pipelines.items():
        try:
            EstimateTransformSettings(**yaml.safe_load(Path(config).read_text()))
        except Exception as error:
            raise ValueError(f"pipeline {name}: {config} does not load: {error}") from None
    case = {
        "name": case_name,
        "moving": str(Path(moving).resolve()),
        "moving_channel": moving_channel,
        "estimate_positions": estimate_positions,
        "evaluate_positions": list(evaluate_positions),
    }
    if reference:
        case.update(
            reference=str(Path(reference).resolve()), reference_channel=reference_channel
        )
    if landmarks:
        case["landmarks"] = landmarks
    if tile_px:
        case["tile_px"] = tile_px
    spec = {
        "cases": [case],
        "pipelines": [
            {"name": name, "config": str(Path(config).resolve())}
            for name, config in pipelines.items()
        ],
    }
    path = Path(path)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path
