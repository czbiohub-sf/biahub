"""The benchmark: positions to test on, candidates evaluated there, choosing a pipeline.

`survey_positions` says which positions are usable and how their structure lies;
`select_positions` picks where to estimate and what to hold out.

`evaluate_candidates` evaluates many candidates (a transforms file, each block of its
stack) on many positions into one table, next to the unregistered pair. The registration
skill runs a few candidate pipelines (estimate configs, e.g. stacks of methods) through
`nextflow/registration_benchmark_wf.nf` on a sample of positions and picks one with
`select_pipeline`, so the choice is a rule on the held-out evaluation, not a guess.
"""

from __future__ import annotations

import csv
import warnings

from pathlib import Path

import numpy as np

from skimage import filters

from biahub.cli.parsing import position_key
from biahub.core.transform import Transform
from biahub.registration.evaluate import Evaluation, detect_landmarks, evaluate
from biahub.registration.methods.landmark.nuclei_matching import detect_nuclei
from biahub.utils.ngff import is_unwritten


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_pipeline(
    benchmark_csv,
    blocks: dict[str, int] | None = None,
    tie_um: float = 0.05,
    must_beat: float = 0.5,
    min_points: int = 10,
) -> dict:
    """Return the pipeline to register with, and why, from a benchmark table.

    Each pipeline's final transform ("transforms" rows; its blocks are not choices) is
    scored by the median over held-out positions of `points_median_um` when every final
    transform matched at least `min_points` landmarks on every position (fewer pairs make
    a median that looks good), else of `tiles_median_um`. The lowest wins; within `tie_um`
    of it, the pipeline with fewer `blocks` (simpler) does. No winner unless its tile
    median is below `must_beat` times the unregistered pair's (judged on tiles: the
    unregistered pair matches few landmarks).
    """
    with open(benchmark_csv) as f:
        rows = list(csv.DictReader(f))
    finals = [r for r in rows if r["candidate"] == "transforms"]
    enough = finals and all(
        _float(r.get("points_median_um")) is not None
        and (_float(r.get("points_n")) or 0) >= min_points
        for r in finals
    )
    metric = "points_median_um" if enough else "tiles_median_um"

    def median(selected, column):
        values = [_float(r.get(column)) for r in selected]
        values = [v for v in values if v is not None]
        return float(np.median(values)) if values else None

    scores, tiles = {}, {}
    for pipeline in dict.fromkeys(r["pipeline"] for r in finals):
        selected = [r for r in finals if r["pipeline"] == pipeline]
        score = median(selected, metric)
        if score is not None:
            scores[pipeline], tiles[pipeline] = score, median(selected, "tiles_median_um")
    unregistered = median(
        [r for r in rows if r["candidate"] == "unregistered"], "tiles_median_um"
    )
    base = {"metric": metric, "unregistered_tiles_um": unregistered, "scores": scores}
    if not scores:
        return {
            **base,
            "pipeline": None,
            "median_um": None,
            "reason": "no pipeline was evaluated",
        }
    best = min(scores.values())
    close = [p for p, v in scores.items() if v - best <= tie_um]
    blocks = blocks or {}
    winner = min(close, key=lambda p: (blocks.get(p, 1), scores[p]))
    if unregistered is not None and (tiles[winner] or np.inf) >= must_beat * unregistered:
        return {
            **base,
            "pipeline": None,
            "median_um": scores[winner],
            "reason": f"no pipeline clearly beats unregistered (tiles {unregistered:.2f} µm): "
            f"the best, {winner}, reaches {tiles[winner]:.2f} µm",
        }
    reason = f"{winner}: {metric} median {scores[winner]:.2f} µm on held-out positions"
    if unregistered is not None:
        reason += f" (unregistered tiles {unregistered:.2f} µm)"
    if not enough:
        reason += f"; tiles decided: some pipeline matched fewer than {min_points} landmarks"
    if len(close) > 1 and blocks.get(winner, 1) < max(blocks.get(p, 1) for p in close):
        reason += f"; within {tie_um} µm of the best, the simpler pipeline"
    return {**base, "pipeline": winner, "median_um": scores[winner], "reason": reason}


# --- survey before registering ------------------------------------------------------------


def survey_positions(
    position_dirpaths: list,
    channels: list[str],
    t: int = 0,
    nuclei: bool = False,
    cells: int = 4,
) -> list[dict]:
    """Return one row per position: whether it is usable, and how its structure lies.

    `empty`: never written (no stored chunks, metadata only). From the first channel at
    `t`: `nan_fraction`; `foreground`, the share of the max projection above Otsu; and
    `spread`, the share of a `cells` x `cells` grid holding foreground (structure all over
    the field constrains scale and rotation; a corner does not). `peak_z` per channel:
    the slice of highest mean signal. `n_nuclei` when `nuclei` (detect_nuclei).
    """
    from iohub import open_ome_zarr

    rows = []
    for path in position_dirpaths:
        row = {"position": position_key(path), "empty": False}
        with open_ome_zarr(path, mode="r") as position:
            if is_unwritten(position):
                rows.append({**row, "empty": True})
                continue
            index = [position.channel_names.index(c) for c in channels]
            volumes = [np.asarray(position.data[t, i], dtype=np.float32) for i in index]
            voxel_um = tuple(position.scale[-3:])
        first = volumes[0]
        row["nan_fraction"] = float(np.isnan(first).mean())
        row["peak_z"] = {
            c: int(np.nanargmax(np.nanmean(v, axis=(1, 2)))) if np.isfinite(v).any() else None
            for c, v in zip(channels, volumes, strict=True)
        }
        mip = np.nanmax(first, axis=0) if np.isfinite(first).any() else None
        if mip is None or np.nanstd(mip) == 0:
            rows.append({**row, "foreground": 0.0, "spread": 0.0})
            continue
        mip = np.nan_to_num(mip, nan=float(np.nanmin(mip)))
        mask = mip > filters.threshold_otsu(mip)
        ny, nx = mask.shape
        filled = [
            mask[
                i * ny // cells : (i + 1) * ny // cells,
                j * nx // cells : (j + 1) * nx // cells,
            ].mean()
            > 0.05
            for i in range(cells)
            for j in range(cells)
        ]
        row["foreground"] = float(mask.mean())
        row["spread"] = float(np.mean(filled))
        if nuclei:
            row["n_nuclei"] = int(len(detect_nuclei(first, voxel_um)))
        rows.append(row)
    return rows


def select_positions(survey_rows: list[dict], n_held_out: int = 3) -> tuple[str, list[str]]:
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


def write_benchmark_config(
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


# --- many candidates on many positions (the benchmark's core) ------------------------------


def _row(record: Evaluation, candidate: str, position: str, t: int) -> dict:
    """Flatten one evaluation into a table row."""
    points, tiles = record.points or {}, record.tiles or {}
    affine = points.get("affine") or {}
    scale = affine.get("scale_zyx")
    return {
        "candidate": candidate,
        "position": position,
        "t": t,
        "tiles_n": tiles.get("n_tiles"),
        "tiles_median_um": (tiles.get("stats") or {}).get("median_um"),
        "points_n": points.get("n_matched"),
        "points_median_um": (points.get("stats") or {}).get("median_um"),
        "scale_yx": None if scale is None else list(scale[1:]),
        "rotation_yx_deg": affine.get("rotation_yx_deg"),
        "z_um": record.z_um,
        "z_from": record.z_from,
        "transform_scale_zyx": record.geometry["scale_zyx"],
        "transform_rotation_yx_deg": record.geometry["rotation_yx_deg"],
        "empty": None,
    }


def _blank(volume: np.ndarray) -> bool:
    """Whether a volume holds no signal: never written, all NaN, or one constant value."""
    finite = volume[np.isfinite(volume)]
    return finite.size == 0 or finite.min() == finite.max()


_MEASURED = (
    "tiles_n",
    "tiles_median_um",
    "points_n",
    "points_median_um",
    "scale_yx",
    "rotation_yx_deg",
    "z_um",
    "z_from",
    "transform_scale_zyx",
    "transform_rotation_yx_deg",
)


def _empty_row(position: str, t: int, empty: str) -> dict:
    """Return the one row of a position with nothing to compare: `_row`'s columns, empty."""
    row = {"candidate": "unregistered", "position": position, "t": t}
    return {**row, **dict.fromkeys(_MEASURED), "empty": empty}


def evaluate_candidates(
    candidates: dict,
    position_pairs: list,
    moving_channel: str,
    reference_channel: str,
    t: int = 0,
    landmarks: str | None = None,
    images: bool = True,
    compare: str = "edges",
    detect_kwargs: dict | None = None,
    figures_dir=None,
    reference_t: int | None = None,
    **image_kwargs,
) -> list[dict]:
    """Evaluate every candidate on every position at `t`: one row each (`_row`).

    `candidates` maps a name to a forward `Transform` for every position, or to a dict
    keyed by `t` or by (position, t) (e.g. `block_candidates`); a candidate with no
    transform for a position is left out there. `position_pairs` are (moving, reference)
    position paths. Each position also gets an "unregistered" row (identity), measured
    once; landmarks are detected once per position. A position whose moving or reference
    image is blank (no signal) gets one row instead, its `empty` saying which, and a warning.
    With `figures_dir`, each candidate's
    figure (`plot_evaluation`, against that unregistered pair) is saved there.
    `reference_t` reads the reference at another timepoint than `t` (a stabilization's
    reference frame); by default the same.
    """
    from iohub import open_ome_zarr

    rows = []
    for mov_path, ref_path in position_pairs:
        key = position_key(mov_path)
        with open_ome_zarr(mov_path, mode="r") as m, open_ome_zarr(ref_path, mode="r") as r:
            mov = np.asarray(m.data[t, m.channel_names.index(moving_channel)], np.float32)
            t_ref = t if reference_t is None else reference_t
            ref = np.asarray(
                r.data[t_ref, r.channel_names.index(reference_channel)], np.float32
            )
            mov_um, ref_um = tuple(m.scale[-3:]), tuple(r.scale[-3:])
        sides = [side for side, v in (("moving", mov), ("reference", ref)) if _blank(v)]
        if sides:  # nothing to compare: one row says so, instead of a row per candidate
            empty = "both" if len(sides) == 2 else sides[0]
            warnings.warn(
                f"{key} t={t}: the {empty} image is blank; not evaluated", stacklevel=2
            )
            rows.append(_empty_row(key, t, empty))
            continue
        marks = detect_landmarks(landmarks, mov, ref, mov_um, ref_um, detect_kwargs)
        named = {"unregistered": Transform.identity(3), **candidates}
        for name, candidate in named.items():
            transform = (
                candidate
                if isinstance(candidate, Transform)
                else candidate.get((key, t), candidate.get(t))
            )
            if transform is None:
                continue
            record = evaluate(
                transform,
                mov,
                ref,
                mov_um,
                ref_um,
                landmarks=marks,
                images=images,
                compare=compare,
                baseline=False,
                **image_kwargs,
            )
            rows.append(_row(record, name, key, t))
            if name == "unregistered":
                unregistered = {
                    k: getattr(record, k)
                    for k in ("points", "tiles", "z_crops", "z_um", "z_from")
                }
            elif figures_dir is not None:
                from biahub.registration.evaluate_view import plot_evaluation

                record.baseline = unregistered  # measured once for the position
                Path(figures_dir).mkdir(parents=True, exist_ok=True)
                name_safe = f"{name}_{key}_t{t}".replace("/", "_").replace(":", "-")
                plot_evaluation(
                    record,
                    path=Path(figures_dir) / f"{name_safe}.png",
                    title=f"{name} {key} t={t}",
                )
    return rows


def block_candidates(run_dir) -> dict:
    """Return each block of a stack run as a candidate: {block: {t or (position, t): T}}.

    From the per-timepoint records the run wrote (`blocks`: every block's transform),
    under `<run>/timepoints/` or, one list per position, `<run>/positions/<r/c/f>/...` --
    only the layout of the positions in the run's `run.json`, not an earlier run's.
    """
    import json

    run_dir = Path(run_dir)
    run_file = run_dir / "run.json"  # the run's positions: only its own layout is read
    positions = json.loads(run_file.read_text())["positions"] if run_file.exists() else None
    if positions is None:
        record_dirs = [(None, run_dir / "timepoints")] + [
            ("/".join(d.parent.relative_to(run_dir / "positions").parts), d)
            for d in sorted((run_dir / "positions").glob("*/*/*/timepoints"))
        ]
    elif len(positions) == 1:
        record_dirs = [(None, run_dir / "timepoints")]
    else:
        record_dirs = [(key, run_dir / "positions" / key / "timepoints") for key in positions]
    candidates: dict = {}
    for position, directory in record_dirs:
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            record = json.loads(path.read_text())
            for block in record.get("blocks") or []:
                if block["matrix"] is None:
                    continue
                key = record["t"] if position is None else (position, record["t"])
                candidates.setdefault(block["block"], {})[key] = Transform(
                    np.asarray(block["matrix"], dtype=float)
                )
    return candidates
