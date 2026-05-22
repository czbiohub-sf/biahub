"""Benchmark cell-tracking outputs against manual annotations.

This module compares per-detection track tables, which makes it usable for
multiple trackers as long as they can export CSV tables with at least:

* ``fov_name``
* ``track_id``
* ``t``
* ``x``
* ``y``

The benchmark is designed for tracking comparisons first. Segmentation is
only represented indirectly through object counts and centroid error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

REQUIRED_COLUMNS = {"track_id", "t", "x", "y"}
DEFAULT_MAX_DISTANCE = 30.0


@dataclass(frozen=True)
class MethodSpec:
    """Description of a prediction source to score."""

    name: str
    kind: str
    path: str
    csv_pattern: str | None = None
    track_csv_name: str | None = None
    command_template: str | None = None
    base_config: str | None = None
    config_overrides: dict[str, Any] | None = None
    working_dir: str | None = None


def _normalize_track_table(
    df: pd.DataFrame, source: str, default_fov_name: str | None = None
) -> pd.DataFrame:
    """Normalize a track table to the common benchmark schema."""
    rename_map = {
        "tracklet_id": "track_id",
        "parent_id": "parent_track_id",
        "parent_tracklet_id": "parent_track_id",
        "fov": "fov_name",
        "fov_key": "fov_name",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{source}: missing required columns {sorted(missing)}")

    out = df.copy()
    if "fov_name" not in out.columns:
        out["fov_name"] = default_fov_name or "UNKNOWN"
    else:
        out["fov_name"] = out["fov_name"].fillna(default_fov_name or "UNKNOWN")
    if "parent_track_id" not in out.columns:
        out["parent_track_id"] = -1

    # Annotated CSVs may include cleaned noise rows without track assignments.
    out = out.dropna(subset=["track_id", "t", "x", "y"]).copy()

    out["track_id"] = out["track_id"].astype(int)
    out["parent_track_id"] = out["parent_track_id"].fillna(-1).astype(int)
    out["t"] = out["t"].astype(int)
    out["x"] = out["x"].astype(float)
    out["y"] = out["y"].astype(float)
    out["fov_name"] = out["fov_name"].astype(str)

    keep_cols = ["fov_name", "track_id", "parent_track_id", "t", "x", "y"]
    extras = [c for c in out.columns if c not in keep_cols]
    return out[keep_cols + extras]


def load_track_table(path: Path, default_fov_name: str | None = None) -> pd.DataFrame:
    """Load a CSV track table and normalize its schema."""
    return _normalize_track_table(pd.read_csv(path), str(path), default_fov_name)


def load_annotation_tables(paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Load annotation CSVs keyed by FOV name."""
    tables: dict[str, pd.DataFrame] = {}
    for path in paths:
        table = load_track_table(path)
        fov_names = table["fov_name"].unique()
        if len(fov_names) != 1:
            raise ValueError(f"{path}: expected one fov_name, got {list(fov_names)}")
        tables[str(fov_names[0])] = table
    return tables


def _resolve_prediction_path(spec: MethodSpec, fov_name: str) -> Path:
    """Resolve a method-specific prediction path for a given FOV."""
    fov_slug = fov_name.replace("/", "_")
    fov_dir = Path(*fov_name.split("/"))
    root = Path(spec.path)

    if spec.kind == "biahub":
        track_csv_name = spec.track_csv_name or f"tracks_{fov_slug}.csv"
        return root / fov_dir / track_csv_name
    if spec.kind == "csv_template":
        if spec.csv_pattern is None:
            raise ValueError(f"{spec.name}: csv_pattern is required for kind=csv_template")
        return Path(
            spec.csv_pattern.format(
                fov_name=fov_name,
                fov_slug=fov_slug,
                fov_path=str(fov_dir),
            )
        )
    if spec.kind == "csv_root":
        track_csv_name = spec.track_csv_name or f"{fov_slug}.csv"
        return root / track_csv_name
    if spec.kind == "command":
        if spec.csv_pattern is not None:
            return Path(
                spec.csv_pattern.format(
                    fov_name=fov_name,
                    fov_slug=fov_slug,
                    fov_path=str(fov_dir),
                )
            )
        track_csv_name = spec.track_csv_name or f"{fov_slug}.csv"
        return root / track_csv_name
    raise ValueError(f"{spec.name}: unsupported kind '{spec.kind}'")


def _write_temporary_config(spec: MethodSpec, fov_name: str) -> Path:
    """Materialize a per-FOV config for command-backed methods."""
    if not spec.base_config:
        raise ValueError(f"{spec.name}: base_config is required for command methods")

    with Path(spec.base_config).open() as f:
        config = yaml.safe_load(f)

    overrides = dict(spec.config_overrides or {})
    overrides.setdefault("fov_key", fov_name)
    if "output_dir" not in overrides:
        raise ValueError(f"{spec.name}: config_overrides must define output_dir")

    config.update(overrides)

    root = Path(spec.path)
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{spec.name}_", dir=root))
    config_path = temp_dir / f"{fov_name.replace('/', '_')}.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config_path


def _run_command_method(spec: MethodSpec, fov_name: str, prediction_path: Path) -> None:
    """Run a method-backed command if the prediction output does not exist."""
    if prediction_path.exists():
        return

    if not spec.command_template:
        raise ValueError(f"{spec.name}: command_template is required for command methods")

    fov_slug = fov_name.replace("/", "_")
    fov_dir = Path(*fov_name.split("/"))
    temp_config = _write_temporary_config(spec, fov_name)
    try:
        output_dir = None if spec.config_overrides is None else spec.config_overrides.get("output_dir")
        if output_dir is None:
            raise ValueError(f"{spec.name}: config_overrides must define output_dir")
        command = spec.command_template.format(
            fov_name=fov_name,
            fov_slug=fov_slug,
            fov_path=str(fov_dir),
            config_path=str(temp_config),
            output_dir=str(Path(output_dir)),
            prediction_path=str(prediction_path),
        )
        subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=spec.working_dir,
        )
    finally:
        shutil.rmtree(temp_config.parent, ignore_errors=True)


def _match_frame(
    pred_frame: pd.DataFrame, ref_frame: pd.DataFrame, max_distance: float
) -> pd.DataFrame:
    """Match detections within one frame using the Hungarian algorithm."""
    if pred_frame.empty or ref_frame.empty:
        return pd.DataFrame(
            columns=["pred_index", "ref_index", "distance", "pred_track_id", "ref_track_id"]
        )

    pred_xy = pred_frame[["x", "y"]].to_numpy(dtype=float)
    ref_xy = ref_frame[["x", "y"]].to_numpy(dtype=float)
    distances = np.linalg.norm(pred_xy[:, None, :] - ref_xy[None, :, :], axis=2)

    row_ind, col_ind = linear_sum_assignment(distances)
    keep = distances[row_ind, col_ind] <= max_distance

    matched = pd.DataFrame(
        {
            "pred_index": pred_frame.index.to_numpy()[row_ind[keep]],
            "ref_index": ref_frame.index.to_numpy()[col_ind[keep]],
            "distance": distances[row_ind[keep], col_ind[keep]],
            "pred_track_id": pred_frame.loc[
                pred_frame.index.to_numpy()[row_ind[keep]], "track_id"
            ].to_numpy(),
            "ref_track_id": ref_frame.loc[
                ref_frame.index.to_numpy()[col_ind[keep]], "track_id"
            ].to_numpy(),
        }
    )
    return matched


def match_tracks(
    prediction: pd.DataFrame,
    reference: pd.DataFrame,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match a prediction table to a reference table."""
    pred = _normalize_track_table(prediction, "prediction")
    ref = _normalize_track_table(reference, "reference")

    if not set(pred["fov_name"].unique()).issubset(set(ref["fov_name"].unique())):
        raise ValueError("Prediction and reference FOV names do not overlap cleanly.")

    matched_frames = []
    for fov_name in sorted(set(pred["fov_name"]).intersection(ref["fov_name"])):
        pred_fov = pred[pred["fov_name"] == fov_name]
        ref_fov = ref[ref["fov_name"] == fov_name]

        for t in sorted(set(pred_fov["t"]).union(ref_fov["t"])):
            pred_frame = pred_fov[pred_fov["t"] == t]
            ref_frame = ref_fov[ref_fov["t"] == t]
            if not pred_frame.empty and not ref_frame.empty:
                matched = _match_frame(pred_frame, ref_frame, max_distance)
                if not matched.empty:
                    matched["fov_name"] = fov_name
                    matched["t"] = t
                    matched_frames.append(matched)

    matches = (
        pd.concat(matched_frames, ignore_index=True)
        if matched_frames
        else pd.DataFrame(
            columns=[
                "pred_index",
                "ref_index",
                "distance",
                "pred_track_id",
                "ref_track_id",
                "fov_name",
                "t",
            ]
        )
    )
    return pred, ref, matches


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_tracks(
    prediction: pd.DataFrame,
    reference: pd.DataFrame,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    min_track_fraction: float = 0.5,
) -> dict[str, Any]:
    """Compute tracking and proxy segmentation metrics."""
    pred, ref, matches = match_tracks(prediction, reference, max_distance=max_distance)

    tp = int(len(matches))
    fp = int(len(pred) - tp)
    fn = int(len(ref) - tp)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    out: dict[str, Any] = {
        "pred_rows": len(pred),
        "ref_rows": len(ref),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_distance": float(matches["distance"].mean()) if tp else float("nan"),
        "median_distance": float(matches["distance"].median()) if tp else float("nan"),
        "pred_tracks": int(pred["track_id"].nunique()),
        "ref_tracks": int(ref["track_id"].nunique()),
        "matched_pred_tracks": int(matches["pred_track_id"].nunique()),
        "matched_ref_tracks": int(matches["ref_track_id"].nunique()),
    }

    pred_track_to_ref = (
        matches.groupby("pred_track_id")["ref_track_id"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    pred_track_purity = []
    for _, group in matches.groupby("pred_track_id"):
        counts = group["ref_track_id"].value_counts(normalize=True)
        pred_track_purity.append(float(counts.iloc[0]))
    out["mean_track_purity"] = float(np.mean(pred_track_purity)) if pred_track_purity else float("nan")

    ref_track_coverage = []
    ref_track_fragmentation = []
    for ref_track_id, group in matches.groupby("ref_track_id"):
        total_ref = int((ref["track_id"] == ref_track_id).sum())
        covered = len(group)
        ref_track_coverage.append(covered / total_ref if total_ref else 0.0)
        ref_track_fragmentation.append(group["pred_track_id"].nunique())
    out["mean_ref_track_coverage"] = (
        float(np.mean(ref_track_coverage)) if ref_track_coverage else float("nan")
    )
    out["ref_track_recall_at_threshold"] = (
        float(np.mean(np.asarray(ref_track_coverage) >= min_track_fraction))
        if ref_track_coverage
        else float("nan")
    )
    out["mean_ref_track_fragmentation"] = (
        float(np.mean(ref_track_fragmentation)) if ref_track_fragmentation else float("nan")
    )
    out["pred_track_precision_at_threshold"] = (
        float(np.mean(np.asarray(pred_track_purity) >= min_track_fraction))
        if pred_track_purity
        else float("nan")
    )

    if "parent_track_id" in pred.columns and "parent_track_id" in ref.columns:
        parent_checks = []
        for pred_track_id, ref_track_id in pred_track_to_ref.items():
            pred_parent_ids = (
                pred.loc[pred["track_id"] == pred_track_id, "parent_track_id"]
                .dropna()
                .astype(int)
                .to_numpy()
            )
            pred_parent_ids = np.unique(pred_parent_ids[pred_parent_ids >= 0])
            if len(pred_parent_ids) != 1:
                continue
            pred_parent = int(pred_parent_ids[0])
            if pred_parent not in pred_track_to_ref:
                continue
            pred_parent_ref = pred_track_to_ref[pred_parent]

            ref_parent_ids = (
                ref.loc[ref["track_id"] == ref_track_id, "parent_track_id"]
                .dropna()
                .astype(int)
                .to_numpy()
            )
            ref_parent_ids = np.unique(ref_parent_ids[ref_parent_ids >= 0])
            if len(ref_parent_ids) != 1:
                continue
            ref_parent = int(ref_parent_ids[0])

            parent_checks.append(int(pred_parent_ref == ref_parent))

        out["parent_accuracy"] = (
            float(np.mean(parent_checks)) if parent_checks else float("nan")
        )
        out["parent_checked_tracks"] = len(parent_checks)

    counts_pred = pred.groupby(["fov_name", "t"]).size()
    counts_ref = ref.groupby(["fov_name", "t"]).size()
    count_index = counts_pred.index.union(counts_ref.index)
    pred_counts = counts_pred.reindex(count_index, fill_value=0).to_numpy()
    ref_counts = counts_ref.reindex(count_index, fill_value=0).to_numpy()
    out["count_rmse"] = float(np.sqrt(np.mean((pred_counts - ref_counts) ** 2)))
    out["count_corr"] = _safe_corr(pred_counts.astype(float), ref_counts.astype(float))
    out["count_mae"] = float(np.mean(np.abs(pred_counts - ref_counts)))

    matched_per_frame = (
        matches.groupby(["fov_name", "t"]).size().reindex(count_index, fill_value=0).to_numpy()
    )
    out["mean_match_rate_per_frame"] = float(
        np.mean(matched_per_frame / np.maximum(ref_counts, 1))
    )

    out["min_track_fraction"] = min_track_fraction
    return out


def _load_methods(config: dict[str, Any]) -> list[MethodSpec]:
    methods = []
    for method in config.get("methods", []):
        methods.append(
            MethodSpec(
                name=method["name"],
                kind=method["kind"],
                path=method["path"],
                csv_pattern=method.get("csv_pattern"),
                track_csv_name=method.get("track_csv_name"),
                command_template=method.get("command_template"),
                base_config=method.get("base_config"),
                config_overrides=method.get("config_overrides"),
                working_dir=method.get("working_dir"),
            )
        )
    return methods


def run_benchmark(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the benchmark and return per-FOV and per-method summaries."""
    with config_path.open() as f:
        config = yaml.safe_load(f)

    annotation_paths = [Path(p) for p in config.get("annotations", [])]
    if not annotation_paths:
        raise ValueError("No annotations provided.")
    reference_tables = load_annotation_tables(annotation_paths)

    max_distance = float(config.get("max_distance", DEFAULT_MAX_DISTANCE))
    min_track_fraction = float(config.get("min_track_fraction", 0.5))

    per_fov_rows = []
    per_method_rows = []

    for method in _load_methods(config):
        for fov_name, reference in reference_tables.items():
            pred_path = _resolve_prediction_path(method, fov_name)
            if method.kind == "command":
                _run_command_method(method, fov_name, pred_path)
            if not pred_path.exists():
                raise FileNotFoundError(f"{method.name}: missing prediction file {pred_path}")

            prediction = load_track_table(pred_path, default_fov_name=fov_name)
            scores = score_tracks(
                prediction=prediction,
                reference=reference,
                max_distance=max_distance,
                min_track_fraction=min_track_fraction,
            )
            scores.update(
                {
                    "method": method.name,
                    "fov_name": fov_name,
                    "prediction_path": str(pred_path),
                }
            )
            per_fov_rows.append(scores)

        method_df = pd.DataFrame([row for row in per_fov_rows if row["method"] == method.name])
        summary = {
            "method": method.name,
            "fov_count": len(method_df),
            "pred_rows": int(method_df["pred_rows"].sum()),
            "ref_rows": int(method_df["ref_rows"].sum()),
            "tp": int(method_df["tp"].sum()),
            "fp": int(method_df["fp"].sum()),
            "fn": int(method_df["fn"].sum()),
            "precision": float(method_df["tp"].sum() / (method_df["tp"].sum() + method_df["fp"].sum()))
            if (method_df["tp"].sum() + method_df["fp"].sum())
            else float("nan"),
            "recall": float(method_df["tp"].sum() / (method_df["tp"].sum() + method_df["fn"].sum()))
            if (method_df["tp"].sum() + method_df["fn"].sum())
            else float("nan"),
            "mean_f1": float(method_df["f1"].mean()),
            "mean_distance": float(method_df["mean_distance"].mean()),
            "median_distance": float(method_df["median_distance"].mean()),
            "mean_track_purity": float(method_df["mean_track_purity"].mean()),
            "mean_ref_track_coverage": float(method_df["mean_ref_track_coverage"].mean()),
            "ref_track_recall_at_threshold": float(
                method_df["ref_track_recall_at_threshold"].mean()
            ),
            "mean_ref_track_fragmentation": float(method_df["mean_ref_track_fragmentation"].mean()),
            "pred_track_precision_at_threshold": float(
                method_df["pred_track_precision_at_threshold"].mean()
            ),
            "count_rmse": float(method_df["count_rmse"].mean()),
            "count_corr": float(method_df["count_corr"].mean()),
            "count_mae": float(method_df["count_mae"].mean()),
            "mean_match_rate_per_frame": float(method_df["mean_match_rate_per_frame"].mean()),
            "prediction_paths": json.dumps(method_df["prediction_path"].tolist()),
        }
        if "parent_accuracy" in method_df.columns:
            summary["parent_accuracy"] = float(method_df["parent_accuracy"].mean())
            summary["parent_checked_tracks"] = float(method_df["parent_checked_tracks"].sum())

        per_method_rows.append(summary)

    return pd.DataFrame(per_fov_rows), pd.DataFrame(per_method_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Benchmark YAML config")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for the per-method summary CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the per-method summary JSON.",
    )
    parser.add_argument(
        "--per-fov-csv",
        type=Path,
        default=None,
        help="Optional path for the per-FOV summary CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the benchmark CLI."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    per_fov, per_method = run_benchmark(args.config)
    per_method = per_method.sort_values(by=["mean_f1", "recall", "precision"], ascending=False)

    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(per_method.to_string(index=False))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        per_method.to_csv(args.output_csv, index=False)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(per_method.to_json(orient="records", indent=2))

    if args.per_fov_csv is not None:
        args.per_fov_csv.parent.mkdir(parents=True, exist_ok=True)
        per_fov.to_csv(args.per_fov_csv, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
