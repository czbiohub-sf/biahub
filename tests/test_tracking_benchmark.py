from __future__ import annotations

from pathlib import Path

import pandas as pd

from biahub.benchmarking.tracking import (
    MethodSpec,
    _resolve_prediction_path,
    _write_temporary_config,
    match_tracks,
    run_benchmark,
    score_tracks,
)


def _tracks(fov_name: str, track_id: int, xs: list[float], ys: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fov_name": [fov_name] * len(xs),
            "track_id": [track_id] * len(xs),
            "parent_track_id": [-1] * len(xs),
            "t": list(range(len(xs))),
            "x": xs,
            "y": ys,
        }
    )


def test_score_tracks_perfect_match():
    ref = pd.concat(
        [
            _tracks("A/1/0", 1, [0, 1, 2], [0, 1, 2]),
            _tracks("A/1/0", 2, [10, 11, 12], [10, 11, 12]),
        ],
        ignore_index=True,
    )
    pred = ref.copy()

    scores = score_tracks(pred, ref, max_distance=0.5)

    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["mean_track_purity"] == 1.0
    assert scores["mean_ref_track_coverage"] == 1.0


def test_match_tracks_rejects_far_points():
    ref = _tracks("A/1/0", 1, [0, 1], [0, 1])
    pred = _tracks("A/1/0", 1, [0, 20], [0, 20])

    _, _, matches = match_tracks(pred, ref, max_distance=2.0)

    assert len(matches) == 1
    assert matches.iloc[0]["t"] == 0


def test_run_benchmark(tmp_path):
    ref = _tracks("A/1/0", 1, [0, 1], [0, 1])
    pred = ref.copy()

    ref_path = tmp_path / "A_1_0.csv"
    pred_path = tmp_path / "pred_A_1_0.csv"
    ref.to_csv(ref_path, index=False)
    pred.to_csv(pred_path, index=False)

    config = tmp_path / "config.yml"
    config.write_text(
        f"""
annotations:
  - {ref_path}
methods:
  - name: test
    kind: csv_root
    path: {tmp_path}
    track_csv_name: pred_A_1_0.csv
"""
    )

    per_fov, per_method = run_benchmark(config)
    assert len(per_fov) == 1
    assert len(per_method) == 1
    assert per_method.iloc[0]["precision"] == 1.0


def test_command_method_writes_temporary_config(tmp_path):
    base_config = tmp_path / "base.yml"
    base_config.write_text(
        """
data_zarr: /tmp/data
output_dir: /tmp/out
solver:
  tracklet_solver: true
"""
    )
    spec = MethodSpec(
        name="ett",
        kind="command",
        path=str(tmp_path / "runner"),
        command_template="echo {config_path}",
        base_config=str(base_config),
        config_overrides={"output_dir": str(tmp_path / "out")},
    )

    config_path = _write_temporary_config(spec, "A/1/0")
    assert config_path.exists()
    assert config_path.parent.parent == tmp_path / "runner"

    pred_path = _resolve_prediction_path(spec, "A/1/0")
    assert pred_path == tmp_path / "runner" / "A_1_0.csv"
