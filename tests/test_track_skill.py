"""The track-with-biahub skill's compare_tracks.py."""

import importlib.util

from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".claude/skills/track-with-biahub/templates/compare_tracks.py"
)


@pytest.fixture(scope="module")
def compare():
    spec = importlib.util.spec_from_file_location("compare_tracks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_annotation_version_is_numeric(compare):
    paths = ["a/v2_tracks.csv", "a/v10_tracks.csv", "a/v9_tracks.csv"]
    assert compare.latest(paths) == "a/v10_tracks.csv"


@pytest.mark.parametrize(
    "matched, predicted, annotated, f1",
    [(0, 5, 4, 0.0), (0, 0, 4, 0.0), (3, 4, 4, 0.75), (0, 0, 0, float("nan"))],
)
def test_division_f1(compare, matched, predicted, annotated, f1):
    result = compare.division_f1(matched, predicted, annotated)
    assert result == pytest.approx(f1, nan_ok=True)


def test_cells_per_frame_counts_empty_frames(compare):
    import pandas as pd

    tracks = pd.DataFrame({"track_id": [1], "t": [2], "y": [0.0], "x": [0.0]})
    assert compare.cells_per_frame(tracks, n_frames=6) == pytest.approx(1 / 6)


def test_summary_with_scored_and_unscored_runs(compare):
    import pandas as pd

    rows = pd.DataFrame(
        [
            {
                "run": "a",
                "fov": "A_1_0",
                "tracks": 5,
                "divisions": 4,
                "cells_per_frame": 3.0,
                "track_length": 2.0,
                "gt_divisions": 4,
                "matched": 3,
            },
            {
                "run": "b",
                "fov": "B_1_0",
                "tracks": 5,
                "divisions": 2,
                "cells_per_frame": 3.0,
                "track_length": 2.0,
            },
        ]
    )
    out = compare.summarize(rows)

    assert out.loc["a", "div_f1"] == pytest.approx(0.75)
    assert pd.isna(out.loc["b", "div_f1"])


def test_view_runs_division_points(tmp_path):
    """The viewer template loads, and marks one point per division."""
    import pandas as pd

    pytest.importorskip("napari")
    spec = importlib.util.spec_from_file_location("view_runs", SCRIPT.parent / "view_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 3],
            "t": [0, 1, 2, 2],
            "y": [5.0, 6.0, 4.0, 8.0],
            "x": [5.0, 6.0, 4.0, 8.0],
            "parent_track_id": [-1, -1, 1, 1],
        }
    )
    assert module.division_points(tracks).tolist() == [[2.0, 6.0, 6.0]]


def test_view_runs_gives_every_run_four_layers():
    """An empty run and a populated run must fill the same number of panel slots."""
    import numpy as np
    import pandas as pd

    pytest.importorskip("napari")
    spec = importlib.util.spec_from_file_location("view_runs", SCRIPT.parent / "view_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = np.zeros((3, 8, 8), dtype=np.float32)
    labels = np.zeros((3, 8, 8), dtype=np.int32)
    empty = pd.DataFrame(columns=["track_id", "t", "y", "x", "parent_track_id"])
    full = pd.DataFrame(
        {
            "track_id": [1, 1],
            "t": [0, 1],
            "y": [2.0, 2.0],
            "x": [2.0, 2.0],
            "parent_track_id": [-1, -1],
        }
    )

    layers_empty = module.run_layers("empty", image, labels, empty)
    layers_full = module.run_layers("full", image, labels, full)

    assert len(layers_empty) == len(layers_full) == module.LAYERS_PER_RUN == 4
    assert [kind for kind, _, _ in layers_full] == ["image", "labels", "tracks", "points"]
    assert [kind for kind, _, _ in layers_empty] == ["image", "labels", "points", "points"]
