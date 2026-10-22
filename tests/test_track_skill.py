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
