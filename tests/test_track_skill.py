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
                "detections": 10,
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
                "detections": 10,
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


def test_track_length_is_weighted_by_track_count(compare):
    import pandas as pd

    rows = pd.DataFrame(
        [
            {
                "run": "a",
                "fov": "1",
                "tracks": 1,
                "detections": 100,
                "divisions": 0,
                "cells_per_frame": 1.0,
            },
            {
                "run": "a",
                "fov": "2",
                "tracks": 9,
                "detections": 90,
                "divisions": 0,
                "cells_per_frame": 1.0,
            },
        ]
    )
    assert compare.summarize(rows).loc["a", "track_length"] == pytest.approx(19.0)


def test_runs_found_by_tracked_label_name(compare, tmp_path):
    """A run is labels/<run>; its CSV is tracks_<run>_<fov>.csv, or tracks_<fov>.csv
    for a run named tracks_* (no output_label)."""
    plate = tmp_path / "plate.zarr"
    a, b = plate / "A/1/000000", plate / "B/2/000001"
    for position, run in [(a, "nuclei_hoct"), (a, "tracks_nuclei"), (b, "nuclei_hoct")]:
        (position / "labels" / run).mkdir(parents=True)
    (a / "tracks_nuclei_hoct_A_1_000000.csv").touch()
    (a / "tracks_A_1_000000.csv").touch()
    (b / "tracks_nuclei_hoct_B_2_000001.csv").touch()

    assert compare.run_positions(plate, "nuclei_hoct") == [a, b]
    assert compare.run_positions(plate, "tracks_nuclei") == [a]
    assert compare.tracks_csv(a, "nuclei_hoct").name == "tracks_nuclei_hoct_A_1_000000.csv"
    assert compare.tracks_csv(a, "tracks_nuclei").name == "tracks_A_1_000000.csv"
    # A named run never falls back to another run's default-stem CSV.
    assert compare.tracks_csv(a, "nuclei_trackastra") is None


@pytest.mark.parametrize("shape", [(3, 1, 1, 4, 5), (3, 1, 4, 5)])
def test_view_runs_reads_5d_and_older_4d_labels(tmp_path, shape):
    import numpy as np
    import zarr

    pytest.importorskip("napari")
    spec = importlib.util.spec_from_file_location("view_runs", SCRIPT.parent / "view_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    position = tmp_path / "plate.zarr/A/1/000000"
    data = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
    zarr.create_array(position / "labels/nuclei_hoct/0", data=data)

    labels = module.read_labels(position, "nuclei_hoct")

    assert labels.shape == (3, 4, 5)
    assert (labels == data.reshape(3, 4, 5)).all()
