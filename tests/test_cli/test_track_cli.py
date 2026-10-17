import os

import numpy as np
import pandas as pd
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli
from biahub.settings import TrackingSettings
from biahub.track import _init_output_plate, track


@pytest.fixture(scope="function")
def example_tracking_plate(tmp_path):
    """
    Create a test plate with nuclei and membrane prediction channels for tracking
    """
    plate_path = tmp_path / "tracking_plate.zarr"

    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
        ("B", "2", "0"),
    )

    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["nuclei_prediction", "membrane_prediction"],
    )

    # Deterministic spots: on random data ultrack occasionally finds no solution, and
    # the failing in-process job drops submitit into pdb (which pytest reports as a
    # read from stdin).
    np.random.seed(0)
    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        # A projected (Z=1) plate, what 2D tracking reads: (T, C, Z, Y, X) = (5, 2, 1, 64, 64)
        data = np.random.uniform(0.1, 0.3, size=(5, 2, 1, 64, 64)).astype(np.float32)

        # Add some bright nuclei spots to channel 0 (11 x 11 px, above the configs'
        # min_area of 100; a single plane no longer merges spots across Z)
        for t in range(5):
            for z in range(1):
                for _ in range(np.random.randint(3, 6)):
                    y, x = np.random.randint(10, 54, 2)
                    data[t, 0, z, y - 5 : y + 6, x - 5 : x + 6] = np.random.uniform(0.7, 1.0)

        # Add some membrane boundaries to channel 1
        for t in range(5):
            for z in range(1):
                for _ in range(np.random.randint(2, 4)):
                    y, x = np.random.randint(15, 49, 2)
                    yy, xx = np.ogrid[:64, :64]
                    mask = (yy - y) ** 2 + (xx - x) ** 2 <= 25
                    data[t, 1, z][mask] = np.random.uniform(0.6, 0.9)

        position["0"] = data

    yield plate_path, plate_dataset


@pytest.fixture(scope="function")
def example_blank_frames_csv(tmp_path):
    """
    Create a CSV file with blank frame information for testing
    """
    csv_path = tmp_path / "blank_frames.csv"

    data = {
        "FOV": ["A_1_0", "B_1_0", "B_2_0"],
        "t": [
            "[0]",
            "[2]",
            "[]",
        ],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    yield csv_path


def _make_tracking_config(plate_path, tmp_path):
    """Create a minimal tracking config pointing at the test plate."""
    config_path = tmp_path / "track_config.yml"
    config = {
        "output_mode": "2D",
        "fov": "*/*/*",
        "target_channel": "nuclei_prediction",
        "input_images": [
            {
                # Primary data source left null -> resolved from the -i input plate.
                # The plate is projected (Z=1), so the channels need no processing.
                "path": None,
                "channels": {"nuclei_prediction": [], "membrane_prediction": []},
            },
            {
                "path": None,
                "channels": {
                    "foreground": [
                        {
                            "function": "ultrack.imgproc.detect_foreground",
                            "input_channels": ["nuclei_prediction"],
                            "kwargs": {"sigma": 90},
                        },
                    ],
                    "contour": [
                        {
                            "function": "biahub.track.mem_nuc_contour",
                            "input_channels": [
                                "nuclei_prediction",
                                "membrane_prediction",
                            ],
                            "kwargs": {},
                        },
                    ],
                },
            },
        ],
        "tracking_config": {
            "segmentation_config": {
                "min_area": 100,
                "max_area": 80000,
                "n_workers": 1,
                "min_frontier": 0.4,
                "max_noise": 0.05,
            },
            "linking_config": {
                "n_workers": 1,
                "max_distance": 15,
                "distance_weight": -0.0001,
                "max_neighbors": 3,
            },
            "tracking_config": {
                "n_threads": 1,
                "disappear_weight": -0.0001,
                "appear_weight": -0.001,
                "division_weight": -0.0001,
            },
        },
    }
    config_path.write_text(yaml.dump(config))
    return config_path


def test_track_cli_local(
    tmp_path, example_tracking_plate, example_track_settings, sbatch_file, monkeypatch
):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")
    os.environ["ULTRACK_ARRAY_MODULE"] = "numpy"

    custom_sbatch_file = tmp_path / "custom_sbatch.txt"
    with open(custom_sbatch_file, "w") as f:
        f.write("#SBATCH --cpus-per-task=1\n")
        f.write("#SBATCH --array-parallelism=1\n")
        f.write("#LOCAL --cpus-per-task=1\n")
        f.write("#LOCAL --timeout-min=5\n")
        f.write("#LOCAL --array-parallelism=1\n")

    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    output_path = tmp_path / "tracking_output.zarr"

    track(
        input_position_dirpaths=[
            str(plate_path / "A" / "1" / "0"),
            str(plate_path / "B" / "1" / "0"),
            str(plate_path / "B" / "2" / "0"),
        ],
        output_dirpath=str(output_path),
        config_filepath=str(config_path),
        sbatch_filepath=str(custom_sbatch_file),
        cluster="local",
    )

    assert output_path.exists()
    for position in ["A/1/0", "B/1/0", "B/2/0"]:
        position_path = output_path / position
        assert position_path.exists()


def test_track_cli_with_blank_frames(
    tmp_path,
    example_tracking_plate,
    example_track_settings,
    example_blank_frames_csv,
    sbatch_file,
    monkeypatch,
):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    output_path = tmp_path / "tracking_output_blank_frames.zarr"

    # Add blank_frames_path to config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["blank_frames_path"] = str(example_blank_frames_csv)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    track(
        input_position_dirpaths=[
            str(plate_path / "A" / "1" / "0"),
            str(plate_path / "B" / "1" / "0"),
            str(plate_path / "B" / "2" / "0"),
        ],
        output_dirpath=str(output_path),
        config_filepath=str(config_path),
        sbatch_filepath=str(sbatch_file),
        cluster="local",
    )

    assert output_path.exists()


def test_track_cli_invalid_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    output_path = tmp_path / "output.zarr"
    invalid_config_path = tmp_path / "invalid_config.yml"

    with open(invalid_config_path, "w") as f:
        f.write("invalid: yaml: content")

    with pytest.raises(Exception):  # noqa: B017
        track(
            input_position_dirpaths=[str(tmp_path / "nonexistent" / "A" / "1" / "0")],
            output_dirpath=str(output_path),
            config_filepath=str(invalid_config_path),
            cluster="local",
        )


def test_track_cli_missing_input_path(tmp_path, example_track_settings, monkeypatch):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    config_path, _ = example_track_settings
    output_path = tmp_path / "output.zarr"

    # -i points at a nonexistent plate, so plate init fails before any tracking.
    with pytest.raises((FileNotFoundError, ValueError, KeyError)):
        track(
            input_position_dirpaths=[str(tmp_path / "nonexistent" / "A" / "1" / "0")],
            output_dirpath=str(output_path),
            config_filepath=str(config_path),
            cluster="local",
        )


def test_track_cli_init_only(tmp_path, example_tracking_plate):
    """Test that --init creates the output store and emits RESOURCES."""
    plate_path, _ = example_tracking_plate
    output_path = tmp_path / "track_output.zarr"
    config_path = _make_tracking_config(plate_path, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "track",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            str(plate_path / "B" / "1" / "0"),
            str(plate_path / "B" / "2" / "0"),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
            "--init",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "RESOURCES:" in result.output

    with open_ome_zarr(str(output_path / "A" / "1" / "0"), mode="r") as ds:
        assert ds.data.dtype == np.uint32
        assert ds.channel_names == ["nuclei_prediction_labels"]


def test_track_cli_debug_single_position(tmp_path, example_tracking_plate, monkeypatch):
    """Test that --cluster debug processes a single position in-process."""
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_tracking_plate
    output_path = tmp_path / "track_output.zarr"
    config_path = _make_tracking_config(plate_path, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "track",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
            "--cluster",
            "debug",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "Tracking complete:" in result.output

    with open_ome_zarr(str(output_path / "A" / "1" / "0"), mode="r") as ds:
        data = ds["0"][:]
        assert data.dtype == np.uint32
        assert data.shape[0] > 0


# ---------------------------------------------------------------------------
# Dimensionality: track neither projects nor selects Z (that is `biahub project`)
# ---------------------------------------------------------------------------


def _minimal_settings(**overrides):
    base = {
        "target_channel": "nuclei_prediction",
        "input_images": [{"path": None, "channels": {"nuclei_prediction": []}}],
    }
    base.update(overrides)
    return TrackingSettings(**base)


def test_init_output_plate_takes_the_input_z(
    tmp_path, example_tracking_plate, create_custom_plate
):
    flat_path, _ = example_tracking_plate  # Z=1
    shape = _init_output_plate(
        [str(flat_path / "A" / "1" / "0")], tmp_path / "flat.zarr", _minimal_settings()
    )
    assert shape[2] == 1
    with open_ome_zarr(str(tmp_path / "flat.zarr" / "A" / "1" / "0"), mode="r") as ds:
        assert ds.data.shape[2] == 1

    stack_path, _ = create_custom_plate(tmp_path, z_size=3)
    shape = _init_output_plate(
        [str(stack_path / "A" / "1" / "0")],
        tmp_path / "stack.zarr",
        _minimal_settings(output_mode="3D"),
    )
    assert shape[2] == 3


@pytest.mark.parametrize(
    "output_mode, z_size, message",
    [("2D", 3, "input is 3D.*biahub project"), ("3D", 1, "one Z-plane")],
)
def test_init_output_plate_refuses_a_dimensionality_mismatch(
    tmp_path, create_custom_plate, output_mode, z_size, message
):
    plate_path, _ = create_custom_plate(tmp_path, z_size=z_size)
    with pytest.raises(ValueError, match=message):
        _init_output_plate(
            [str(plate_path / "A" / "1" / "0")],
            tmp_path / "o.zarr",
            _minimal_settings(output_mode=output_mode),
        )
    assert not (tmp_path / "o.zarr").exists()


def test_track_cli_refuses_a_3d_input_for_2d_tracking(tmp_path, create_custom_plate):
    plate_path, _ = create_custom_plate(tmp_path, z_size=3)
    config_path = _make_tracking_config(plate_path, tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "-i",
            str(plate_path / "A/1/0"),
            "-o",
            str(tmp_path / "o.zarr"),
            "-c",
            str(config_path),
            "--init",
        ],
    )

    assert result.exit_code != 0
    assert "biahub project" in str(result.exception)


def test_z_slicing_is_rejected_with_a_pointer_to_project():
    with pytest.raises(ValueError, match="z_slicing was removed.*biahub project"):
        _minimal_settings(z_slicing={"method": "focus"})


# ---------------------------------------------------------------------------
# Input-path resolution
# ---------------------------------------------------------------------------


def test_input_images_path_override(tmp_path, example_tracking_plate, monkeypatch):
    """A null primary path is filled from --input-images-path when provided."""
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_tracking_plate
    output_path = tmp_path / "override_output.zarr"
    config_path = _make_tracking_config(plate_path, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "track",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
            "--cluster",
            "debug",
            "--input-images-path",
            str(plate_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_track_init_rejects_an_unknown_cellpose_model(
    tmp_path, example_tracking_plate, monkeypatch
):
    """A bad model name must fail once at --init, not later in every worker."""
    import sys
    import types

    class Model:
        def __init__(self, gpu=False, pretrained_model="cpsam_v2", device=None):
            # Cellpose 4 would silently substitute cpsam_v2 here.
            self.pretrained_model = str(tmp_path / "cpsam_v2")

    models = types.ModuleType("cellpose.models")
    models.CellposeModel = Model
    models.MODEL_NAMES = ["cpsam_v2", "cpsam", "cpdino", "cpdino-vitb"]
    models.get_user_models = lambda: []
    package = types.ModuleType("cellpose")
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)

    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["segmentation_method"] = "cellpose"
    config["cellpose_config"] = {
        "pretrained_model": "nuclei",
        "input_channel": "nuclei_prediction",
    }
    config_path.write_text(yaml.safe_dump(config))

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
            "--init",
        ],
    )

    assert result.exit_code != 0
    assert "Unknown cellpose model 'nuclei'" in str(result.exception) + result.output


def test_filter_short_siblings_merges_a_one_frame_daughter():
    """A daughter that lives 1 frame (e.g. a VS hallucination linked as a division) is
    removed and its sibling continues the parent track, in the tracks and the labels."""
    from biahub.track import filter_short_siblings

    labels = np.zeros((6, 8, 8), dtype=np.int32)
    rows = []
    for t in range(3):  # parent, track 1
        labels[t, 1:3, 1:3] = 1
        rows.append((1, t, 2.0, 2.0, -1))
    for t in range(3, 6):  # long daughter, track 2
        labels[t, 1:3, 1:3] = 2
        rows.append((2, t, 2.0, 2.0, 1))
    labels[3, 5:7, 5:7] = 3  # one-frame daughter, track 3
    rows.append((3, 3, 6.0, 6.0, 1))
    tracks = pd.DataFrame(rows, columns=["track_id", "t", "y", "x", "parent_track_id"])

    new_tracks, new_labels = filter_short_siblings(tracks, labels, min_length=2)

    assert set(new_tracks.track_id) == {1}
    assert (new_tracks.parent_track_id == -1).all()
    assert not (new_labels == 3).any()
    assert (new_labels[:, 1:3, 1:3] == 1).all()  # the parent continues through t=5


def test_min_sibling_length_is_off_by_default(example_track_settings):
    _, settings = example_track_settings
    settings.pop("min_sibling_length", None)
    assert TrackingSettings(**settings).min_sibling_length is None


def test_track_passes_min_sibling_length_to_each_position(
    tmp_path, example_tracking_plate, monkeypatch
):
    import contextlib
    import types

    submitted = {}

    class Executor:
        def __init__(self, folder, cluster):
            pass

        def update_parameters(self, **kwargs):
            pass

        def batch(self):
            return contextlib.nullcontext()

        def submit(self, fn, **kwargs):
            submitted.update(kwargs)
            return types.SimpleNamespace(job_id="0", wait=lambda: None)

    monkeypatch.setattr("biahub.track.submitit.AutoExecutor", Executor)
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["min_sibling_length"] = 3
    config_path.write_text(yaml.safe_dump(config))

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "debug",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert submitted["min_sibling_length"] == 3


def _lineage(short_len, long_len=3, parent_len=3):
    """Parent track 1, daughters 2 (long_len frames) and 3 (short_len frames)."""
    T = parent_len + max(long_len, short_len)
    labels = np.zeros((T, 8, 8), dtype=np.int32)
    rows = []
    for t in range(parent_len):
        labels[t, 1:3, 1:3] = 1
        rows.append((1, t, 2.0, 2.0, -1))
    for t in range(parent_len, parent_len + long_len):
        labels[t, 1:3, 1:3] = 2
        rows.append((2, t, 2.0, 2.0, 1))
    for t in range(parent_len, parent_len + short_len):
        labels[t, 5:7, 5:7] = 3
        rows.append((3, t, 6.0, 6.0, 1))
    return pd.DataFrame(rows, columns=["track_id", "t", "y", "x", "parent_track_id"]), labels


def test_filter_removes_daughters_of_min_length_or_fewer():
    """Same meaning as ultrack: a daughter of exactly ``min_length`` frames is removed."""
    from biahub.track import filter_short_siblings

    tracks, labels = _lineage(short_len=3, long_len=5)
    new_tracks, new_labels = filter_short_siblings(tracks, labels, min_length=3)
    assert set(new_tracks.track_id) == {1}
    assert not (new_labels == 3).any()

    tracks, labels = _lineage(short_len=4, long_len=5)
    new_tracks, _ = filter_short_siblings(tracks, labels, min_length=3)
    assert set(new_tracks.track_id) == {1, 2, 3}


def test_filter_keeps_a_division_when_both_daughters_are_short():
    from biahub.track import filter_short_siblings

    tracks, labels = _lineage(short_len=2, long_len=2)
    new_tracks, _ = filter_short_siblings(tracks, labels, min_length=3)
    assert set(new_tracks.track_id) == {1, 2, 3}


def test_filter_leaves_no_temporary_files_and_sorts_rows(tmp_path, monkeypatch):
    import tempfile

    from biahub.track import filter_short_siblings

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)  # re-read TMPDIR
    tracks, labels = _lineage(short_len=1, long_len=4)

    new_tracks, _ = filter_short_siblings(tracks, labels, min_length=2)

    assert list(tmp_path.iterdir()) == []
    assert new_tracks[["track_id", "t"]].equals(
        new_tracks[["track_id", "t"]].sort_values(["track_id", "t"]).reset_index(drop=True)
    )


def test_track_debug_run_applies_the_sibling_filter(tmp_path, example_tracking_plate):
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["min_sibling_length"] = 3
    config_path.write_text(yaml.safe_dump(config))
    position = str(plate_path / "A" / "1" / "0")
    out = str(tmp_path / "out.zarr")

    assert (
        CliRunner()
        .invoke(cli, ["track", "--init", "-i", position, "-o", out, "-c", str(config_path)])
        .exit_code
        == 0
    )
    result = CliRunner().invoke(
        cli, ["track", "--cluster", "debug", "-i", position, "-o", out, "-c", str(config_path)]
    )

    assert result.exit_code == 0, result.output
    assert "Sibling filter (min_length=3)" in result.output


def test_track_gives_local_jobs_a_gpu(tmp_path, example_tracking_plate, monkeypatch):
    """Same as segment: submitit's local executor hides GPUs without gpus_per_node."""
    import contextlib

    recorded = {}

    class Executor:
        def __init__(self, folder, cluster):
            recorded["cluster"] = cluster

        def update_parameters(self, **kwargs):
            recorded.update(kwargs)

        def batch(self):
            return contextlib.nullcontext()

        def submit(self, *args, **kwargs):
            return types.SimpleNamespace(job_id="0", wait=lambda: None)

    import types

    monkeypatch.setattr("biahub.track.submitit.AutoExecutor", Executor)
    monkeypatch.delenv("CI", raising=False)
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "local",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded["cluster"] == "local"
    assert recorded.get("gpus_per_node") == 1
    assert (
        recorded.get("timeout_min") == recorded["slurm_time"]
    )  # local default kills at 2 min


def test_track_local_sbatch_file_overrides_computed_parameters(
    tmp_path, example_tracking_plate, monkeypatch
):
    """'#LOCAL --timeout-min' must win over the computed local timeout, not collide
    (TypeError: multiple values for keyword argument 'timeout_min')."""
    import contextlib
    import types

    recorded = {}

    class Executor:
        def __init__(self, folder, cluster):
            recorded["cluster"] = cluster

        def update_parameters(self, **kwargs):
            recorded.update(kwargs)

        def batch(self):
            return contextlib.nullcontext()

        def submit(self, *args, **kwargs):
            return types.SimpleNamespace(job_id="0", wait=lambda: None)

    monkeypatch.setattr("biahub.track.submitit.AutoExecutor", Executor)
    monkeypatch.delenv("CI", raising=False)  # conftest forces CI (debug) otherwise
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    sbatch = tmp_path / "local.sbatch"
    sbatch.write_text("#LOCAL --timeout-min=5\n")

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "local",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
            "-sb",
            str(sbatch),
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded["timeout_min"] == 5
    assert recorded["gpus_per_node"] == 1


def test_track_local_timeout_ignores_a_slurm_time_string(
    tmp_path, example_tracking_plate, monkeypatch
):
    """'#SBATCH --time=3:00:00' sets slurm_time to a string; the local timeout_min must
    stay the computed number of minutes."""
    import contextlib
    import types

    recorded = {}

    class Executor:
        def __init__(self, folder, cluster):
            recorded["cluster"] = cluster

        def update_parameters(self, **kwargs):
            recorded.update(kwargs)

        def batch(self):
            return contextlib.nullcontext()

        def submit(self, *args, **kwargs):
            return types.SimpleNamespace(job_id="0", wait=lambda: None)

    monkeypatch.setattr("biahub.track.submitit.AutoExecutor", Executor)
    monkeypatch.delenv("CI", raising=False)  # conftest forces CI (debug) otherwise
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    sbatch = tmp_path / "slurm.sbatch"
    sbatch.write_text("#SBATCH --time=3:00:00\n")

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "local",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
            "-sb",
            str(sbatch),
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded["slurm_time"] == "3:00:00"
    assert recorded["timeout_min"] == 60  # track's computed time, in minutes


def test_track_mem_per_cpu_override_drops_the_computed_total(
    tmp_path, example_tracking_plate, monkeypatch
):
    """'#SBATCH --mem-per-cpu=16G' left --mem set too, and SLURM rejects both."""
    import contextlib
    import types

    recorded = {}

    class Executor:
        def __init__(self, folder, cluster):
            recorded["cluster"] = cluster

        def update_parameters(self, **kwargs):
            recorded.update(kwargs)

        def batch(self):
            return contextlib.nullcontext()

        def submit(self, *args, **kwargs):
            return types.SimpleNamespace(job_id="0", wait=lambda: None)

    monkeypatch.setattr("biahub.track.submitit.AutoExecutor", Executor)
    monkeypatch.delenv("CI", raising=False)  # conftest forces CI (debug) otherwise
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    sbatch = tmp_path / "mem.sbatch"
    sbatch.write_text("#SBATCH --mem-per-cpu=16G\n")

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "slurm",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
            "-sb",
            str(sbatch),
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded["slurm_mem_per_cpu"] == "16G"
    assert "slurm_mem" not in recorded
