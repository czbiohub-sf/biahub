import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli
from biahub.settings import TrackingSettings, ZSlicing
from biahub.track import (
    _init_labels,
    _tracked_shape,
    label_chunks_and_shards,
    resolve_z_slice,
    track,
    tracked_z_window,
    write_tracking_labels,
)

LABEL = "nuclei_prediction"


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

    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        # Shape: (T, C, Z, Y, X) = (5, 2, 3, 64, 64)
        data = np.random.uniform(0.1, 0.3, size=(5, 2, 3, 64, 64)).astype(np.float32)

        # Add some bright nuclei spots to channel 0
        for t in range(5):
            for z in range(3):
                for _ in range(np.random.randint(3, 6)):
                    y, x = np.random.randint(10, 54, 2)
                    data[t, 0, z, y - 3 : y + 4, x - 3 : x + 4] = np.random.uniform(0.7, 1.0)

        # Add some membrane boundaries to channel 1
        for t in range(5):
            for z in range(3):
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


def _assert_tracking_outputs(position_path, z_index):
    """Labels at z_index only, a valid GEFF linked to them, and the CSV."""
    import geff

    with open_ome_zarr(str(position_path), mode="r") as pos:
        labels = pos.get_label(LABEL)["0"][:]
    assert labels.dtype == np.uint32
    assert labels[:, z_index].any()
    assert not np.delete(labels, z_index, axis=1).any()

    geff_path = position_path / "tracks.geff"
    geff.validate_structure(str(geff_path))
    _, metadata = geff.read(str(geff_path), backend="networkx")
    (related,) = metadata.related_objects
    assert related.path == f"../labels/{LABEL}"
    assert related.node_prop == "seg_id"

    (csv_path,) = position_path.glob("tracks_*.csv")
    tracks = pd.read_csv(csv_path)
    assert set(tracks["track_id"]) <= set(np.unique(labels)) - {0}

    # The full Ultrack config is provenance on the label image, not a file beside it.
    label_attrs = yaml.safe_load((position_path / "labels" / LABEL / "zarr.json").read_text())
    ultrack_config = label_attrs["attributes"]["biahub-track"]["ultrack_config"]
    assert "solution_gap" in ultrack_config["tracking"]
    assert "working_dir" not in ultrack_config["data"]


def _make_tracking_config(plate_path, tmp_path):
    """Create a minimal tracking config pointing at the test plate."""
    config_path = tmp_path / "track_config.yml"
    config = {
        "output_mode": "2D",
        "fov": "*/*/*",
        "z_slicing": {"method": "central"},
        "target_channel": "nuclei_prediction",
        "input_images": [
            {
                # Primary data source left null -> resolved from the -i input plate.
                "path": None,
                "channels": {
                    "nuclei_prediction": [
                        {
                            "function": "np.mean",
                            "kwargs": {"axis": 1},
                            "per_timepoint": False,
                        },
                    ],
                    "membrane_prediction": [
                        {
                            "function": "np.mean",
                            "kwargs": {"axis": 1},
                            "per_timepoint": False,
                        },
                    ],
                },
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
    output_path = tmp_path / "tracking_output"

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

    # central z_slicing on Z=3 tracks planes [0, 3), so the 2D labels sit at z=1.
    for position in ["A/1/0", "B/1/0", "B/2/0"]:
        _assert_tracking_outputs(plate_path / position, z_index=1)
        assert (output_path / position.replace("/", "_")).is_dir()  # Ultrack database


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
    output_path = tmp_path / "tracking_output_blank_frames"

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

    _assert_tracking_outputs(plate_path / "A" / "1" / "0", z_index=1)


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
    """--init creates empty full-Z label images in the -i positions and emits RESOURCES."""
    plate_path, _ = example_tracking_plate
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
            "-c",
            str(config_path),
            "--init",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "RESOURCES:" in result.output

    for position in ["A/1/0", "B/1/0", "B/2/0"]:
        with open_ome_zarr(str(plate_path / position), mode="r") as pos:
            assert pos.label_names() == [LABEL]
            assert pos.channel_names == ["nuclei_prediction", "membrane_prediction"]
            label = pos.get_label(LABEL)
            assert label["0"].shape == (5, 3, 64, 64)
            assert label["0"].dtype == np.uint32
            assert not label["0"][:].any()
            assert [ax.name.lower() for ax in label.axes] == ["t", "z", "y", "x"]
        label_attrs = yaml.safe_load(
            (plate_path / position / "labels" / LABEL / "zarr.json").read_text()
        )["attributes"]
        assert label_attrs["biahub-track"]["target_channel"] == LABEL
        # The labels listing lives on the labels group (OME-NGFF 0.5).
        labels_attrs = yaml.safe_load(
            (plate_path / position / "labels" / "zarr.json").read_text()
        )["attributes"]
        assert labels_attrs["ome"]["labels"] == [LABEL]


def test_track_cli_debug_single_position(tmp_path, example_tracking_plate, monkeypatch):
    """Test that --cluster debug processes a single position in-process."""
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    # No -o: the Ultrack database goes to $TMPDIR and the submitit folder to ./
    temp_dir = tmp_path / "tmpdir"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    monkeypatch.chdir(tmp_path)

    def run():
        return CliRunner().invoke(
            cli,
            [
                "track",
                "-i",
                str(plate_path / "A" / "1" / "0"),
                "-c",
                str(config_path),
                "--cluster",
                "debug",
            ],
        )

    result = run()
    assert result.exit_code == 0, result.output
    assert "Tracking complete:" in result.output
    _assert_tracking_outputs(plate_path / "A" / "1" / "0", z_index=1)
    assert not any(temp_dir.iterdir())  # the temporary database was deleted
    assert (tmp_path / "slurm_output").is_dir()

    # A retry overwrites the same outputs.
    result = run()
    assert result.exit_code == 0, result.output
    _assert_tracking_outputs(plate_path / "A" / "1" / "0", z_index=1)


# ---------------------------------------------------------------------------
# Z-slicing resolution
# ---------------------------------------------------------------------------


def test_resolve_z_slice_all():
    z_slices, n = resolve_z_slice(ZSlicing(method="all"), z_shape=30)
    assert z_slices == slice(None)
    assert n == 30


def test_resolve_z_slice_central():
    z_slices, n = resolve_z_slice(ZSlicing(method="central"), z_shape=21)
    assert n == z_slices.stop - z_slices.start
    assert n >= 3


def test_resolve_z_slice_range():
    z_slices, n = resolve_z_slice(ZSlicing(method="range", range=(5, 10)), z_shape=30)
    assert z_slices == slice(5, 10)
    assert n == 5


def test_resolve_z_slice_range_invalid():
    with pytest.raises(ValueError):
        resolve_z_slice(ZSlicing(method="range", range=(10, 5)), z_shape=30)


def test_resolve_z_slice_focus_loads_full_reports_window():
    # focus loads the full stack at read time; the count is the fixed window size.
    z_slices, n = resolve_z_slice(ZSlicing(method="focus", window_size=15), z_shape=30)
    assert z_slices == slice(None)
    assert n == 15
    # window larger than the stack collapses the count to the full depth.
    _, n = resolve_z_slice(ZSlicing(method="focus", window_size=50), z_shape=30)
    assert n == 30


def test_focus_window_fixed_size_and_shifts_at_edges():
    from biahub.track import _focus_window

    # centred window of the requested size.
    assert _focus_window(15, 6, 30, 1 / 3) == (slice(13, 19), 6)
    # window bigger than the stack collapses to the full range.
    assert _focus_window(15, 50, 30, 1 / 3) == (slice(0, 30), 30)
    # a window that would spill past an edge is shifted, not clipped.
    z_slices, n = _focus_window(29, 6, 30, 0.0)
    assert n == 6
    assert 0 <= z_slices.start and z_slices.stop <= 30


def test_apply_focus_slicing_uniform_window(monkeypatch):
    from biahub import track as track_mod

    # Deterministic focus centre so the test doesn't depend on waveorder.
    monkeypatch.setattr(track_mod, "_median_focus_plane", lambda stack, pixel_size: 15)

    T, Z, Y, X = 4, 30, 8, 8
    data_dict = {
        "a": np.arange(T * Z * Y * X).reshape(T, Z, Y, X).astype(float),
        "b": np.zeros((T, Z, Y, X)),
    }
    z = ZSlicing(method="focus", window_size=6, frac_below=1 / 3)
    out, window = track_mod.apply_focus_slicing(data_dict, z, pixel_size=0.5)

    # Same fixed window applied to every channel (center=15 -> slice(13, 19)).
    assert window == slice(13, 19)
    assert out["a"].shape == (T, 6, Y, X)
    assert out["b"].shape == (T, 6, Y, X)
    assert np.array_equal(out["a"], data_dict["a"][:, 13:19])


def test_zslicing_focus_defaults_window_size():
    # focus works without an explicit window_size (defaults to 48).
    z = ZSlicing(method="focus")
    assert z.window_size == 48


def test_zslicing_ignores_irrelevant_fields():
    # method decides which fields are used; the rest are ignored, not rejected.
    z_slices, n = resolve_z_slice(
        ZSlicing(method="all", range=(0, 5), window_size=10), z_shape=30
    )
    assert z_slices == slice(None)
    assert n == 30


def test_resolve_z_slice_range_unset_falls_back_to_all():
    z_slices, n = resolve_z_slice(ZSlicing(method="range"), z_shape=30)
    assert z_slices == slice(None)
    assert n == 30


# ---------------------------------------------------------------------------
# Output plate shape matches tracked Z
# ---------------------------------------------------------------------------


def _minimal_settings(**overrides):
    base = {
        "target_channel": "nuclei_prediction",
        "input_images": [
            {
                "path": None,
                "channels": {
                    "nuclei_prediction": [{"function": "np.mean", "kwargs": {"axis": 1}}]
                },
            }
        ],
    }
    base.update(overrides)
    return TrackingSettings(**base)


@pytest.mark.parametrize(
    "output_mode, z_slicing, expected_z",
    [
        ("2D", {"method": "central"}, 1),
        ("3D", {"method": "range", "range": (0, 2)}, 2),
        # Regression: focus + 3D tracked Z equals the fixed focus window, not the
        # full stack.
        ("3D", {"method": "focus", "window_size": 2}, 2),
    ],
)
def test_init_labels_shape(
    tmp_path, example_tracking_plate, output_mode, z_slicing, expected_z
):
    plate_path, _ = example_tracking_plate
    position_path = plate_path / "A" / "1" / "0"
    settings = _minimal_settings(output_mode=output_mode, z_slicing=z_slicing)

    assert _tracked_shape(position_path, settings)[2] == expected_z
    # The label image always spans the image's full Z.
    _init_labels([position_path], settings)
    with open_ome_zarr(str(position_path), mode="r") as pos:
        assert pos.get_label(LABEL)["0"].shape == (5, 3, 64, 64)


def test_init_labels_leaves_existing_labels(tmp_path, example_tracking_plate):
    plate_path, _ = example_tracking_plate
    position_path = plate_path / "A" / "1" / "0"
    settings = _minimal_settings()

    _init_labels([position_path], settings)
    with open_ome_zarr(str(position_path), mode="r+") as pos:
        pos.get_label(LABEL)["0"][0, 1, 0, 0] = 7
        image = pos["0"][:]
    _init_labels([position_path], settings)
    with open_ome_zarr(str(position_path), mode="r") as pos:
        assert pos.get_label(LABEL)["0"][0, 1, 0, 0] == 7
        assert np.array_equal(pos["0"][:], image)


def test_label_chunks_and_shards():
    # 1 MB uint32 chunks (DCA SHOULD); shards cover YX up to 16 chunks.
    chunks, shards = label_chunks_and_shards((67, 86, 1664, 1193))
    assert chunks == (16, 1, 128, 128)
    assert shards == (1, 1, 13, 10)
    assert label_chunks_and_shards((67, 86, 4096, 300))[1] == (1, 1, 16, 3)
    # Clamped to small arrays.
    assert label_chunks_and_shards((5, 3, 64, 64))[0] == (5, 1, 64, 64)


@pytest.mark.parametrize(
    "read_slice, focus_slice, z_shape, expected",
    [
        (slice(None), None, 30, (0, 30)),  # all
        (slice(9, 12), None, 21, (9, 12)),  # central / range
        (slice(10, 11), None, 30, (10, 11)),  # one plane
        (slice(None), slice(13, 19), 30, (13, 19)),  # focus window
    ],
)
def test_tracked_z_window(read_slice, focus_slice, z_shape, expected):
    assert tracked_z_window(read_slice, focus_slice, z_shape) == expected


@pytest.mark.parametrize(
    "z_window, expected_z",
    [
        ((0, 30), 15),  # projection of the whole volume -> middle of the volume
        ((10, 11), 10),  # one plane -> that plane
        ((13, 19), 16),  # projected section -> middle of the section
    ],
)
def test_write_tracking_labels_2d_placement(z_window, expected_z):
    label_array = np.zeros((2, 30, 4, 4), dtype=np.uint32)
    labels = np.ones((2, 4, 4), dtype=np.int64)

    assert write_tracking_labels(label_array, labels, z_window, "2D") == expected_z
    assert label_array[:, expected_z].all()
    assert label_array.sum() == labels.sum()


def test_write_tracking_labels_2d_unprojected_single_plane():
    label_array = np.zeros((2, 30, 4, 4), dtype=np.uint32)
    write_tracking_labels(label_array, np.ones((2, 1, 4, 4)), (10, 11), "2D")
    assert label_array[:, 10].all()


def test_write_tracking_labels_3d_fills_window():
    label_array = np.zeros((2, 30, 4, 4), dtype=np.uint32)
    labels = np.ones((2, 6, 4, 4), dtype=np.int64)

    assert write_tracking_labels(label_array, labels, (13, 19), "3D") == slice(13, 19)
    assert label_array[:, 13:19].all()
    assert label_array.sum() == labels.sum()
    with pytest.raises(ValueError, match="3D"):
        write_tracking_labels(label_array, np.ones((2, 5, 4, 4)), (13, 19), "3D")


# ---------------------------------------------------------------------------
# Input-path resolution
# ---------------------------------------------------------------------------


def test_input_images_path_override(tmp_path, example_tracking_plate, monkeypatch):
    """A null primary path is filled from --input-images-path when provided."""
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_tracking_plate
    output_path = tmp_path / "override_output"
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
    _assert_tracking_outputs(plate_path / "A" / "1" / "0", z_index=1)


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
            str(tmp_path / "out"),
            "-c",
            str(config_path),
            "--init",
        ],
    )

    assert result.exit_code != 0
    assert "Unknown cellpose model 'nuclei'" in str(result.exception) + result.output
