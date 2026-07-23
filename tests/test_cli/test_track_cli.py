import os

import numpy as np
import pandas as pd
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli
from biahub.settings import TrackingSettings, ZSlicing
from biahub.track import _init_output_plate, resolve_z_slice, track


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
    out = track_mod.apply_focus_slicing(data_dict, z, pixel_size=0.5)

    # Same fixed window applied to every channel (center=15 -> slice(13, 19)).
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
        # Regression: focus + 3D output Z equals the fixed focus window, not the
        # full stack -- this is the shape-mismatch bug _init_output_plate had.
        ("3D", {"method": "focus", "window_size": 2}, 2),
    ],
)
def test_init_output_plate_shape(
    tmp_path, example_tracking_plate, output_mode, z_slicing, expected_z
):
    plate_path, _ = example_tracking_plate
    output_path = tmp_path / "init_shape.zarr"
    settings = _minimal_settings(output_mode=output_mode, z_slicing=z_slicing)

    shape = _init_output_plate([str(plate_path / "A" / "1" / "0")], output_path, settings)
    assert shape[2] == expected_z
    with open_ome_zarr(str(output_path / "A" / "1" / "0"), mode="r") as ds:
        assert ds.data.shape[2] == expected_z


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
