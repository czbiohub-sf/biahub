import os

import numpy as np
import pandas as pd
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli
from biahub.track import track


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
        "mode": "2D",
        "fov": "*/*/*",
        "z_range": [-1, -1],
        "target_channel": "nuclei_prediction",
        "input_images": [
            {
                "path": str(plate_path),
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

    test_config_path = tmp_path / "test_track_config_missing.yml"
    with open(config_path) as f:
        config_content = f.read()

    config_content = config_content.replace(
        "/path/to/virtual_staining.zarr", "/non/existent/path"
    )

    with open(test_config_path, "w") as f:
        f.write(config_content)

    with pytest.raises((FileNotFoundError, ValueError)):
        track(
            input_position_dirpaths=[str(tmp_path / "nonexistent" / "A" / "1" / "0")],
            output_dirpath=str(output_path),
            config_filepath=str(test_config_path),
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
