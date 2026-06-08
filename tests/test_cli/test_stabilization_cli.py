import numpy as np
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli


@pytest.fixture()
def stabilize_config(tmp_path):
    """Config with 3 identity transforms (matching example_plate T=3)."""
    config_path = tmp_path / "stabilize.yml"
    config = {
        "stabilization_estimation_channel": "GFP",
        "stabilization_type": "xyz",
        "stabilization_method": "focus-finding",
        "stabilization_channels": ["GFP", "RFP"],
        "affine_transform_zyx_list": [np.eye(4).tolist() for _ in range(3)],
        "time_indices": "all",
        "output_voxel_size": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_stabilize_init_only(tmp_path, example_plate, stabilize_config):
    plate_path, _ = example_plate
    output_path = tmp_path / "stabilized.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stabilize",
            "--init",
            "-i",
            str(plate_path) + "/A/1/0",
            "-o",
            str(output_path),
            "-c",
            str(stabilize_config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "RESOURCES:" in result.output
    assert "Initialized" in result.output
    assert output_path.exists()

    with open_ome_zarr(str(output_path / "A/1/0"), mode="r") as ds:
        assert ds.data.shape[0] == 3  # T=3


def test_stabilize_cluster_debug(tmp_path, example_plate, stabilize_config):
    plate_path, _ = example_plate
    output_path = tmp_path / "stabilized.zarr"
    positions = ["A/1/0", "B/1/0", "B/2/0"]

    runner = CliRunner()
    # Init with all positions
    result = runner.invoke(
        cli,
        [
            "stabilize",
            "--init",
            "-i",
            *[str(plate_path / p) for p in positions],
            "-o",
            str(output_path),
            "-c",
            str(stabilize_config),
        ],
    )
    assert result.exit_code == 0, result.output

    # Run single position with --cluster debug
    result = runner.invoke(
        cli,
        [
            "stabilize",
            "--cluster",
            "debug",
            "-i",
            str(plate_path / "A/1/0"),
            "-o",
            str(output_path),
            "-c",
            str(stabilize_config),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Stabilization complete" in result.output

    with open_ome_zarr(str(output_path / "A/1/0"), mode="r") as ds:
        assert ds.data.shape == (3, 6, 4, 5, 6)
        assert not np.all(ds.data[:] == 0)


def test_stabilize_backward_compat(
    tmp_path, example_plate, example_stabilize_timelapse_settings, sbatch_file
):
    """Existing test: --local still works (mapped to cluster='local' internally)."""
    plate_path, _ = example_plate
    config_path, _ = example_stabilize_timelapse_settings
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stabilize",
            "-i",
            str(plate_path) + "/A/1/0",
            "-o",
            str(output_path),
            "-c",
            str(config_path),
            "--cluster",
            "debug",
        ],
    )

    assert output_path.exists()
    assert result.exit_code == 0


def test_combine_transforms(tmp_path):
    config_a_path = tmp_path / "config_a.yml"
    config_b_path = tmp_path / "config_b.yml"
    output_path = tmp_path / "combined.yml"

    # A: translate Z by +2
    transform_a = np.eye(4)
    transform_a[0, 3] = 2.0
    # B: translate Y by +3
    transform_b = np.eye(4)
    transform_b[1, 3] = 3.0

    config_template = {
        "stabilization_estimation_channel": "GFP",
        "stabilization_type": "xyz",
        "stabilization_method": "focus-finding",
        "stabilization_channels": ["GFP"],
        "affine_transform_zyx_list": None,
        "time_indices": "all",
        "output_voxel_size": [1.0, 1.0, 1.0, 1.0, 1.0],
    }

    config_a = {**config_template, "affine_transform_zyx_list": [transform_a.tolist()] * 2}
    config_b = {**config_template, "affine_transform_zyx_list": [transform_b.tolist()] * 2}

    with open(config_a_path, "w") as f:
        yaml.dump(config_a, f)
    with open(config_b_path, "w") as f:
        yaml.dump(config_b, f)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "combine-transforms",
            "-a",
            str(config_a_path),
            "-b",
            str(config_b_path),
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()

    with open(output_path) as f:
        combined = yaml.safe_load(f)
    composed = np.array(combined["affine_transform_zyx_list"][0])
    expected = transform_a @ transform_b
    np.testing.assert_array_almost_equal(composed, expected)


@pytest.fixture()
def estimate_z_config(tmp_path):
    """Config for z-focus estimation."""
    config_path = tmp_path / "estimate_z.yml"
    config = {
        "stabilization_estimation_channel": "GFP",
        "stabilization_type": "z",
        "stabilization_method": "focus-finding",
        "stabilization_channels": ["GFP"],
        "focus_finding_settings": {
            "center_crop_xy": [4, 4],
        },
        "verbose": False,
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_estimate_stabilization_step_z_focus(tmp_path, example_plate, estimate_z_config):
    plate_path, _ = example_plate
    output_path = tmp_path / "estimate_output"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "estimate-stabilization",
            "--step",
            "z-focus",
            "-i",
            str(plate_path) + "/A/1/0",
            "-o",
            str(output_path),
            "-c",
            str(estimate_z_config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Z-focus estimation done" in result.output
    assert (output_path / "z_focus_positions").exists()
    assert (output_path / "z_transforms").exists()
