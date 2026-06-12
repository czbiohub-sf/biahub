import numpy as np
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli


@pytest.fixture()
def reconstruct_config(tmp_path):
    cfg = {
        "input_channel_names": ["Phase3D"],
        "time_indices": "all",
        "reconstruction_dimension": 3,
        "phase": {
            "transfer_function": {
                "wavelength_illumination": 0.450,
                "yx_pixel_size": 0.1,
                "z_pixel_size": 0.25,
                "z_padding": 0,
                "index_of_refraction_media": 1.3,
                "numerical_aperture_detection": 1.2,
                "numerical_aperture_illumination": 0.5,
                "invert_phase_contrast": False,
            },
            "apply_inverse": {
                "reconstruction_algorithm": "Tikhonov",
                "regularization_strength": 1e-3,
            },
        },
    }
    config_path = tmp_path / "reconstruct.yml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_path


@pytest.fixture()
def reconstruct_plate(tmp_path):
    """Small plate with a Phase3D channel for reconstruction tests."""
    plate_path = tmp_path / "input.zarr"

    position_list = (("A", "1", "0"), ("B", "1", "0"))

    plate = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["Phase3D"],
    )

    for row, col, fov in position_list:
        position = plate.create_position(row, col, fov)
        position["0"] = np.random.uniform(1.0, 100.0, size=(1, 1, 5, 8, 8)).astype(np.float32)

    plate.close()
    return plate_path


def test_apply_inv_tf_cli_init_only(tmp_path, reconstruct_plate, reconstruct_config):
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "apply-inv-tf",
            "--init",
            "-i",
            str(reconstruct_plate) + "/A/1/0",
            str(reconstruct_plate) + "/B/1/0",
            "-c",
            str(reconstruct_config),
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "RESOURCES:" in result.output


def test_apply_inv_tf_cli_debug_single_position(
    tmp_path, reconstruct_plate, reconstruct_config
):
    """Exercise biahub's --cluster debug submitit fan-out path.

    compute-tf (a thin waveorder pass-through) and the reconstruction math are
    covered upstream in waveorder; here we only assert that biahub's debug
    orchestration runs to completion against a real position.
    """
    output_path = tmp_path / "output.zarr"
    tf_path = tmp_path / "tf.zarr"

    runner = CliRunner()

    # Setup: init the output plate, then compute a TF to feed the debug run.
    for cmd in (
        [
            "apply-inv-tf",
            "--init",
            "-i",
            str(reconstruct_plate) + "/A/1/0",
            "-c",
            str(reconstruct_config),
            "-o",
            str(output_path),
        ],
        [
            "compute-tf",
            "-i",
            str(reconstruct_plate) + "/A/1/0",
            "-c",
            str(reconstruct_config),
            "-o",
            str(tf_path),
        ],
    ):
        result = runner.invoke(cli, cmd)
        assert result.exit_code == 0, result.output

    # Apply inv TF in debug mode (the biahub-specific in-process fan-out).
    result = runner.invoke(
        cli,
        [
            "apply-inv-tf",
            "--cluster",
            "debug",
            "-i",
            str(reconstruct_plate) + "/A/1/0",
            "-t",
            str(tf_path),
            "-c",
            str(reconstruct_config),
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Apply-inv-tf complete:" in result.output
