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

    resolved = tmp_path / "reconstruct_resolved.yml"
    assert resolved.exists()
    with open(resolved) as f:
        cfg = yaml.safe_load(f)
    tf_cfg = cfg["phase"]["transfer_function"]
    assert tf_cfg["yx_pixel_size"] is not None
    assert tf_cfg["z_pixel_size"] is not None


def test_apply_inv_tf_cli_debug_single_position(
    tmp_path, reconstruct_plate, reconstruct_config
):
    output_path = tmp_path / "output.zarr"
    tf_path = tmp_path / "tf.zarr"

    runner = CliRunner()

    # Step 1: init
    result = runner.invoke(
        cli,
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
    )
    assert result.exit_code == 0, result.output

    resolved_config = tmp_path / "reconstruct_resolved.yml"

    # Step 2: compute TF (using the standalone compute-tf command)
    result = runner.invoke(
        cli,
        [
            "compute-tf",
            "-i",
            str(reconstruct_plate) + "/A/1/0",
            "-c",
            str(resolved_config),
            "-o",
            str(tf_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert tf_path.exists()

    # Step 3: apply inv TF in debug mode
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
            str(resolved_config),
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Apply-inv-tf done:" in result.output

    with open_ome_zarr(str(output_path / "A" / "1" / "0"), mode="r") as ds:
        data = ds["0"][:]
        assert data.shape[0] > 0
