import numpy as np
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate

from biahub.cli.main import cli


@pytest.fixture()
def predict_config(tmp_path):
    cfg = {
        "data": {
            "init_args": {
                "source_channel": "Phase3D",
                "target_channel": ["nuclei", "membrane"],
                "z_window_size": 15,
                "batch_size": 1,
                "num_workers": 0,
            }
        }
    }
    config_path = tmp_path / "predict.yml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_path


@pytest.fixture()
def vs_input_plate(tmp_path):
    plate_path = tmp_path / "input.zarr"
    position_list = (("B", "3", "000000"), ("B", "3", "000001"))

    plate = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["Phase3D"],
    )

    for row, col, fov in position_list:
        position = plate.create_position(row, col, fov)
        position["0"] = np.random.uniform(1.0, 100.0, size=(1, 1, 15, 32, 32)).astype(
            np.float32
        )

    plate.close()
    return plate_path


def test_virtual_stain_cli_init_only(tmp_path, vs_input_plate, predict_config):
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "virtual-stain",
            "--init",
            "-i",
            str(vs_input_plate) + "/B/3/000000",
            str(vs_input_plate) + "/B/3/000001",
            "-c",
            str(predict_config),
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "RESOURCES:" in result.output
    assert "nuclei_prediction" in result.output
    assert "membrane_prediction" in result.output

    with open_ome_zarr(str(output_path / "B" / "3" / "000000"), mode="r") as ds:
        assert ds.channel_names == ["nuclei_prediction", "membrane_prediction"]
        assert ds.data.shape == (1, 2, 15, 32, 32)

    with open_ome_zarr(str(output_path / "B" / "3" / "000001"), mode="r") as ds:
        assert ds.channel_names == ["nuclei_prediction", "membrane_prediction"]


def test_virtual_stain_cli_copy(tmp_path, predict_config):
    output_path = tmp_path / "output.zarr"

    create_empty_plate(
        store_path=output_path,
        position_keys=[("B", "3", "000000")],
        channel_names=["nuclei_prediction", "membrane_prediction"],
        shape=(1, 2, 5, 8, 8),
        scale=(1.0, 1.0, 1.0, 0.5, 0.5),
        version="0.5",
        dtype=np.float32,
    )

    temp_zarr = tmp_path / "temp" / "B_3_000000.zarr"
    create_empty_plate(
        store_path=temp_zarr,
        position_keys=[("B", "3", "000000")],
        channel_names=["nuclei_prediction", "membrane_prediction"],
        shape=(1, 2, 5, 8, 8),
        scale=(1.0, 1.0, 1.0, 0.5, 0.5),
        version="0.5",
        dtype=np.float32,
    )

    test_data = np.random.uniform(0, 1, (1, 2, 5, 8, 8)).astype(np.float32)
    with open_ome_zarr(str(temp_zarr / "B" / "3" / "000000"), mode="r+") as ds:
        ds[0][:] = test_data

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "virtual-stain",
            "--copy",
            "-t",
            str(temp_zarr),
            "-o",
            str(output_path),
            "-p",
            "B/3/000000",
            "-c",
            str(predict_config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Virtual stain copied: B/3/000000" in result.output

    with open_ome_zarr(str(output_path / "B" / "3" / "000000"), mode="r") as ds:
        copied = ds[0][:]
        np.testing.assert_array_equal(copied, test_data)

    assert not temp_zarr.exists(), "temp zarr should be cleaned up"
