import numpy as np
import pytest
import yaml

from iohub.ngff import open_ome_zarr
from typer.testing import CliRunner

from biahub.cli.main import app


@pytest.fixture()
def flat_field_config(tmp_path):
    config_path = tmp_path / "flat_field.yml"
    config_path.write_text(yaml.dump({"channel_names": None}))
    return config_path


def test_flat_field_cli(tmp_path, example_plate, flat_field_config, sbatch_file):
    plate_path, _ = example_plate
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "flat-field",
            "-i",
            str(plate_path) + "/A/1/0",
            str(plate_path) + "/B/1/0",
            str(plate_path) + "/B/2/0",
            "-c",
            str(flat_field_config),
            "-o",
            str(output_path),
            "--cluster",
            "local",
            "--sbatch-filepath",
            str(sbatch_file),
        ],
    )

    assert output_path.exists(), result.output
    assert result.exit_code == 0, result.output


def test_flat_field_cli_init_only(tmp_path, example_plate, flat_field_config):
    plate_path, _ = example_plate
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "flat-field",
            "-i",
            str(plate_path) + "/A/1/0",
            str(plate_path) + "/B/1/0",
            str(plate_path) + "/B/2/0",
            "-c",
            str(flat_field_config),
            "-o",
            str(output_path),
            "--init",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "RESOURCES:" in result.output


def test_flat_field_cli_debug_single_position(tmp_path, example_plate, flat_field_config):
    plate_path, _ = example_plate
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "flat-field",
            "-i",
            str(plate_path) + "/A/1/0",
            "-c",
            str(flat_field_config),
            "-o",
            str(output_path),
            "--cluster",
            "debug",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "Flat-field complete:" in result.output

    with open_ome_zarr(str(output_path / "A" / "1" / "0"), mode="r") as ds:
        data = ds["0"][:]
        assert data.dtype == np.float32
        assert data.shape[0] > 0


def test_flat_field_invalid_channel_is_a_clean_cli_error(tmp_path, example_plate):
    plate_path, _ = example_plate
    config_path = tmp_path / "invalid_flat_field.yml"
    config_path.write_text(yaml.dump({"channel_names": ["missing"]}))

    result = CliRunner().invoke(
        app,
        [
            "flat-field",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-c",
            str(config_path),
            "-o",
            str(tmp_path / "output.zarr"),
            "--cluster",
            "debug",
        ],
    )

    assert result.exit_code == 1
    assert "Error: Channel 'missing' not found" in result.stderr
    assert "Traceback" not in result.output
