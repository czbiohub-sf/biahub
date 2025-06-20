from click.testing import CliRunner

from biahub.cli.main import cli


def test_estimate_stabilization(
    tmp_path, example_plate, example_estimate_stabilization_settings
):
    plate_path, _ = example_plate
    output_path = tmp_path / "z_stabilization_settings"
    config_path, _ = example_estimate_stabilization_settings
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "estimate-stabilization",
            "-i",
            str(plate_path) + "/A/1/0",
            "-o",
            str(output_path),
            "-c",
            str(config_path),
            "--local",
        ],
    )

    # Weak test
    assert "stabilization_type='z" in result.output
    assert result.exit_code == 0


def test_apply_stabilization(tmp_path, example_plate, example_stabilize_timelapse_settings):
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
            "--local",
        ],
    )

    # Weak test
    assert output_path.exists()
    assert result.exit_code == 0
