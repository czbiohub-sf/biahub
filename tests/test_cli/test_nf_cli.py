import yaml

from click.testing import CliRunner

from biahub.cli.main import cli


def test_list_positions(example_plate):
    plate_path, _ = example_plate
    runner = CliRunner()
    result = runner.invoke(cli, ["nf", "list-positions", "-i", str(plate_path)])

    assert result.exit_code == 0
    lines = [l for l in result.output.strip().splitlines() if l]
    assert len(lines) == 3
    assert "A/1/0" in lines
    assert "B/1/0" in lines
    assert "B/2/0" in lines


def test_init_flat_field(tmp_path, example_plate):
    plate_path, _ = example_plate
    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "ff_config.yml"
    config_path.write_text(yaml.dump({"channel_names": None}))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "init-flat-field",
            "-i",
            str(plate_path),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "RESOURCES:" in result.output
