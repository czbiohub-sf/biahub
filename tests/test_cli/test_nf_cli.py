from click.testing import CliRunner

from biahub.cli.main import cli


def test_list_positions(example_plate):
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["nf", "list-positions", "-i", str(plate_path)])
    assert result.exit_code == 0, result.output

    lines = [line for line in result.output.strip().split("\n") if line]
    assert len(lines) == 3
    assert "A/1/0" in lines
    assert "B/1/0" in lines
    assert "B/2/0" in lines
