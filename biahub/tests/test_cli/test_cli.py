from click.testing import CliRunner

from biahub.cli.main import cli


def test_main():
    runner = CliRunner()
    result = runner.invoke(cli)

    assert result.exit_code == 0
    assert "tools for biahub" in result.output
