import pytest

from click.testing import CliRunner

from biahub.cli.main import cli


def test_main():
    runner = CliRunner()
    result = runner.invoke(cli)

    assert result.exit_code == 0


@pytest.mark.parametrize(
    "command",
    [
        "estimate-bleaching",
        "estimate-deskew",
        "deskew",
        "estimate-registration",
        "optimize-registration",
        "register",
        "estimate-stitch",
        "stitch",
        "update-scale-metadata",
        "concatenate",
        "estimate-stabilization",
        "stabilize",
        "estimate-psf",
        "deconvolve",
        "characterize-psf",
        "segment",
    ],
)
def test_command_help(command: str):
    runner = CliRunner()
    result = runner.invoke(cli, [command, "--help"])
    assert result.exit_code == 0
