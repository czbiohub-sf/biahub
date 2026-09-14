import subprocess
import sys

import pytest

from typer.testing import CliRunner

from biahub.cli.main import app


def test_main():
    runner = CliRunner()
    result = runner.invoke(app)

    assert result.exit_code == 2


def test_main_help_keeps_heavy_command_modules_lazy():
    code = """
import sys
from typer.testing import CliRunner
from biahub.cli.main import app

result = CliRunner().invoke(app, ["--help"])
assert result.exit_code == 0, result.output
assert not ({"ants", "monai", "napari", "torch", "ultrack"} & sys.modules.keys())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "estimate-bleaching",
        "estimate-deskew",
        "deskew",
        "estimate-registration",
        "flat-field",
        "flip",
        "optimize-registration",
        "pyramid",
        "register",
        "estimate-stitch",
        "stitch",
        "concatenate",
        "estimate-stabilization",
        "stabilize",
        "estimate-crop",
        "compute-tf",
        "apply-inv-tf",
        "reconstruct",
        "estimate-psf",
        "deconvolve",
        "characterize-psf",
        "segment",
        "virtual-stain",
        "track",
        "process-with-config",
        "nf",
    ],
)
def test_command_help(command: str):
    runner = CliRunner()
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
