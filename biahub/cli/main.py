import click

from biahub.cli.characterize_psf import characterize_psf
from biahub.cli.concatenate import concatenate_cli
from biahub.cli.deconvolve import deconvolve
from biahub.cli.deskew import deskew
from biahub.cli.estimate_bleaching import estimate_bleaching
from biahub.cli.estimate_deskew import estimate_deskew
from biahub.cli.estimate_psf import estimate_psf
from biahub.cli.estimate_registration import estimate_registration
from biahub.cli.estimate_stabilization import estimate_stabilization
from biahub.cli.estimate_stitch import estimate_stitch
from biahub.cli.optimize_registration import optimize_registration
from biahub.cli.register import register
from biahub.cli.segment import segment
from biahub.cli.stabilize import stabilize
from biahub.cli.stitch import stitch

CONTEXT = {"help_option_names": ["-h", "--help"]}


# `biahub -h` will show subcommands in the order they are added
class NaturalOrderGroup(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(context_settings=CONTEXT, cls=NaturalOrderGroup)
def cli():
    """command-line tools for biahub"""


cli.add_command(estimate_bleaching)
cli.add_command(estimate_deskew)
cli.add_command(deskew)
cli.add_command(estimate_registration)
cli.add_command(optimize_registration)
cli.add_command(register)
cli.add_command(estimate_stitch)
cli.add_command(stitch)
cli.add_command(concatenate_cli)
cli.add_command(estimate_stabilization)
cli.add_command(stabilize)
cli.add_command(estimate_psf)
cli.add_command(deconvolve)
cli.add_command(characterize_psf)
cli.add_command(segment)
