import click

from biahub.cli.characterize_psf import characterize_psf
from biahub.cli.estimate_psf import estimate_psf
from biahub.cli.estimate_registration import estimate_registration
from biahub.cli.optimize_registration import optimize_registration
from biahub.cli.stabilize import stabilize
from biahub.concatenate import concatenate_cli
from biahub.deconvolve import deconvolve_cli
from biahub.deskew import deskew_cli
from biahub.estimate_bleaching import estimate_bleaching_cli
from biahub.estimate_deskew import estimate_deskew_cli
from biahub.estimate_stabilization import estimate_stabilization_cli
from biahub.estimate_stitch import estimate_stitch_cli
from biahub.register import register_cli
from biahub.segment import segment_cli
from biahub.stitch import stitch_cli

CONTEXT = {"help_option_names": ["-h", "--help"]}


# `biahub -h` will show subcommands in the order they are added
class NaturalOrderGroup(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(context_settings=CONTEXT, cls=NaturalOrderGroup)
def cli():
    """command-line tools for biahub"""


cli.add_command(estimate_bleaching_cli)
cli.add_command(estimate_deskew_cli)
cli.add_command(deskew_cli)
cli.add_command(estimate_registration)
cli.add_command(optimize_registration)
cli.add_command(register_cli)
cli.add_command(estimate_stitch_cli)
cli.add_command(stitch_cli)
cli.add_command(concatenate_cli)
cli.add_command(estimate_stabilization_cli)
cli.add_command(stabilize)
cli.add_command(estimate_psf)
cli.add_command(deconvolve_cli)
cli.add_command(characterize_psf)
cli.add_command(segment_cli)
