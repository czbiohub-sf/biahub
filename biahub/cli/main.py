import click

from biahub.apply_inverse_transfer_function import apply_inverse_transfer_function_cli
from biahub.characterize_psf import characterize_psf_cli
from biahub.compute_transfer_function import compute_transfer_function_cli
from biahub.concatenate import concatenate_cli
from biahub.deconvolve import deconvolve_cli
from biahub.deskew import deskew_cli
from biahub.estimate_bleaching import estimate_bleaching_cli
from biahub.estimate_deskew import estimate_deskew_cli
from biahub.estimate_psf import estimate_psf_cli
from biahub.estimate_registration import estimate_registration_cli
from biahub.estimate_stabilization import estimate_stabilization_cli
from biahub.estimate_stitch import estimate_stitch_cli
from biahub.optimize_registration import optimize_registration_cli
from biahub.process_data import process_with_config_cli
from biahub.reconstruct import reconstruct_cli
from biahub.register import register_cli
from biahub.segment import segment_cli
from biahub.stabilize import stabilize_cli
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
cli.add_command(estimate_registration_cli)
cli.add_command(optimize_registration_cli)
cli.add_command(register_cli)
cli.add_command(estimate_stitch_cli)
cli.add_command(stitch_cli)
cli.add_command(concatenate_cli)
cli.add_command(estimate_stabilization_cli)
cli.add_command(stabilize_cli)
cli.add_command(compute_transfer_function_cli)
cli.add_command(apply_inverse_transfer_function_cli)
cli.add_command(reconstruct_cli)
cli.add_command(estimate_psf_cli)
cli.add_command(deconvolve_cli)
cli.add_command(characterize_psf_cli)
cli.add_command(segment_cli)
cli.add_command(process_with_config_cli)
