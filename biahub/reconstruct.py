from pathlib import Path

import click

from biahub.apply_inverse_transfer_function import apply_inverse_transfer_function_cli
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    num_processes,
    output_dirpath,
    sbatch_filepath,
)
from waveorder.cli.compute_transfer_function import compute_transfer_function_cli


@click.command("reconstruct")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@num_processes()
@sbatch_filepath()
@local()
def _reconstruct_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    num_processes: int,
    sbatch_filepath: Path,
    local: bool = False,
):
    """
    Reconstruct a dataset using a configuration file. This is a
    convenience function for a `compute-tf` call followed by a `apply-inv-tf`
    call.

    Calculates the transfer function based on the shape of the first position
    in the list `input-position-dirpaths`, then applies that transfer function
    to all positions in the list `input-position-dirpaths`, so all positions
    must have the same TCZYX shape.

    See https://github.com/mehta-lab/waveorder/tree/main/docs/examples for example configuration files.

    >> biahub reconstruct -i ./input.zarr/*/*/* -c ./examples/birefringence.yml -o ./output.zarr
    """
    # glob all positions in input_position_dirpaths

    # Handle transfer function path
    transfer_function_path = output_dirpath.parent / Path(
        "transfer_function_" + config_filepath.stem + ".zarr"
    )

    # Compute transfer function
    # call cli function directly
    compute_transfer_function_cli(
        input_position_dirpaths[0],
        config_filepath,
        transfer_function_path,
    )

    # Apply inverse transfer function
    apply_inverse_transfer_function_cli(
        input_position_dirpaths,
        transfer_function_path,
        config_filepath,
        output_dirpath,
        num_processes,
        sbatch_filepath,
        local,
    )


if __name__ == "__main__":
    _reconstruct_cli()