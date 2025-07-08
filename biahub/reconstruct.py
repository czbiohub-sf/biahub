from pathlib import Path

import click

from waveorder.cli.compute_transfer_function import (
    compute_transfer_function_cli as compute_transfer_function,
)

from biahub.apply_inverse_transfer_function import apply_inverse_transfer_function
from biahub.cli.parsing import (
    config_filepaths,
    input_position_dirpaths,
    local,
    monitor,
    num_processes,
    output_dirpath,
    sbatch_filepath,
)


@click.command("reconstruct")
@input_position_dirpaths()
@config_filepaths()
@output_dirpath()
@num_processes()
@sbatch_filepath()
@local()
@monitor()
def reconstruct_cli(
    input_position_dirpaths: list[Path],
    config_filepaths: list[str],
    output_dirpath: Path,
    num_processes: int,
    sbatch_filepath: Path,
    local: bool = False,
    monitor: bool = True,
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

    if len(config_filepaths) == 1:
        config_filepath = Path(config_filepaths[0])
    else:
        raise ValueError(
            "Only one configuration file is supported for reconstruct. Please provide a single configuration file."
        )

    # Handle transfer function path
    transfer_function_path = output_dirpath.parent / Path(
        "transfer_function_" + config_filepath.stem + ".zarr"
    )

    # Compute transfer function
    # call cli function directly
    compute_transfer_function(
        input_position_dirpaths[0],
        config_filepath,
        transfer_function_path,
    )

    # Apply inverse transfer function
    apply_inverse_transfer_function(
        input_position_dirpaths,
        transfer_function_path,
        config_filepath,
        output_dirpath,
        num_processes,
        sbatch_filepath,
        local,
        monitor,
    )


if __name__ == "__main__":
    reconstruct_cli()
