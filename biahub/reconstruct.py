from pathlib import Path

from waveorder.cli.compute_transfer_function import (
    compute_transfer_function_cli as compute_transfer_function,
)

from biahub.apply_inverse_transfer_function import apply_inverse_transfer_function
from biahub.cli.parsing import (
    ConfigFilepath,
    InputPositionDirpaths,
    OutputDirpath,
    SbatchFilepath,
    cluster,
    monitor,
)


def reconstruct_cli(
    input_position_dirpaths: InputPositionDirpaths,
    config_filepath: ConfigFilepath,
    output_dirpath: OutputDirpath,
    sbatch_filepath: SbatchFilepath = None,
    cluster: cluster = "slurm",
    monitor: monitor = False,
):
    """Reconstruct a dataset using a configuration file.

    This is a convenience function for a `compute-tf` call followed by a
    `apply-inv-tf` call.

    Calculates the transfer function based on the shape of the first position
    in the list `input-position-dirpaths`, then applies that transfer function
    to all positions in the list `input-position-dirpaths`, so all positions
    must have the same TCZYX shape.

    See https://github.com/mehta-lab/waveorder/tree/main/docs/examples for example configuration files.

    \b
    >>> biahub reconstruct -i ./input.zarr/*/*/* -c ./examples/birefringence.yml -o ./output.zarr
    """  # noqa: D301
    # glob all positions in input_position_dirpaths

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
        sbatch_filepath,
        cluster,
        monitor,
    )
