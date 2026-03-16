from pathlib import Path

import click

from waveorder.cli.compute_transfer_function import (
    compute_transfer_function_cli as compute_transfer_function,
)

from biahub.apply_inverse_transfer_function import apply_inverse_transfer_function
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    monitor,
    num_processes,
    output_dirpath,
    sbatch_filepath,
)


@click.command("reconstruct")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@num_processes()
@sbatch_filepath()
@local()
@monitor()
def reconstruct_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    num_processes: int,
    sbatch_filepath: Path | None,
    local: bool = False,
    monitor: bool = True,
) -> None:
    """
    Reconstruct a dataset using a configuration file.

    This is a convenience function that combines `compute-tf` and `apply-inv-tf`
    calls. It calculates the transfer function based on the shape of the first
    position in the list, then applies that transfer function to all positions.
    All positions must have the same TCZYX shape.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        List of paths to the input position directories (OME-Zarr format).
    config_filepath : Path
        Path to the YAML configuration file specifying reconstruction settings.
    output_dirpath : Path
        Path to the output directory where the reconstructed dataset will be saved.
    num_processes : int
        Number of processes to use for parallel computation.
    sbatch_filepath : Path | None, optional
        Path to the SLURM batch file for cluster submission, by default None.
    local : bool, optional
        If True, run the jobs locally instead of submitting to a SLURM cluster, by default False.
    monitor : bool, optional
        If True, monitor the progress of the submitted jobs, by default True.

    Returns
    -------
    None
        The reconstructed data is written to the `output_dirpath`.

    Notes
    -----
    See https://github.com/mehta-lab/waveorder/tree/main/docs/examples for example configuration files.

    Examples
    --------
    >> biahub reconstruct -i ./input.zarr/*/*/* -c ./examples/birefringence.yml -o ./output.zarr
    """
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
        num_processes,
        sbatch_filepath,
        local,
        monitor,
    )


if __name__ == "__main__":
    reconstruct_cli()
