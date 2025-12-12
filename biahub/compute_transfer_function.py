from pathlib import Path

import click

from waveorder.cli.compute_transfer_function import (
    compute_transfer_function_cli as compute_transfer_function,
)

from biahub.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath


@click.command("compute-tf")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def compute_transfer_function_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
) -> None:
    """
    Compute a transfer function using a dataset and configuration file.

    This command-line tool calculates the transfer function based on the shape
    of the first position in the input list. The transfer function is used for
    reconstruction tasks such as deconvolution or phase retrieval.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        List of paths to the input position directories (OME-Zarr format).
        Only the first position is used to determine the transfer function shape.
    config_filepath : Path
        Path to the YAML configuration file specifying transfer function settings.
    output_dirpath : Path
        Path to the output directory where the computed transfer function will be saved.

    Returns
    -------
    None
        The transfer function is saved to the specified output directory.

    Notes
    -----
    See https://github.com/mehta-lab/waveorder/tree/main/docs/examples for example configuration files.

    Examples
    --------
    >> biahub compute-tf -i ./input.zarr/0/0/0 -c ./examples/birefringence.yml -o ./transfer_function.zarr
    """

    compute_transfer_function(input_position_dirpaths[0], config_filepath, output_dirpath)
    click.echo(f"Transfer function computed and saved to {output_dirpath}.")


if __name__ == "__main__":
    compute_transfer_function_cli()
