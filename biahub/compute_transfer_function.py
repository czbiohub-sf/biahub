from pathlib import Path

import click

from waveorder.cli.compute_transfer_function import (
    compute_transfer_function_cli as compute_transfer_function,
)

from biahub.cli.parsing import config_filepaths, input_position_dirpaths, output_dirpath


@click.command("compute-tf")
@input_position_dirpaths()
@config_filepaths()
@output_dirpath()
def compute_transfer_function_cli(
    input_position_dirpaths: list[Path],
    config_filepaths: list[str],
    output_dirpath: Path,
) -> None:
    """
    Compute a transfer function using a dataset and configuration file.

    Calculates the transfer function based on the shape of the first position
    in the list `input-position-dirpaths`.

    See https://github.com/mehta-lab/waveorder/tree/main/docs/examples for example configuration files.

    >> biahub compute-tf -i ./input.zarr/0/0/0 -c ./examples/birefringence.yml -o ./transfer_function.zarr
    """
    if len(config_filepaths) == 1:
        config_filepath = Path(config_filepaths[0])
    else:
        raise ValueError(
            "Only one configuration file is supported for compute_transfer_function. Please provide a single configuration file."
        )

    compute_transfer_function(input_position_dirpaths[0], config_filepath, output_dirpath)
    click.echo(f"Transfer function computed and saved to {output_dirpath}.")


if __name__ == "__main__":
    compute_transfer_function_cli()
