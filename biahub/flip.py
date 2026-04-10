import click

from iohub import open_ome_zarr

from biahub.cli.parsing import input_position_dirpaths


@click.command("flip")
@input_position_dirpaths()
@click.option("-x", is_flag=True, help="Enable the x flag.")
@click.option("-y", is_flag=True, help="Enable the y flag.")
def flip_cli(input_position_dirpaths: list[str], x: bool, y: bool):
    """Flip the input position files in the specified direction.

    >>> biahub flip -i ./input.zarr/*/*/* --x
    """
    for input_position_filepath in input_position_dirpaths:
        print(f"Flipping {input_position_filepath}")
        with open_ome_zarr(input_position_filepath, mode="a") as dataset:
            array = dataset["0"]
            T, C, _, _, _ = array.shape

            for t in range(T):
                for c in range(C):
                    print(f"\tFlipping {t=}, {c=}")
                    temp = array[t, c, :, :, :]  # read
                    if x:
                        temp = temp[:, :, ::-1]  # flip along x
                    if y:
                        temp = temp[:, ::-1, :]  # flip along y

                    array[t, c, :, :, :] = temp  # write
