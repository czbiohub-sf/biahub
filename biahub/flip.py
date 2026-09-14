from typing import Annotated

import typer

from iohub import open_ome_zarr

from biahub.cli.parsing import InputPositionDirpaths


def flip_cli(
    input_position_dirpaths: InputPositionDirpaths,
    x: Annotated[bool, typer.Option("-x", help="Enable the x flag.")] = False,
    y: Annotated[bool, typer.Option("-y", help="Enable the y flag.")] = False,
):
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
