from collections import defaultdict
from itertools import product
from typing import Callable

import click
import dask.array as da
import numpy as np

from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta
from tqdm import tqdm

from biahub.cli.utils import yaml_to_model
from biahub.settings import ProcessingSettings, StitchSettings


def process_dataset(
    data_array: np.ndarray | da.Array,
    settings: ProcessingSettings,
    verbose: bool = True,
) -> np.ndarray:
    flip = np.flip
    rot = np.rot90
    if isinstance(data_array, da.Array):
        flip = da.flip
        rot = da.rot90

    if settings:
        if settings.flipud:
            if verbose:
                click.echo("Flipping data array up-down")
            data_array = flip(data_array, axis=-2)

        if settings.fliplr:
            if verbose:
                click.echo("Flipping data array left-right")
            data_array = flip(data_array, axis=-1)

        if settings.rot90 != 0:
            if verbose:
                click.echo(f"Rotating data array {settings.rot90} times counterclockwise")
            data_array = rot(data_array, settings.rot90, axes=(-2, -1))

    return data_array


def get_output_shape(shifts: dict, tile_shape: tuple) -> tuple:
    """Get the output shape of the stitched image from the raw shifts"""

    z_shifts = [shift[0] for shift in shifts.values()]
    y_shifts = [shift[1] for shift in shifts.values()]
    x_shifts = [shift[2] for shift in shifts.values()]

    max_z = int(np.max(np.asarray(z_shifts)))
    max_y = int(np.max(np.asarray(y_shifts)))
    max_x = int(np.max(np.asarray(x_shifts)))

    return max_z + tile_shape[-3], max_y + tile_shape[-2], max_x + tile_shape[-1]


def divide_tile(
    *in_arrays: np.ndarray,
    func: Callable,
    out_array: np.ndarray,
    tile: tuple,
    overlap: tuple = (0, 0),
):

    final_shape = out_array.shape[-2:]

    tiling_start = list(
        product(
            *[
                range(o, size + 2 * o, t + o)  # t + o step, because of blending map
                for size, t, o in zip(final_shape, tile, overlap)
            ]
        )
    )
    for start_indices in tqdm(tiling_start):
        slicing = (...,) + tuple(
            slice(start - o, start + t + o)
            for start, t, o in zip(start_indices, tile, overlap)
        )
        out_array[slicing] = func(*[a[slicing] for a in in_arrays])

    return out_array


@click.command("stitch")
@click.option(
    "-i",
    "--input_dirpath",
    required=True,
    type=click.Path(exists=True, dir_okay=True),
    help="Path to zarr store containing individual FOVs to be stitched",
)
@click.option(
    "-o",
    "--output_dirpath",
    required=True,
    type=click.Path(exists=False, dir_okay=True),
    help="Path to zarr store where stitched FOVs will be saved",
)
@click.option(
    "-c",
    "--config_filepath",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to yaml file containing stitching parameters",
)
def stitch_cli(
    input_dirpath: str,
    output_dirpath: str,
    config_filepath: str,
    divide_tiling_shape: tuple = (5000, 5000),
) -> None:

    settings = yaml_to_model(config_filepath, StitchSettings)
    input_fov_store = open_ome_zarr(input_dirpath, mode="r")
    shifts = settings.total_translation

    input_store_channels = input_fov_store.channel_names
    if settings.channels is None:
        settings.channels = input_store_channels
    channel_idx = np.asarray([input_store_channels.index(ch) for ch in settings.channels])

    if not all(channel in input_store_channels for channel in settings.channels):
        raise ValueError("Invalid channel(s) provided.")

    output_store = open_ome_zarr(
        output_dirpath, layout='hcs', mode="w-", channel_names=settings.channels
    )

    grouped_shifts = defaultdict(dict)
    for key, value in shifts.items():
        group = key.split("/")[1]
        grouped_shifts[group][key] = value

    for g in grouped_shifts:
        pos_shifts = grouped_shifts[g]
        temp_pos = list(grouped_shifts[g].keys())[0]
        tile_shape = input_fov_store[temp_pos].data.shape
        final_shape_zyx = get_output_shape(pos_shifts, tile_shape)

        final_shape = (
            tile_shape[0],
            len(channel_idx),
        ) + final_shape_zyx

        output_image = np.zeros(final_shape, dtype=np.float32)  # check dtype
        divisor = np.zeros(final_shape, dtype=np.uint8)

        well_name = temp_pos[:3]
        output_chunk_size = input_fov_store[temp_pos].data.chunks
        output_scale = input_fov_store[temp_pos].scale

        for tile_name, shift in pos_shifts.items():
            tile = input_fov_store[tile_name].data

            tile = process_dataset(tile[:, channel_idx, :, :, :], settings.preprocessing)
            shift_array = np.asarray(shift).astype(np.uint16)  # round shift to ints

            output_image[
                :,
                :,
                shift_array[0] : shift_array[0] + tile_shape[-3],
                shift_array[1] : shift_array[1] + tile_shape[-2],
                shift_array[2] : shift_array[2] + tile_shape[-1],
            ] += tile

            divisor[
                :,
                :,
                shift_array[0] : shift_array[0] + tile_shape[-3],
                shift_array[1] : shift_array[1] + tile_shape[-2],
                shift_array[2] : shift_array[2] + tile_shape[-1],
            ] += 1

        stitched = np.zeros_like(output_image, dtype=np.float16)

        def _divide(a, b):
            return np.nan_to_num(a / b)

        divide_tile(
            output_image,
            divisor,
            func=_divide,
            out_array=stitched,
            tile=divide_tiling_shape,
        )

        stitched = process_dataset(stitched, settings.postprocessing)

        stitched_pos = output_store.create_position(well_name[0], well_name[2], "0")
        stitched_pos.create_image(
            "0",
            data=stitched,
            chunks=output_chunk_size,
            transform=[TransformationMeta(type="scale", scale=output_scale)],
        )

    return


if __name__ == '__main__':
    stitch_cli()
