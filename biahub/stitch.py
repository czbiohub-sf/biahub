from collections import defaultdict
from itertools import product
from typing import List

import click
import dask.array as da
import numpy as np
import scipy.ndimage

from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta

from biahub.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from biahub.cli.utils import yaml_to_model
from biahub.settings import ProcessingSettings, StitchSettings


def list_of_nd_slices_from_array_shape(
    array_shape: tuple[int, int, int], chunk_shape: tuple[int, int, int]
) -> list[tuple[slice, slice, slice]]:
    """
    Return a list of slices dividing an array of shape `array_shape`
    into chunks of shape `chunk_shape`.

    Example:
        list_of_nd_slices_from_array_shape((4, 5, 6), (2, 3, 4))
        # [
        #   (slice(0, 2), slice(0, 3), slice(0, 4)),
        #   (slice(0, 2), slice(0, 3), slice(4, 6)),
        #   (slice(0, 2), slice(3, 5), slice(0, 4)),
        #   (slice(0, 2), slice(3, 5), slice(4, 6)),
        #   (slice(2, 4), slice(0, 3), slice(0, 4)),
        #   (slice(2, 4), slice(0, 3), slice(4, 6)),
        #   (slice(2, 4), slice(3, 5), slice(0, 4)),
        #   (slice(2, 4), slice(3, 5), slice(4, 6)),
        # ]
    """
    chunk_slices: list[tuple[slice, slice, slice]] = []
    for idx in product(*[range(0, s, c) for s, c in zip(array_shape, chunk_shape)]):
        chunk_slices.append(
            tuple(slice(i, min(i + c, s)) for i, c, s in zip(idx, chunk_shape, array_shape))
        )
    return chunk_slices


def check_overlap(chunk, fov_shift, fov_extent):
    for dim in range(3):
        if (
            chunk[dim].start >= fov_shift[dim] + fov_extent[dim]
            or chunk[dim].stop <= fov_shift[dim]
        ):
            return False
    return True


def overlap_slices(chunk_corner, chunk_extent, fov_corner, fov_extent):
    """Return (fixed_slice, moving_slice) for overlapping region, or (None, None) if no overlap."""
    fixed, moving = [], []
    for d in range(3):
        start = max(chunk_corner[d], fov_corner[d])
        stop = min(chunk_corner[d] + chunk_extent[d], fov_corner[d] + fov_extent[d])
        if stop <= start:
            return None, None
        fixed_slice = slice(int(start - chunk_corner[d]), int(stop - chunk_corner[d]))
        moving_slice = slice(int(start - fov_corner[d]), int(stop - fov_corner[d]))
        # Ensure both slices are the same size
        fixed_len = fixed_slice.stop - fixed_slice.start
        moving_len = moving_slice.stop - moving_slice.start
        max_len = max(fixed_len, moving_len)
        fixed.append(slice(fixed_slice.start, fixed_slice.start + max_len))
        moving.append(slice(moving_slice.start, moving_slice.start + max_len))
    return tuple(fixed), tuple(moving)


def find_contributing_fovs(chunk, fov_shifts, fov_extent):
    contributing_fovs = []
    for fov_key, fov_shift in fov_shifts.items():
        if check_overlap(chunk, fov_shift, fov_extent):
            contributing_fovs.append(fov_key)
    return contributing_fovs


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


@click.command("stitch")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    type=bool,
    help="Verbose stitching output. Default is False.",
)
def stitch_cli(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepath: str,
    verbose: bool = False,
) -> None:
    """
    Stitch FOVs in each well together into a single FOV.
    Uses shift from configuration file generated with `biahub estimate-stitch`.

    >> biahub stitch -i ./input.zarr/*/*/* -c ./config.yaml -o ./output.zarr
    """

    click.echo("Starting stitching...")
    settings = yaml_to_model(config_filepath, StitchSettings)
    input_plate = open_ome_zarr(input_position_dirpaths[0].parents[2], mode="r")
    all_shifts = settings.total_translation

    input_channels = input_plate.channel_names
    if settings.channels is None:
        settings.channels = input_channels
    channel_idx = np.asarray([input_channels.index(ch) for ch in settings.channels])

    if not all(channel in input_channels for channel in settings.channels):
        raise ValueError("Invalid channel(s) provided.")

    # Create output store
    output_plate = open_ome_zarr(
        output_dirpath, layout='hcs', mode="w", channel_names=settings.channels
    )

    # Group shift metadata by well
    shifts_by_well = defaultdict(dict)
    for key, value in all_shifts.items():
        well_name = "/".join(key.split("/")[:2])
        shifts_by_well[well_name][key] = value

    # Process each well
    for well_name, fov_shifts in shifts_by_well.items():
        if verbose:
            click.echo(
                f"Processing well {list(shifts_by_well.keys()).index(well_name)+1}/{len(shifts_by_well)}: {well_name}"
            )
        first_fov_name = list(shifts_by_well[well_name].keys())[0]
        input_fov_shape = input_plate[first_fov_name].data.shape
        output_chunk_size = input_plate[first_fov_name].data.chunks
        output_shape_zyx = get_output_shape(fov_shifts, input_fov_shape)
        output_scale = input_plate[first_fov_name].scale

        output_shape = (
            input_fov_shape[0],
            len(channel_idx),
        ) + output_shape_zyx

        # Create the output array
        # note that output shape is different for each well, so we are not
        # using iohub.ngff.utils.create_empty_plate here
        output_position = output_plate.create_position(
            first_fov_name.split("/")[0],
            first_fov_name.split("/")[1],
            "0",
        )
        output_array = output_position.create_zeros(
            "0",
            shape=output_shape,
            chunks=output_chunk_size,
            dtype=np.float32,
            transform=[TransformationMeta(type="scale", scale=output_scale)],
        )

        # Split the output array into chunks
        chunk_list = list_of_nd_slices_from_array_shape(
            output_shape_zyx,
            output_chunk_size[2:],
        )

        for chunk in chunk_list:
            if verbose:
                click.echo(
                    f"\tProcessing chunk {chunk_list.index(chunk)+1}/{len(chunk_list)}: {chunk}"
                )

            # For each output chunk, find the input fovs that contribute to it
            contributing_fov_names = find_contributing_fovs(
                chunk, fov_shifts, input_fov_shape[-3:]
            )
            chunk_corner = np.array([chunk[dim].start for dim in range(3)])
            chunk_extent = np.array([chunk[dim].stop - chunk[dim].start for dim in range(3)])

            idx = (slice(None), channel_idx, *chunk)
            array_shape = output_array[idx].shape
            output_chunk = np.zeros(array_shape)

            fixed_slices = []
            moving_slices = []

            # Compute overlap slices
            for fov_name in contributing_fov_names:
                fov_corner = np.array([fov_shifts[fov_name][d] for d in range(3)])
                fov_extent = np.array([input_fov_shape[d + 2] for d in range(3)])
                fixed_slice, moving_slice = overlap_slices(
                    chunk_corner, chunk_extent, fov_corner, fov_extent
                )
                if fixed_slice is None or moving_slice is None:
                    continue
                else:
                    fixed_slices.append(fixed_slice)
                    moving_slices.append(moving_slice)

            # Build distance maps (this will need to change for 3D)
            temp = np.zeros(fov_extent)
            temp[:, 1:-1, 1:-1] = 1
            mask = temp != 0

            centered_distance_map_2d = scipy.ndimage.distance_transform_edt(mask[0])
            centered_distance_map = np.tile(
                centered_distance_map_2d[None, :, :], (output_chunk.shape[-3], 1, 1)
            )

            # Build distance maps
            distance_maps = np.zeros((len(contributing_fov_names),) + output_chunk.shape[-3:])
            for i, (fixed_slice, moving_slice) in enumerate(zip(fixed_slices, moving_slices)):
                if verbose:
                    click.echo(f"\t\tComputing distance map for {contributing_fov_names[i]}")
                fixed_idx = (i, *fixed_slice)  # needed for black formatting
                tmp_moving_idx = (*moving_slice,)  # needed for black formatting
                distance_maps[fixed_idx] = centered_distance_map[tmp_moving_idx]

            # Build weight maps
            if verbose:
                click.echo("\t\tBuilding weight maps")
            k = 1
            w = np.power(distance_maps, k, where=(distance_maps > 0))
            sum_w = np.sum(w, axis=0, keepdims=True)
            weight_maps = w / (sum_w + 1e-8)

            # Apply weights to each fov
            for i, (fov_name, fixed_slice, moving_slice) in enumerate(
                zip(contributing_fov_names, fixed_slices, moving_slices)
            ):
                if verbose:
                    click.echo(f"\t\tApplying weight maps to {fov_name}")
                # Get the fov data
                fov_data = input_plate[fov_name].data

                # Apply weights to the fov data
                moving_idx = (slice(None), channel_idx, *moving_slice)
                fixed_idx = (i, *fixed_slice)
                temp = fov_data[moving_idx] * weight_maps[fixed_idx]

                # Add to the output chunk
                idx = (slice(None), channel_idx, *fixed_slice)
                output_chunk[idx] += temp

            # Write chunk to output array
            if verbose:
                click.echo(f"\t\tWriting chunk to output array: {chunk}")
            idx = (slice(None), channel_idx, *chunk)
            output_array[idx] = output_chunk

    return


if __name__ == '__main__':
    stitch_cli()
