import shutil

from pathlib import Path
from typing import Literal, Callable, List
from itertools import product
from collections import defaultdict, OrderedDict

from tqdm import tqdm
import ants
import click
import dask.array as da
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import submitit

from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta
from iohub.ngff.utils import create_empty_plate
from skimage.registration import phase_cross_correlation

from biahub.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from biahub.cli.utils import process_single_position_v2, yaml_to_model
from biahub.register import convert_transform_to_ants
from biahub.settings import ProcessingSettings, StitchSettings


def estimate_shift(
    im0: np.ndarray,
    im1: np.ndarray,
    percent_overlap: float,
    direction: Literal["row", "col"],
    add_offset: bool = False,
):
    """
    Estimate the shift between two images based on a given percentage overlap and direction.

    Parameters
    ----------
    im0 : np.ndarray
        The first image.
    im1 : np.ndarray
        The second image.
    percent_overlap : float
        The percentage of overlap between the two images. Must be between 0 and 1.
    direction : Literal["row", "col"]
        The direction of the shift. Can be either "row" or "col". See estimate_zarr_fov_shifts
    add_offset : bool
        Add offsets to shift-x and shift-y when stitching data from ISS microscope.
        Not clear why we need to do that. By default False

    Returns
    -------
    np.ndarray
        The estimated shift between the two images.

    Raises
    ------
    AssertionError
        If percent_overlap is not between 0 and 1.
        If direction is not "row" or "col".
        If the shape of im0 and im1 are not the same.
    """
    if not (0 <= percent_overlap <= 1):
        raise ValueError("percent_overlap must be between 0 and 1")
    if direction not in ["row", "col"]:
        raise ValueError("direction must be either 'row' or 'col'")
    if im0.shape != im1.shape:
        raise ValueError("Images must have the same shape")

    sizeY, sizeX = im0.shape[-2:]

    # TODO: there may be a one pixel error in the estimated shift
    if direction == "row":
        y_roi = int(sizeY * np.minimum(percent_overlap + 0.05, 1))
        shift, _, _ = phase_cross_correlation(
            im0[..., -y_roi:, :], im1[..., :y_roi, :], upsample_factor=1
        )
        shift[-2] += sizeY
        if add_offset:
            shift[-2] -= y_roi
    elif direction == "col":
        x_roi = int(sizeX * np.minimum(percent_overlap + 0.05, 1))
        shift, _, _ = phase_cross_correlation(
            im0[..., :, -x_roi:], im1[..., :, :x_roi], upsample_factor=1
        )
        shift[-1] += sizeX
        if add_offset:
            shift[-1] -= x_roi

    # TODO: we shouldn't need to flip the order, will cause problems in 3D
    return shift[::-1]


def get_grid_rows_cols(fov_names: list[str]):
    grid_rows = set()
    grid_cols = set()

    for fov_name in fov_names:
        grid_rows.add(fov_name[3:])  # 1-Pos<COL>_<ROW> syntax
        grid_cols.add(fov_name[:3])

    return sorted(grid_rows), sorted(grid_cols)


def get_stitch_output_shape(n_rows, n_cols, sizeY, sizeX, col_translation, row_translation):
    """
    Compute the output shape of the stitched image and the global translation when only col and row translation are given
    """
    global_translation = (
        np.ceil(np.abs(np.minimum(row_translation[0] * (n_rows - 1), 0))).astype(int),
        np.ceil(np.abs(np.minimum(col_translation[1] * (n_cols - 1), 0))).astype(int),
    )
    xy_output_shape = (
        np.ceil(
            sizeY
            + col_translation[1] * (n_cols - 1)
            + row_translation[1] * (n_rows - 1)
            + global_translation[1]
        ).astype(int),
        np.ceil(
            sizeX
            + col_translation[0] * (n_cols - 1)
            + row_translation[0] * (n_rows - 1)
            + global_translation[0]
        ).astype(int),
    )
    return xy_output_shape, global_translation


def get_image_shift(
    col_idx, row_idx, col_translation, row_translation, global_translation
) -> list:
    """
    Compute total translation when only col and row translation are given
    """
    total_translation = [
        col_translation[1] * col_idx + row_translation[1] * row_idx + global_translation[1],
        col_translation[0] * col_idx + row_translation[0] * row_idx + global_translation[0],
    ]

    return total_translation


def shift_image(
    czyx_data: np.ndarray,
    yx_output_shape: tuple[float, float],
    transform: list,
    verbose: bool = False,
) -> np.ndarray:
    if czyx_data.ndim != 4:
        raise ValueError("Input data must be a CZYX array")
    C, Z, Y, X = czyx_data.shape

    if verbose:
        print(f"Transforming image with {transform}")
    # Create array of output_shape and put input data at (0, 0)
    output = np.zeros((C, Z) + yx_output_shape, dtype=np.float32)

    transform = np.asarray(transform)
    if transform.shape == (2,):
        output[..., :Y, :X] = czyx_data.astype(np.float32)
        return ndi.shift(output, (0, 0) + tuple(transform), order=0)
    elif transform.shape == (4, 4):
        ants_transform = convert_transform_to_ants(transform)
        ants_reference = ants.from_numpy(output[0])
        for i, img in enumerate(czyx_data):
            ants_input = ants.from_numpy(img)
            ants_output = ants_transform.apply_to_image(ants_input, ants_reference)
            output[i] = ants_output.numpy().astype('float32')
        return output
    else:
        raise ValueError('Provided transform is not of shape (2,) or (4, 4).')


def _stitch_images(
    data_array: np.ndarray,
    total_translation: dict[str : tuple[float, float]] = None,
    percent_overlap: float = None,
    col_translation: float | tuple[float, float] = None,
    row_translation: float | tuple[float, float] = None,
) -> np.ndarray:
    """
    Stitch an array of 2D images together to create a larger composite image.
    This function is not actively maintained.

    Parameters
    ----------
    data_array : np.ndarray
        The data array to with shape (ROWS, COLS, Y, X) that will be stitched. Call this function multiple
        times to stitch multiple channels, slices, or time points.
    total_translation : dict[str: tuple[float, float]], optional
        Shift to be applied to each fov, given as {fov: (y_shift, x_shift)}. Defaults to None.
    percent_overlap : float, optional
        The percentage of overlap between adjacent images. Must be between 0 and 1. Defaults to None.
    col_translation : float | tuple[float, float], optional
        The translation distance in pixels in the column direction. Can be a single value or a tuple
        of (x_translation, y_translation) when moving across columns. Defaults to None.
    row_translation : float | tuple[float, float], optional
        See col_translation. Defaults to None.

    Returns
    -------
    np.ndarray
        The stitched composite 2D image

    Raises
    ------
    AssertionError
        If percent_overlap is not between 0 and 1.

    """

    n_rows, n_cols, sizeY, sizeX = data_array.shape

    if total_translation is None:
        if percent_overlap is not None:
            assert 0 <= percent_overlap <= 1, "percent_overlap must be between 0 and 1"
            col_translation = sizeX * (1 - percent_overlap)
            row_translation = sizeY * (1 - percent_overlap)
        if not isinstance(col_translation, tuple):
            col_translation = (col_translation, 0)
        if not isinstance(row_translation, tuple):
            row_translation = (0, row_translation)
        xy_output_shape, global_translation = get_stitch_output_shape(
            n_rows, n_cols, sizeY, sizeX, col_translation, row_translation
        )
    else:
        df = pd.DataFrame.from_dict(
            total_translation, orient="index", columns=["shift-y", "shift-x"]
        )
        xy_output_shape = (
            np.ceil(df["shift-y"].max() + sizeY).astype(int),
            np.ceil(df["shift-x"].max() + sizeX).astype(int),
        )
    stitched_array = np.zeros(xy_output_shape, dtype=np.float32)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            image = data_array[row_idx, col_idx]

            if total_translation is None:
                shift = get_image_shift(
                    col_idx, row_idx, col_translation, row_translation, global_translation
                )
            else:
                shift = total_translation[f"{col_idx:03d}{row_idx:03d}"]

            warped_image = shift_image(image, xy_output_shape, shift)
            overlap = np.logical_and(stitched_array, warped_image)
            stitched_array[:, :] += warped_image
            stitched_array[overlap] /= 2  # average blending in the overlapping region

    return stitched_array


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


if __name__ == '__main__':
    stitch_cli()
