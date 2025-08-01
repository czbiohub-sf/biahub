import shutil

from pathlib import Path
from typing import Literal

import ants
import click
import dask.array as da
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import submitit

from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from skimage.registration import phase_cross_correlation

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    monitor,
    output_dirpath,
)
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


def preprocess_and_shift(
    image,
    settings: ProcessingSettings,
    output_shape: tuple[int, int],
    transform: list,
    verbose=True,
):
    return shift_image(
        process_dataset(image, settings, verbose), output_shape, transform, verbose
    )


def blend(array: da.Array, method: Literal["average"] = "average"):
    """
    Blend array of pre-shifted images stacked across axis=0

    Parameters
    ----------
    array : da.Array
        Input dask array
    method : str, optional
        Blending method. Defaults to "average".

    Raises
    ------
    NotImplementedError
        Raise error is blending method is not implemented.

    Returns
    -------
    da.Array
        Stitched array
    """
    if method == "average":
        # Sum up all images
        array_sum = array.sum(axis=0)
        # Count how many images contribute to each pixel in the stitched image
        array_bool_sum = (array != 0).sum(axis=0)
        # Replace 0s with 1s to avoid division by zero
        array_bool_sum[array_bool_sum == 0] = 1
        # Divide the sum of images by the number of images contributing to each pixel
        stitched_array = array_sum / array_bool_sum
    else:
        raise NotImplementedError(f"Blending method {method} is not implemented")

    return stitched_array


def stitch_shifted_store(
    input_data_path: str,
    output_data_path: str,
    settings: ProcessingSettings,
    well_names: list = None,
    blending="average",
    verbose=True,
):
    """
    Stitch a zarr store of pre-shifted images.

    Parameters
    ----------
    input_data_path : str
        Path to the input zarr store.
    output_data_path : str
        Path to the output zarr store.
    settings : ProcessingSettings
        Postprocessing settings.
    well_names: list
        Names of wells to stitch
    blending : str, optional
        Blending method. Defaults to "average".
    verbose : bool, optional
        Whether to print verbose output. Defaults to True.
    """
    click.echo(f'Stitching zarr store: {input_data_path}')
    with open_ome_zarr(input_data_path, mode="r") as input_dataset:
        if not well_names:
            # Get all well names
            well_names = [well_name for well_name, _ in input_dataset.wells()]

        for well_name in well_names:
            well = input_dataset[well_name]

            if verbose:
                click.echo(f'Processing well {well_name}')

            # Stack images along axis=0
            dask_array = da.stack(
                [da.from_zarr(pos.data) for _, pos in well.positions()], axis=0
            )

            # Blend images
            stitched_array = blend(dask_array, method=blending)

            # Postprocessing
            stitched_array = process_dataset(stitched_array, settings, verbose)

            # Save stitched array
            click.echo('Computing and writing data')
            with open_ome_zarr(
                Path(output_data_path, well_name, '0'), mode="a"
            ) as output_image:
                da.to_zarr(stitched_array, output_image['0'])
            click.echo(f'Finishing writing data for well {well_name}')


def estimate_zarr_fov_shifts(
    fov0_zarr_path: str,
    fov1_zarr_path: str,
    tcz_index: tuple[int, int, int],
    percent_overlap: float,
    fliplr: bool,
    flipud: bool,
    rot90: int,
    add_offset: bool,
    direction: Literal["row", "col"],
    output_dirname: str = None,
):
    """
    Estimate shift between two zarr FOVs using phase cross-correlation.Apply flips (fliplr, flipud) as preprocessing step.
    Phase cross-correlation is computed only across an ROI defined by (percent_overlap + 0.05) for the given direction.

    Parameters
    ----------
    fov0_zarr_path : str
        Path to the first zarr FOV.
    fov1_zarr_path : str
        Path to the second zarr FOV.
    tcz_index : tuple[int, int, int]
        Index of the time, channel, and z-slice to use for the shift estimation.
    percent_overlap : float
        The percentage of overlap between the two FOVs. Can be approximate.
    fliplr : bool
        Flag indicating whether to flip the FOVs horizontally before estimating shift.
    flipud : bool
        Flag indicating whether to flip the FOVs vertically before estimating shift.
    direction : Literal["row", "col"]
        The direction in which to compute the shift.
        "row" computes vertical overlap with fov1 below fov0.
        "col" computes horizontal overlap with fov1 to the right of fov0.
    output_dirname : str, optional
        The directory to save the output csv file.
        If None, the function returns a DataFrame with the estimated shift.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the estimated shift between the two FOVs.
    """
    fov0_zarr_path = Path(fov0_zarr_path)
    fov1_zarr_path = Path(fov1_zarr_path)
    well_name = Path(*fov0_zarr_path.parts[-3:-1])
    fov0 = fov0_zarr_path.name
    fov1 = fov1_zarr_path.name
    click.echo(f'Estimating shift between FOVs {fov0} and {fov1} in well {well_name}...')

    T, C, Z = tcz_index
    im0 = open_ome_zarr(fov0_zarr_path).data[T, C, Z]
    im1 = open_ome_zarr(fov1_zarr_path).data[T, C, Z]

    if fliplr:
        im0 = np.fliplr(im0)
        im1 = np.fliplr(im1)
    if flipud:
        im0 = np.flipud(im0)
        im1 = np.flipud(im1)
    if rot90 != 0:
        im0 = np.rot90(im0, k=rot90)
        im1 = np.rot90(im1, k=rot90)

    shift = estimate_shift(im0, im1, percent_overlap, direction, add_offset=add_offset)

    df = pd.DataFrame(
        {
            "well": str(well_name),
            "fov0": fov0,
            "fov1": fov1,
            "shift-x": shift[0],
            "shift-y": shift[1],
            "direction": direction,
        },
        index=[0],
    )
    click.echo(f'Estimated shift:\n {df.to_string(index=False)}')

    if output_dirname:
        df.to_csv(
            Path(output_dirname, f"{'_'.join(well_name.parts + (fov0, fov1))}_shift.csv"),
            index=False,
        )
    else:
        return df


def consolidate_zarr_fov_shifts(
    input_dirname: str,
    output_filepath: str,
):
    """
    Consolidate all csv files in input_dirname into a single csv file.

    Parameters
    ----------
    input_dirname : str
        Directory containing "*_shift.csv" files
    output_filepath : str
        Path to output .csv file
    """
    # read all csv files in input_dirname and combine into a single dataframe
    csv_files = Path(input_dirname).rglob("*_shift.csv")
    df = pd.concat(
        [pd.read_csv(csv_file, dtype={'fov0': str, 'fov1': str}) for csv_file in csv_files],
        ignore_index=True,
    )
    df.to_csv(output_filepath, index=False)


def cleanup_shifts(csv_filepath: str, pixel_size_um: float):
    """
    Clean up outlier FOV shifts within a larger grid in case the phase cross-correlation
    between individual FOVs returned spurious results.

    Since FOVs are acquired in snake fashion, FOVs in a given row should share the same vertical (i.e. row) shift.
    Hence, the vertical shift for FOVs in a given row is replaced by the median value of all FOVs in that row.

    FOVs across the grid should have similar horizontal (i.e. column) shifts.
    Values outside of the median +/- MAX_STAGE_ERROR_UM are replaced by the median.

    Parameters
    ----------
    csv_filepath : str
        Path to .csv file containing FOV shifts
    """
    MAX_STAGE_ERROR_UM = 5
    max_stage_error_pix = MAX_STAGE_ERROR_UM / pixel_size_um

    df = pd.read_csv(csv_filepath, dtype={'fov0': str, 'fov1': str})
    df['shift-x-raw'] = df['shift-x']
    df['shift-y-raw'] = df['shift-y']

    # replace row shifts with median value calculated across all columns
    _df = df[df['direction'] == 'row']
    # group by well and last three characters of fov0
    groupby = _df.groupby(['well', _df['fov0'].str[-3:]])
    _df.loc[:, 'shift-x'] = groupby['shift-x-raw'].transform('median')
    _df.loc[:, 'shift-y'] = groupby['shift-y-raw'].transform('median')
    df.loc[df['direction'] == 'row', ['shift-x', 'shift-y']] = _df[['shift-x', 'shift-y']]

    # replace col shifts outside of the median +/- MAX_STAGE_ERROR_UM with the median value
    _df = df[df['direction'] == 'col']
    x_median, y_median = _df['shift-x-raw'].median(), _df['shift-y-raw'].median()
    x_low, x_hi = x_median - max_stage_error_pix, x_median + max_stage_error_pix
    y_low, y_hi = y_median - max_stage_error_pix, y_median + max_stage_error_pix
    x_outliers = (_df['shift-x-raw'] <= x_low) | (_df['shift-x-raw'] >= x_hi)
    y_outliers = (_df['shift-y-raw'] <= y_low) | (_df['shift-y-raw'] >= y_hi)
    outliers = x_outliers | y_outliers
    num_outliers = sum(outliers)

    _df.loc[outliers, ['shift-x', 'shift-y']] = (x_median, y_median)
    df.loc[df['direction'] == 'col', ['shift-x', 'shift-y']] = _df[['shift-x', 'shift-y']]
    if num_outliers > 0:
        click.echo(f'Replaced {num_outliers} column shift outliers')

    df.to_csv(csv_filepath, index=False)


def compute_total_translation(csv_filepath: str) -> pd.DataFrame:
    """
    Compute the total translation for each FOV based on the estimated row and col translation shifts.

    Parameters
    ----------
    csv_filepath : str
        Path to .csv file containing FOV shifts

    Returns
    -------
    pd.DataFrame
        Dataframe with total translation shift per FOV
    """
    df = pd.read_csv(csv_filepath, dtype={'fov0': str, 'fov1': str})

    # create 'row' and 'col' number columns and sort the dataframe by 'fov1'
    df['row'] = df['fov1'].str[-3:].astype(int)
    df['col'] = df['fov1'].str[:3].astype(int)
    df_row = df[(df['direction'] == 'row')]
    df_col = df[(df['direction'] == 'col')]
    row_anchors = sorted(df_row['fov0'][~df_row['fov0'].isin(df_row['fov1'])].unique())
    col_anchors = sorted(df_col['fov0'][~df_col['fov0'].isin(df_col['fov1'])].unique())
    row_col_anchors = sorted(set(row_anchors).intersection(col_anchors))
    df['fov0'] = df[['well', 'fov0']].agg('/'.join, axis=1)
    df['fov1'] = df[['well', 'fov1']].agg('/'.join, axis=1)
    df.set_index('fov1', inplace=True)

    for well in df['well'].unique():
        # add anchors
        df = pd.concat(
            (
                pd.DataFrame(
                    {
                        'well': well,
                        'shift-x': 0,
                        'shift-y': 0,
                        'direction': 'row',
                        'row': [int(a[3:]) for a in row_anchors],
                        'col': [int(a[:3]) for a in row_anchors],
                    },
                    index=['/'.join((well, a)) for a in row_anchors],
                ),
                pd.DataFrame(
                    {
                        'well': well,
                        'shift-x': 0,
                        'shift-y': 0,
                        'direction': 'col',
                        'row': [int(a[3:]) for a in col_anchors],
                        'col': [int(a[:3]) for a in col_anchors],
                    },
                    index=['/'.join((well, a)) for a in col_anchors],
                ),
                df,
            )
        )

        for anchor in row_col_anchors[::-1]:
            df_well = df[df['well'] == well]
            df_well_col = df_well[df_well['direction'] == 'col']
            df_well_row = df_well[df_well['direction'] == 'row']

            _row = int(anchor[3:])
            idx1 = df_well_col[df_well_col['row'] == _row].index
            idx_out = ['/'.join((well, a)) for a in row_anchors if a[3:] == anchor[3:]]
            idx_in = sorted(idx1[~idx1.isin(idx_out)])

            if len(idx_in) > 0:  # will be zero for first row
                shift_x = df_well_row[
                    (df_well_row['row'] <= _row)
                    & (df_well_row['col'] == int(idx_in[0][-6:-3]))
                ]['shift-x'].sum()
                shift_y = df_well_row[
                    (df_well_row['row'] <= _row)
                    & (df_well_row['col'] == int(idx_in[0][-6:-3]))
                ]['shift-y'].sum()

                df_well_row.loc[idx_out, ['shift-x', 'shift-y']] = (shift_x, shift_y)

            df[(df['direction'] == 'row') & (df['well'] == well)] = df_well_row

        for anchor in col_anchors:
            _col = int(anchor[:3])

            shift_x = (
                df_well_col[(df_well_col['col'] <= _col) & (df_well_col['shift-x'] != 0)]
                .groupby('col')['shift-x']
                .median()
                .sum()
            )
            shift_y = (
                df_well_col[(df_well_col['col'] <= _col) & (df_well_col['shift-y'] != 0)]
                .groupby('col')['shift-y']
                .median()
                .sum()
            )

            df_well_col.loc['/'.join((well, anchor)), ['shift-x', 'shift-y']] = (
                shift_x,
                shift_y,
            )

        df[(df['direction'] == 'col') & (df['well'] == well)] = df_well_col

    df.sort_index(inplace=True)  # TODO: remember to sort index after any additions

    total_shift = []
    for well in df['well'].unique():
        # calculate cumulative shifts for each row and column
        _df = df[(df['direction'] == 'col') & (df['well'] == well)]
        col_shifts = _df.groupby('row')[['shift-x', 'shift-y']].cumsum()
        _df = df[(df['direction'] == 'row') & (df['well'] == well)]
        row_shifts = _df.groupby('col')[['shift-x', 'shift-y']].cumsum()
        _total_shift = col_shifts.add(row_shifts, fill_value=0)

        # add global offset to remove negative values
        _total_shift['shift-x'] += -np.minimum(_total_shift['shift-x'].min(), 0)
        _total_shift['shift-y'] += -np.minimum(_total_shift['shift-y'].min(), 0)
        total_shift.append(_total_shift)

    return pd.concat(total_shift)


@click.command("stitch")
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@click.option(
    "--temp-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default='./',
    help="Path to temporary directory, ideally with fast read/write speeds, e.g. /hpc/scratch/group.comp.micro/",
)
@click.option("--debug", is_flag=True, help="Run in debug mode")
@monitor()
def stitch_cli(
    input_position_dirpaths: list[Path],
    output_dirpath: str,
    config_filepath: str,
    temp_path: str,
    debug: bool,
    monitor: bool,
) -> None:
    """
    Stitch positions in wells of a zarr store using a configuration file generated by estimate-stitch.

    >>> biahub stitch -i ./input.zarr/*/*/* -c ./stitch_params.yml -o ./output.zarr --temp-path /hpc/scratch/group.comp.micro/
    """
    slurm_out_path = Path(output_dirpath).parent / "slurm_output"
    dataset = input_position_dirpaths[0].parts[-4][:-5]
    shifted_store_path = Path(temp_path, f"TEMP_{dataset}.zarr").resolve()
    settings = yaml_to_model(config_filepath, StitchSettings)

    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        input_dataset_channels = input_dataset.channel_names
        T, C, Z, Y, X = input_dataset.data.shape
        scale = tuple(input_dataset.scale)
        chunks = input_dataset.data.chunks

    if settings.channels is None:
        settings.channels = input_dataset_channels

    if not all(channel in input_dataset_channels for channel in settings.channels):
        raise ValueError("Invalid channel(s) provided.")

    position_paths = [Path(*p.parts[-3:]).as_posix() for p in input_position_dirpaths]
    wells = list(set([Path(*p.parts[-3:-1]).as_posix() for p in input_position_dirpaths]))
    fov_names = set([p.name for p in input_position_dirpaths])
    grid_rows, grid_cols = get_grid_rows_cols(fov_names)
    n_rows = len(grid_rows)
    n_cols = len(grid_cols)

    # Determine output shape
    if all((settings.column_translation, settings.row_translation)):
        output_shape, global_translation = get_stitch_output_shape(
            n_rows, n_cols, Y, X, settings.column_translation, settings.row_translation
        )
    elif settings.total_translation is not None:
        all_shifts = []
        for _pos, _shift in settings.total_translation.items():
            if _pos in position_paths:
                all_shifts.append(_shift)
        output_shape = np.ceil(np.asarray(all_shifts).max(axis=0) + np.asarray([Y, X]))
        output_shape = tuple(output_shape.astype(int))
    elif settings.affine_transform is not None:
        all_shifts = []
        for _pos, _transform in settings.affine_transform.items():
            if _pos in position_paths:
                # These are inverse transforms so here we take the negative shift
                all_shifts.append([-_transform[1][3], -_transform[2][3]])
        output_shape = np.ceil(
            np.asarray(all_shifts).max(axis=0)
            - np.asarray(all_shifts).min(axis=0)
            + np.asarray([Y, X])
        )
        output_shape = tuple(output_shape.astype(int))
    else:
        raise ValueError('Invalid RegistrationSettings config file')

    # Create output zarr store with final stitch
    stitched_shape = (T, len(settings.channels), Z) + output_shape
    stitched_chunks = chunks[:3] + (4096, 4096)
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[Path(well, '0').parts for well in wells],
        channel_names=settings.channels,
        shape=stitched_shape,
        chunks=stitched_chunks,
        scale=scale,
    )

    shift_job_ids = []
    if shifted_store_path.exists():
        click.echo(f'WARNING: Using existing shifted zarr store at {shifted_store_path}')
    else:
        # Create temp zarr store with shifted FOVs
        click.echo(f'Creating temporary zarr store at {shifted_store_path}')
        slurm_args = {
            "slurm_mem_per_cpu": "8G",
            "slurm_cpus_per_task": 1,
            "slurm_time": 30,
            "slurm_job_name": "temp_store",
            "slurm_partition": "cpu",
        }

        def _populate_wells(well, fovs):
            for fov in fovs:
                pos = well.create_position(fov)
                pos.create_zeros(
                    name='0',
                    shape=stitched_shape,
                    dtype=np.float32,
                    chunks=stitched_chunks,
                )

        # First create wells
        _wells = []
        with open_ome_zarr(
            shifted_store_path,
            layout='hcs',
            mode='w-',
            channel_names=settings.channels,
        ) as ds:
            for well in wells:
                _wells.append(ds.create_well(*Path(well).parts))

        # Then populate wells with FOVs
        # this can take a while, so submitting in batches per well
        executor = submitit.AutoExecutor(folder=slurm_out_path)
        executor.update_parameters(**slurm_args)
        temp_zarr_jobs = []
        with submitit.helpers.clean_env(), executor.batch():
            for well in _wells:
                job = executor.submit(_populate_wells, well, fov_names)
                temp_zarr_jobs.append(job)
        temp_zarr_job_ids = [job.job_id for job in temp_zarr_jobs]

        # Collect transforms
        transforms = []
        for in_path in input_position_dirpaths:
            well = Path(*in_path.parts[-3:-1])
            col, row = (in_path.name[:3], in_path.name[3:])
            fov = str(well / (col + row))

            if settings.affine_transform is not None:
                # COL+ROW order here is important
                transforms.append(settings.affine_transform[fov])
            elif settings.total_translation is not None:
                transforms.append(settings.total_translation[fov])
            else:
                transforms.append(
                    get_image_shift(
                        int(col),
                        int(row),
                        settings.column_translation,
                        settings.row_translation,
                        global_translation,
                    )
                )

        slurm_args = {
            "slurm_mem_per_cpu": "24G",
            "slurm_cpus_per_task": 6,
            "slurm_array_parallelism": 100,  # only 100 jobs can run at the same time
            "slurm_time": 30,
            "slurm_job_name": "shift",
            "slurm_partition": "preempted",
            "slurm_dependency": f"afterok:{temp_zarr_job_ids[0]}:{temp_zarr_job_ids[-1]}",
        }
        # Affine transform needs more resources
        if settings.affine_transform is not None:
            slurm_args.update(
                {
                    "slurm_mem_per_cpu": "48G",
                    "slurm_cpus_per_task": 8,
                    "slurm_time": 60,
                }
            )

        executor = submitit.AutoExecutor(folder=slurm_out_path)
        executor.update_parameters(**slurm_args)
        click.echo('Submitting SLURM jobs')
        shift_jobs = []
        with submitit.helpers.clean_env(), executor.batch():
            for in_path, transform in zip(input_position_dirpaths, transforms):
                job = executor.submit(
                    process_single_position_v2,
                    preprocess_and_shift,
                    input_channel_idx=[
                        input_dataset_channels.index(ch) for ch in settings.channels
                    ],
                    output_channel_idx=list(range(len(settings.channels))),
                    time_indices='all',
                    num_processes=slurm_args['slurm_cpus_per_task'],
                    settings=settings.preprocessing,
                    output_shape=output_shape,
                    verbose=True,
                    transform=transform,
                    input_data_path=in_path,
                    output_path=shifted_store_path,
                )
                shift_jobs.append(job)

        shift_job_ids = [job.job_id for job in shift_jobs]

    slurm_args = {
        "slurm_mem_per_cpu": "8G",
        "slurm_cpus_per_task": 64,
        "slurm_time": "3-00:00:00",  # in [DD-]HH:MM:SS format
        "slurm_partition": "cpu",
        "slurm_job_name": "stitch",
    }
    if shift_job_ids:
        slurm_args["slurm_dependency"] = f"afterok:{shift_job_ids[0]}:{shift_job_ids[-1]}"

    executor = submitit.AutoExecutor(folder=slurm_out_path)
    executor.update_parameters(**slurm_args)

    stitch_jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for well in wells:
            job = executor.submit(
                stitch_shifted_store,
                shifted_store_path,
                output_dirpath,
                settings.postprocessing,
                well_names=[well],
                blending='average',
                verbose=True,
            )
            stitch_jobs.append(job)
    stitch_job_ids = [job.job_id for job in stitch_jobs]

    cleanup_jobs = []
    if not debug:
        slurm_args = {
            "slurm_partition": "cpu",
            "slurm_mem_per_cpu": "12G",
            "slurm_cpus_per_task": 1,
            "slurm_time": "0-01:00:00",  # in [DD-]HH:MM:SS format
            "slurm_job_name": "cleanup",
            "slurm_dependency": f"afterok:{stitch_job_ids[0]}:{stitch_job_ids[-1]}",
        }
        executor = submitit.AutoExecutor(folder=slurm_out_path)
        executor.update_parameters(**slurm_args)
        with submitit.helpers.clean_env():
            cleanup_jobs.append(executor.submit(shutil.rmtree, shifted_store_path))

    job_ids = [
        job.job_id for job in stitch_jobs + shift_jobs + temp_zarr_job_ids + cleanup_jobs
    ]
    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    if monitor:
        monitor_jobs(stitch_jobs, wells)


if __name__ == '__main__':
    stitch_cli()
