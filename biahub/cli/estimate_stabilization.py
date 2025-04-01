import itertools
import multiprocessing as mp

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import click
import dask.array as da
import numpy as np
import pandas as pd

from iohub.ngff import open_ome_zarr
from pystackreg import StackReg
from waveorder.focus import focus_from_transverse_band

from biahub.analysis.AnalysisSettings import (
    EstimateStabilizationSettings,
    StabilizationSettings,
)
from biahub.cli.estimate_registration import (
    _get_tform_from_beads,
    _interpolate_transforms,
    _validate_transforms,
)
from biahub.cli.parsing import config_filepath, input_position_dirpaths, output_filepath
from biahub.cli.utils import model_to_yaml, yaml_to_model

NA_DET = 1.35
LAMBDA_ILL = 0.500


# TODO: Do we need to compute focus fiding on n_number of channels?
def estimate_position_focus(
    input_data_path: Path,
    input_channel_indices: Tuple[int, ...],
    crop_size_xy: list[int, int],
):
    position, time_idx, channel, focus_idx = [], [], [], []

    with open_ome_zarr(input_data_path) as dataset:
        channel_names = dataset.channel_names
        T, C, Z, Y, X = dataset[0].shape
        T_scale, _, _, _, X_scale = dataset.scale

        for tc_idx in itertools.product(range(T), input_channel_indices):
            data_zyx = dataset.data[tc_idx][
                :,
                Y // 2 - crop_size_xy[1] // 2 : Y // 2 + crop_size_xy[1] // 2,
                X // 2 - crop_size_xy[0] // 2 : X // 2 + crop_size_xy[0] // 2,
            ]

            # if the FOV is empty, set the focal plane to 0
            if np.sum(data_zyx) == 0:
                focal_plane = 0
            else:
                focal_plane = focus_from_transverse_band(
                    data_zyx,
                    NA_det=NA_DET,
                    lambda_ill=LAMBDA_ILL,
                    pixel_size=X_scale,
                )

            position.append(str(Path(*input_data_path.parts[-3:])))
            time_idx.append(tc_idx[0])
            channel.append(channel_names[tc_idx[1]])
            focus_idx.append(focal_plane)

    position_stats_stabilized = {
        "position": position,
        "time_idx": time_idx,
        "channel": channel,
        "focus_idx": focus_idx,
    }
    return position_stats_stabilized


def get_mean_z_positions(dataframe_path: Path, verbose: bool = False) -> None:
    df = pd.read_csv(dataframe_path)

    # Sort the DataFrame based on 'time_idx'
    df = df.sort_values("time_idx")

    # TODO: this is a hack to deal with the fact that the focus finding function returns 0 if it fails
    df["focus_idx"] = df["focus_idx"].replace(0, np.nan).ffill()

    # Get the mean of positions for each time point
    average_focus_idx = df.groupby("time_idx")["focus_idx"].mean().reset_index()

    if verbose:
        import matplotlib.pyplot as plt

        # Get the moving average of the focus_idx
        plt.plot(average_focus_idx["focus_idx"], linestyle="--", label="mean of all positions")
        plt.xlabel('Time index')
        plt.ylabel('Focus index')
        plt.legend()
        plt.savefig(dataframe_path.parent / "z_drift.png")

    return average_focus_idx["focus_idx"].values


def estimate_z_stabilization(
    input_data_paths: Path,
    output_folder_path: Path,
    z_drift_channel_idx: int = 0,
    num_processes: int = 1,
    crop_size_xy: list[int, int] = [600, 600],
    verbose: bool = False,
) -> np.ndarray:
    output_folder_path.mkdir(parents=True, exist_ok=True)

    fun = partial(
        estimate_position_focus,
        input_channel_indices=(z_drift_channel_idx,),
        crop_size_xy=crop_size_xy,
    )
    # TODO: do we need to natsort the input_data_paths?
    with mp.Pool(processes=num_processes) as pool:
        position_stats_stabilized = pool.map(fun, input_data_paths)

    df = pd.concat([pd.DataFrame.from_dict(stats) for stats in position_stats_stabilized])
    df.to_csv(output_folder_path / 'positions_focus.csv', index=False)

    # Calculate and save the output file
    z_drift_offsets = get_mean_z_positions(
        output_folder_path / 'positions_focus.csv',
        verbose=verbose,
    )

    # Calculate the z focus shift matrices
    z_focus_shift = [np.eye(4)]
    # Find the z focus shift matrices for each time point based on the z_drift_offsets relative to the first timepoint.
    z_val = z_drift_offsets[0]
    for z_val_next in z_drift_offsets[1:]:
        shift = np.eye(4)
        shift[0, 3] = z_val_next - z_val
        z_focus_shift.append(shift)
    z_focus_shift = np.array(z_focus_shift)

    if verbose:
        click.echo(f"Saving z focus shift matrices to {output_folder_path}")
        z_focus_shift_filepath = output_folder_path / "z_focus_shift.npy"
        np.save(z_focus_shift_filepath, z_focus_shift)

    return z_focus_shift


def estimate_xy_stabilization(
    input_data_paths: Path,
    output_folder_path: Path,
    c_idx: int = 0,
    crop_size_xy: list[int, int] = (400, 400),
    verbose: bool = False,
) -> np.ndarray:
    input_data_path = input_data_paths[0]
    input_position = open_ome_zarr(input_data_path)
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    focus_params = {
        "NA_det": NA_DET,
        "lambda_ill": LAMBDA_ILL,
        "pixel_size": input_position.scale[-1],
    }

    # Get metadata
    T, C, Z, Y, X = input_position.data.shape
    x_idx = slice(X // 2 - crop_size_xy[0] // 2, X // 2 + crop_size_xy[0] // 2)
    y_idx = slice(Y // 2 - crop_size_xy[1] // 2, Y // 2 + crop_size_xy[1] // 2)

    if (output_folder_path / "positions_focus.csv").exists():
        df = pd.read_csv(output_folder_path / "positions_focus.csv")
        pos_idx = str(Path(*input_data_path.parts[-3:]))
        focus_idx = df[df["position"] == pos_idx]["focus_idx"]
        # forward fill 0 values, when replace remaining NaN with the mean
        focus_idx = focus_idx.replace(0, np.nan).ffill()
        focus_idx = focus_idx.fillna(focus_idx.mean())
        z_idx = focus_idx.astype(int).to_list()
    else:
        z_idx = [
            focus_from_transverse_band(
                input_position[0][
                    0,
                    c_idx,
                    :,
                    y_idx,
                    x_idx,
                ],
                **focus_params,
            )
        ] * T
        if verbose:
            click.echo(f"Estimated in-focus slice: {z_idx}")

    # Load timelapse and ensure negative values are not present
    tyx_data = np.stack(
        [
            input_position[0][_t_idx, c_idx, _z_idx, y_idx, x_idx]
            for _t_idx, _z_idx in zip(range(T), z_idx)
        ]
    )
    tyx_data = np.clip(tyx_data, a_min=0, a_max=None)

    # register each frame to the previous (already registered) one
    # this is what the original StackReg ImageJ plugin uses
    sr = StackReg(StackReg.TRANSLATION)

    T_stackreg = sr.register_stack(tyx_data, reference="previous", axis=0)

    # Swap values in the array since stackreg is xy and we need yx
    for subarray in T_stackreg:
        subarray[0, 2], subarray[1, 2] = subarray[1, 2], subarray[0, 2]

    T_zyx_shift = np.zeros((T_stackreg.shape[0], 4, 4))
    T_zyx_shift[:, 1:4, 1:4] = T_stackreg
    T_zyx_shift[:, 0, 0] = 1

    # Save the translation matrices
    if verbose:
        click.echo(f"Saving translation matrices to {output_folder_path}")
        yx_shake_translation_tx_filepath = (
            output_folder_path / "yx_shake_translation_tx_ants.npy"
        )
        np.save(yx_shake_translation_tx_filepath, T_zyx_shift)

    input_position.close()

    return T_zyx_shift


def estimate_xyz_stabilization_with_beads(
    channel_tzyx: da.Array,
    num_processes: int = 1,
    window_size: int = 10,
    tolerance: float = 100,
    angle_threshold: int = 0,
    referece: str = "first",
    verbose: bool = False,
):
    """
    Perform beads-based temporal registration of 4D data using affine transformations.

    This function calculates timepoint-specific affine transformations to align a source channel
    to a target channel in 4D (T, Z, Y, X) data. It validates, smooths, and interpolates transformations
    across timepoints for consistent registration.

    Parameters:
    - channel_tzyx (da.Array): 4D array (T, Z, Y, X) of the source channel (Dask array).
    - approx_tform (list): Initial approximate affine transform (4x4 matrix) for guiding registration.
    - num_processes (int): Number of parallel processes for transformation computation.
    - window_size (int): Size of the moving window for smoothing transformations.
    - tolerance (float): Maximum allowed difference between consecutive transformations for validation.
    - angle_threshold (int): Threshold for filtering outliers in detected bead matches (in degrees).
    - verbose (bool): If True, prints detailed logs of the registration process.

    Returns:
    - transforms (list): List of affine transformation matrices (4x4), one for each timepoint.
                         Invalid or missing transformations are interpolated.

    Notes:
    - Each timepoint is processed in parallel using a multiprocessing pool.
    - Transformations are smoothed with a moving average window and validated against a reference.
    - Missing transformations are interpolated linearly across timepoints.
    - Use verbose=True for detailed logging during registration.
    """

    (T, Z, Y, X) = channel_tzyx.shape

    approx_tform = np.eye(4)

    if referece == "first":
        target_channel_tzyx = np.broadcast_to(channel_tzyx[0], (T, Z, Y, X)).copy()
    elif referece == "previous":
        target_channel_tzyx = np.roll(channel_tzyx, shift=-1, axis=0)
    else:
        raise ValueError("Invalid reference. Please use 'first' or 'previous as reference")
    target_channel_tzyx[0] = channel_tzyx[0]

    transforms = np.zeros((T, 4, 4))
    transforms[0] = approx_tform
    # Compute transformations in parallel

    with Pool(num_processes) as pool:
        transforms = pool.map(
            partial(
                _get_tform_from_beads,
                approx_tform,
                channel_tzyx,
                target_channel_tzyx,
                angle_threshold,
                verbose,
                source_block_size=[32, 16, 16],
                source_threshold_abs=0.8,
                source_nms_distance=16,
                source_min_distance=0,
                target_block_size=[32, 16, 16],
                target_threshold_abs=0.8,
                target_nms_distance=16,
                target_min_distance=0,
            ),
            t_idx=range(1, T, 1),
        )

    # Validate and filter transforms
    transforms = _validate_transforms(transforms, window_size, tolerance, Z, Y, X, verbose)
    # Interpolate missing transforms
    transforms = _interpolate_transforms(transforms)

    return transforms


@click.command()
@input_position_dirpaths()
@output_filepath()
@config_filepath()
@click.option(
    "--num-processes",
    "-j",
    default=1,
    help="Number of parallel processes. Default is 1.",
    required=False,
    type=int,
)
def estimate_stabilization(
    input_position_dirpaths,
    output_filepath,
    config_filepath,
    num_processes,
):
    """
    Estimate the Z and/or XY timelapse stabilization matrices.

    This function estimates xy and z drifts and returns the affine matrices per timepoint taking t=0 as reference saved as a yaml file.
    The level of verbosity can be controlled with the verbose flag.
    The size of the crop in xy can be specified with the crop-size-xy option.

    Example usage:
    biahub stabilization -i ./timelapse.zarr/0/0/0 -o ./stabilization.yml -y -z -b -v --crop-size-xy 300 300

    Note: the verbose output will be saved at the same level as the output zarr.
    """

    # Load the settings
    config_filepath = Path(config_filepath)

    settings = yaml_to_model(config_filepath, EstimateStabilizationSettings)
    verbose = settings.verbose
    crop_size_xy = settings.crop_size_xy
    estimate_stabilization_channel = settings.estimate_stabilization_channel
    stabilization_type = settings.stabilization_type
    beads = settings.beads
    if "z" in stabilization_type:
        stabilize_z = True
    else:
        stabilize_z = False
    if "xy" in stabilization_type:
        stabilize_xy = True
    else:
        stabilize_xy = False
    if "xyz" in stabilization_type:
        stabilize_z = True
        stabilize_xy = True
    if not (stabilize_xy or stabilize_z):
        raise ValueError("At least one of 'stabilize_xy' or 'stabilize_z' must be selected")

    if output_filepath.suffix not in [".yml", ".yaml"]:
        raise ValueError("Output file must be a yaml file")

    output_dirpath = output_filepath.parent
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # Channel names to process
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        channel_names = dataset.channel_names
        voxel_size = dataset.scale
        channel_index = channel_names.index(estimate_stabilization_channel)

    # Estimate z drift
    if stabilize_z and not beads:
        click.echo("Estimating z stabilization parameters")
        T_z_drift_mats = estimate_z_stabilization(
            input_data_paths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            z_drift_channel_idx=channel_index,
            num_processes=num_processes,
            crop_size_xy=crop_size_xy,
            verbose=verbose,
        )

    # Estimate yx drift
    if stabilize_xy and not beads:
        click.echo("Estimating xy stabilization parameters")
        T_translation_mats = estimate_xy_stabilization(
            input_data_paths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            c_idx=channel_index,
            crop_size_xy=crop_size_xy,
            verbose=verbose,
        )

    if stabilize_z and stabilize_xy:
        if beads:
            click.echo("Estimating xyz stabilization parameters with beads")
            with open_ome_zarr(input_position_dirpaths[0], mode="r") as beads_position:
                source_channels = beads_position.channel_names
                source_channel_index = source_channels.index(estimate_stabilization_channel)
                channel_tzyx = beads_position.data.dask_array()[:, source_channel_index]
            combined_mats = estimate_xyz_stabilization_with_beads(
                channel_tzyx=channel_tzyx,
                num_processes=num_processes,
                window_size=5,
                tolerance=0.2,
                angle_threshold=30,
                referece=settings.reference,
                verbose=verbose,
            )
            # replace nan with 0
            combined_mats = np.nan_to_num(combined_mats)

        else:
            if T_translation_mats.shape[0] != T_z_drift_mats.shape[0]:
                raise ValueError(
                    "The number of translation matrices and z drift matrices must be the same"
                )
            combined_mats = np.array(
                [a @ b for a, b in zip(T_translation_mats, T_z_drift_mats)]
            )

    # NOTE: we've checked that one of the two conditions below is true
    elif stabilize_z:
        combined_mats = T_z_drift_mats
    elif stabilize_xy:
        combined_mats = T_translation_mats

    # Save the combined matrices
    model = StabilizationSettings(
        stabilization_type=stabilization_type,
        stabilization_estimation_channel=estimate_stabilization_channel,
        stabilization_channels=settings.stabilization_channels,
        affine_transform_zyx_list=combined_mats.tolist(),
        time_indices="all",
        output_voxel_size=voxel_size,
    )
    model_to_yaml(model, output_filepath)


if __name__ == "__main__":
    estimate_stabilization()
