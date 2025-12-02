import itertools
import os
import shutil

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

import click
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import submitit

from iohub.ngff import open_ome_zarr
from numpy.typing import ArrayLike
from pystackreg import StackReg
from scipy.fftpack import next_fast_len
from tqdm import tqdm
from waveorder.focus import focus_from_transverse_band

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.estimate_registration import (
    estimate_transform_from_beads,
    evaluate_transforms,
    save_transforms,
)
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    EstimateStabilizationSettings,
    FocusFindingSettings,
    PhaseCrossCorrSettings,
    StabilizationSettings,
    StackRegSettings,
)

from biahub.registration.phase_cross_correlation import estimate_xyz_stabilization_pcc, estimate_z_stabilization, estimate_xy_stabilization
from biahub.registration.beads import estimate_xyz_stabilization_with_beads

def estimate_stabilization(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = False,
    vs_path: str = None,
) -> None:
    """
    Estimate the stabilization matrices for a list of positions.

    Parameters
    ----------
    input_position_dirpaths : list[str]
        Paths to the input position directories.
    output_filepath : str
        Path to the output file.
    config_filepath : str
        Path to the configuration file.
    sbatch_filepath : str
        Path to the sbatch file.
    local : bool
        If True, run locally.
    vs_path : str
        Path to the virtual stain data.
    Returns
    -------
    None

    Notes
    -----
    The verbose output will be saved at the same level as the output zarr.
    """

    # Load the settings
    config_filepath = Path(config_filepath)

    settings = yaml_to_model(config_filepath, EstimateStabilizationSettings)
    click.echo(f"Settings: {settings}")

    verbose = settings.verbose
    stabilization_estimation_channel = settings.stabilization_estimation_channel
    stabilization_type = settings.stabilization_type
    stabilization_method = settings.stabilization_method

    output_dirpath = Path(output_dirpath)
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # Channel names to process
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        channel_names = dataset.channel_names
        voxel_size = dataset.scale
        channel_index = channel_names.index(stabilization_estimation_channel)
        T, C, Z, Y, X = dataset.data.shape

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Load the evaluation settings
    eval_transform_settings = settings.eval_transform_settings

    if "xyz" == stabilization_type:
        if stabilization_method == "focus-finding":
            click.echo(
                "Estimating xyz stabilization parameters with focus finding and stack registration"
            )

            z_transforms_dict = estimate_z_stabilization(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                focus_finding_settings=settings.focus_finding_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
                vs_path=vs_path,
            )

            xy_transforms_dict = estimate_xy_stabilization(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                stack_reg_settings=settings.stack_reg_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )

            try:

                for fov, xy_transforms in tqdm(
                    xy_transforms_dict.items(), desc="Processing FOVs"
                ):

                    z_transforms = np.asarray(z_transforms_dict[fov])
                    xy_transforms = np.asarray(xy_transforms)

                    if xy_transforms.shape[0] != z_transforms.shape[0]:
                        raise ValueError(
                            "The number of translation matrices and z drift matrices must be the same"
                        )

                    xyz_transforms = np.asarray(
                        [a @ b for a, b in zip(xy_transforms, z_transforms)]
                    ).tolist()

                    if eval_transform_settings:
                        xyz_transforms = evaluate_transforms(
                            transforms=xyz_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=verbose,
                        )
                        z_transforms = evaluate_transforms(
                            transforms=z_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=verbose,
                        )
                        xy_transforms = evaluate_transforms(
                            transforms=xy_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=verbose,
                        )

                    save_transforms(
                        model=model,
                        transforms=xyz_transforms,
                        output_filepath_settings=output_dirpath
                        / "xyz_stabilization_settings"
                        / f"{fov}.yml",
                        output_filepath_plot=output_dirpath
                        / "translation_plots"
                        / f"{fov}.png",
                        verbose=verbose,
                    )
                    save_transforms(
                        model=model,
                        transforms=z_transforms,
                        output_filepath_settings=output_dirpath
                        / "z_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                    )
                    save_transforms(
                        model=model,
                        transforms=xy_transforms,
                        output_filepath_settings=output_dirpath
                        / "xy_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                    )

            except Exception as e:
                click.echo(
                    f"Error estimating {stabilization_type} stabilization parameters: {e}"
                )

        elif stabilization_method == "beads":

            click.echo("Estimating xyz stabilization parameters with beads")
            with open_ome_zarr(input_position_dirpaths[0], mode="r") as beads_position:
                source_channels = beads_position.channel_names
                source_channel_index = source_channels.index(stabilization_estimation_channel)
                channel_tzyx = beads_position.data.dask_array()[:, source_channel_index]

            xyz_transforms = estimate_xyz_stabilization_with_beads(
                channel_tzyx=channel_tzyx,
                beads_match_settings=settings.beads_match_settings,
                affine_transform_settings=settings.affine_transform_settings,
                verbose=verbose,
                output_folder_path=output_dirpath,
                cluster=cluster,
                sbatch_filepath=sbatch_filepath,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )

            if eval_transform_settings:
                xyz_transforms = evaluate_transforms(
                    transforms=xyz_transforms,
                    shape_zyx=(Z, Y, X),
                    validation_window_size=eval_transform_settings.validation_window_size,
                    validation_tolerance=eval_transform_settings.validation_tolerance,
                    interpolation_window_size=eval_transform_settings.interpolation_window_size,
                    interpolation_type=eval_transform_settings.interpolation_type,
                    verbose=verbose,
                )

            save_transforms(
                model=model,
                transforms=xyz_transforms,
                output_filepath_settings=output_dirpath / "xyz_stabilization_settings.yml",
                verbose=verbose,
                output_filepath_plot=output_dirpath / "translation_plots" / "beads.png",
            )

        elif stabilization_method == "phase-cross-corr":
            click.echo("Estimating xyz stabilization parameters with phase cross correlation")

            xyz_transforms_dict = estimate_xyz_stabilization_pcc(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                phase_cross_corr_settings=settings.phase_cross_corr_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )

            try:
                for fov, xyz_transforms in tqdm(
                    xyz_transforms_dict.items(), desc="Processing FOVs"
                ):
                    if eval_transform_settings:
                        xyz_transforms = evaluate_transforms(
                            transforms=xyz_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=verbose,
                        )

                    save_transforms(
                        model=model,
                        transforms=xyz_transforms,
                        output_filepath_settings=output_dirpath
                        / "xyz_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                        output_filepath_plot=output_dirpath
                        / "translation_plots"
                        / f"{fov}.png",
                    )
            except Exception as e:
                click.echo(
                    f"Error estimating {stabilization_type} stabilization parameters: {e}"
                )

    # Estimate z drift
    if "z" == stabilization_type and stabilization_method == "focus-finding":
        click.echo("Estimating z stabilization parameters with focus finding")

        z_transforms_dict = estimate_z_stabilization(
            input_position_dirpaths=input_position_dirpaths,
            output_folder_path=output_dirpath,
            channel_index=channel_index,
            focus_finding_settings=settings.focus_finding_settings,
            sbatch_filepath=sbatch_filepath,
            vs_path=vs_path,
            cluster=cluster,
            verbose=verbose,
        )

        model = StabilizationSettings(
            stabilization_type=settings.stabilization_type,
            stabilization_method=settings.stabilization_method,
            stabilization_estimation_channel=settings.stabilization_estimation_channel,
            stabilization_channels=settings.stabilization_channels,
            affine_transform_zyx_list=[],
            time_indices="all",
            output_voxel_size=voxel_size,
        )

        try:
            for fov, z_transforms in tqdm(z_transforms_dict.items(), desc="Processing FOVs"):
                if eval_transform_settings:
                    z_transforms = evaluate_transforms(
                        transforms=z_transforms,
                        shape_zyx=(Z, Y, X),
                        validation_window_size=eval_transform_settings.validation_window_size,
                        validation_tolerance=eval_transform_settings.validation_tolerance,
                        interpolation_window_size=eval_transform_settings.interpolation_window_size,
                        interpolation_type=eval_transform_settings.interpolation_type,
                        verbose=verbose,
                    )

                save_transforms(
                    model=model,
                    transforms=z_transforms,
                    output_filepath_settings=output_dirpath
                    / "z_stabilization_settings"
                    / f"{fov}.yml",
                    verbose=verbose,
                    output_filepath_plot=output_dirpath / "translation_plots" / f"{fov}.png",
                )
        except Exception as e:
            click.echo(f"Error estimating {stabilization_type} stabilization parameters: {e}")

    # Estimate yx drift
    if "xy" == stabilization_type:
        if stabilization_method == "focus-finding":
            click.echo(
                "Estimating xy stabilization parameters with focus finding and stack registration"
            )

            xy_transforms_dict = estimate_xy_stabilization(
                input_position_dirpaths=input_position_dirpaths,
                output_folder_path=output_dirpath,
                channel_index=channel_index,
                stack_reg_settings=settings.stack_reg_settings,
                sbatch_filepath=sbatch_filepath,
                cluster=cluster,
                verbose=verbose,
                vs_path=vs_path,
            )

            model = StabilizationSettings(
                stabilization_type=settings.stabilization_type,
                stabilization_method=settings.stabilization_method,
                stabilization_estimation_channel=settings.stabilization_estimation_channel,
                stabilization_channels=settings.stabilization_channels,
                affine_transform_zyx_list=[],
                time_indices="all",
                output_voxel_size=voxel_size,
            )
            try:
                for fov, xy_transforms in tqdm(
                    xy_transforms_dict.items(), desc="Processing FOVs"
                ):
                    if eval_transform_settings:
                        xy_transforms = evaluate_transforms(
                            transforms=xy_transforms,
                            shape_zyx=(Z, Y, X),
                            validation_window_size=eval_transform_settings.validation_window_size,
                            validation_tolerance=eval_transform_settings.validation_tolerance,
                            interpolation_window_size=eval_transform_settings.interpolation_window_size,
                            interpolation_type=eval_transform_settings.interpolation_type,
                            verbose=verbose,
                        )

                    save_transforms(
                        model=model,
                        transforms=xy_transforms,
                        output_filepath_settings=output_dirpath
                        / "xy_stabilization_settings"
                        / f"{fov}.yml",
                        verbose=verbose,
                        output_filepath_plot=output_dirpath
                        / "translation_plots"
                        / f"{fov}.png",
                    )
            except Exception as e:
                click.echo(
                    f"Error estimating {stabilization_type} stabilization parameters: {e}"
                )


@click.command("estimate-stabilization")
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
@click.option(
    "--vs_path", type=str, required=False, help="Path to the virtual stain data.", default=None
)
def estimate_stabilization_cli(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepath: Path,
    sbatch_filepath: str = None,
    local: bool = False,
    vs_path: str = None,
):
    """
    Estimate translation matrices for XYZ stabilization of a timelapse dataset.

    Stabilization parameters may be computed for the XY, Z, or XYZ dimensions using
    focus finding, beads, or phase cross correlation methods.

    Example usage:
    biahub estimate-stabilization -i ./timelapse.zarr/0/0/0 -o ./stabilization.yml  -c ./config.yml -s ./sbatch.sh --local --verbose

    """
    estimate_stabilization(
        input_position_dirpaths=input_position_dirpaths,
        output_dirpath=output_dirpath,
        config_filepath=config_filepath,
        sbatch_filepath=sbatch_filepath,
        local=local,
        vs_path=vs_path,
    )


if __name__ == "__main__":
    estimate_stabilization_cli()
