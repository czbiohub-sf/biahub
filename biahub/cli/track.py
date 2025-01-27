# %%
import os

from pathlib import Path
from typing import List, Tuple, Union

import click
import numpy as np
import submitit
import toml

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from ultrack import Tracker, MainConfig
from ultrack.imgproc import Cellpose, detect_foreground, inverted_edt, normalize
from ultrack.utils.array import array_apply

from biahub.analysis.AnalysisSettings import ProcessingImportFuncSettings, FunctionSettings, TrackingSettings
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import _check_nan_n_zeros, create_empty_hcs_zarr, yaml_to_model, estimate_resources


def mem_nuc_contour(nuclei_prediction: ArrayLike, membrane_prediction: ArrayLike) -> ArrayLike:
    return (np.asarray(membrane_prediction) + (1 - np.asarray(nuclei_prediction))) / 2


def fill_empty_frames(arr: ArrayLike, empty_frames_idx: List[int]) -> ArrayLike:
    """
    Fills empty frames in an array by propagating values from the nearest non-empty frames.

    Args:
        arr (np.ndarray): Input array (e.g., 3D: T, Y, X or 4D: T, C, Y, X).
        empty_frames_idx (List[int]): Indices of empty frames.

    Returns:
        np.ndarray: Array with empty frames filled.
    """
    if not empty_frames_idx:
        return arr  # No empty frames to fill

    num_frames = arr.shape[0]

    for idx in empty_frames_idx:
        if idx == 0:  # First frame is empty
            # Use the next non-empty frame to fill the first frame
            next_non_empty = next(
                (i for i in range(idx + 1, num_frames) if i not in empty_frames_idx), None
            )
            if next_non_empty is not None:
                arr[idx] = arr[next_non_empty]
        elif idx == num_frames - 1:  # Last frame is empty
            # Use the previous non-empty frame to fill the last frame
            prev_non_empty = next(
                (i for i in range(idx - 1, -1, -1) if i not in empty_frames_idx), None
            )
            if prev_non_empty is not None:
                arr[idx] = arr[prev_non_empty]
        else:  # Middle frames are empty
            # Find the nearest non-empty frame (previous or next)
            prev_non_empty = next(
                (i for i in range(idx - 1, -1, -1) if i not in empty_frames_idx), None
            )
            next_non_empty = next(
                (i for i in range(idx + 1, num_frames) if i not in empty_frames_idx), None
            )

            if prev_non_empty is not None and next_non_empty is not None:
                arr[idx] = arr[prev_non_empty]
            elif prev_non_empty is not None:
                # Fill with the previous non-empty frame
                arr[idx] = arr[prev_non_empty]
            elif next_non_empty is not None:
                # Fill with the next non-empty frame
                arr[idx] = arr[next_non_empty]

    return arr


def data_preprocessing(
    data_dict: dict,
    preprocessing_config: FunctionSettings,
    foreground_config: FunctionSettings,
    contour_config: FunctionSettings,
) -> Tuple[ArrayLike, ArrayLike]:

    # Check for empty frames
    empty_frames = _check_nan_n_zeros(data_dict["lf_image"])
    click.echo(f"Empty frames: {empty_frames}")

    # Drop label free image
    data_dict.pop("lf_image")

    # Fill empty frames for tracking
    for key in data_dict:
        click.echo(f"Filling empty frames for {key}...")
        if empty_frames:
            data_dict[key] = fill_empty_frames(data_dict[key], empty_frames)

    # Preprocess inputs
    for key in preprocessing_config.input_array:
        click.echo(f"Preprocessing {key}...")
        func = get_function(preprocessing_config.func)
        data_dict[key] = array_apply(
            data_dict[key], func=func, **preprocessing_config.additional_params
        )

    # Generate foreground mask
    click.echo("Generating foreground mask...")
    foreground_func = get_function(foreground_config.func)
    fg_input_arrays = [data_dict[key] for key in foreground_config.input_array]
    foreground_mask = array_apply(
        *fg_input_arrays, func=foreground_func, **foreground_config.additional_params
    )

    # Generate contour gradient map
    click.echo("Generating contour gradient map...")
    contour_func = get_function(contour_config.func)
    contour_input_arrays = [data_dict[key] for key in contour_config.input_array]
    contour_gradient_map = array_apply(
        *contour_input_arrays, func=contour_func, **contour_config.additional_params
    )

    return foreground_mask, contour_gradient_map


def ultrack(
    tracking_config,
    foreground_mask: ArrayLike,
    contour_gradient_map: ArrayLike,
    scale: Union[Tuple[float, float], Tuple[float, float, float]],
    databaset_path,
):
    cfg = tracking_config

    cfg.data_config.working_dir = databaset_path

    tracker = Tracker(cfg)
    tracker.track(
        detection=foreground_mask,
        edges=contour_gradient_map,
        scale=scale,
    )

    tracks_df, graph = tracker.to_tracks_layer()
    labels = tracker.to_zarr(
        tracks_df=tracks_df,
    )

    with open(databaset_path / "config.toml", mode="w") as f:
        toml.dump(cfg.dict(by_alias=True), f)

    return (
        labels,
        tracks_df,
        graph,
    )


def track_one_position(
    input_lf_dirpath: Path,
    input_vs_path: Path,
    output_dirpath: Path,
    functions: ProcessingImportFuncSettings,
    z_slice: tuple,
    tracking_config: MainConfig,
) -> None:
    position_key = input_vs_path.parts[-3:]
    fov = "_".join(position_key)
    click.echo(f"Processing position: {fov}")

    click.echo(f"Loading data from: {input_vs_path} and {input_lf_dirpath}...")
    input_im_path = input_lf_dirpath / Path(*position_key)
    im_dataset = open_ome_zarr(input_im_path)
    vs_dataset = open_ome_zarr(input_vs_path)

    T, C, Z, Y, X = vs_dataset.data.shape
    channel_names = vs_dataset.channel_names

    click.echo(f"Virtual Stanining Channel names: {channel_names}")

    yx_scale = vs_dataset.scale[-2:]

    output_metadata = {
        "shape": (T, 1, 1, Y, X),
        "chunks": None,
        "scale": vs_dataset.scale,
        "channel_names": [f"{channel_names[0]}_labels"],
        "dtype": np.uint32,
    }

    create_empty_hcs_zarr(
        store_path=output_dirpath, position_keys=[position_key], **output_metadata
    )
    z_slices = slice(z_slice[0], z_slice[1])

    click.echo(f"Processing z-stack: {z_slices}")

    nuclei_prediction = vs_dataset[0][:, channel_names.index("nuclei_prediction"), z_slices]
    membrane_prediction = vs_dataset[0][
        :, channel_names.index("membrane_prediction"), z_slices
    ]
    lf_image = im_dataset[0][:, 0, z_slices.start, :, :]

    function_names = [func.name for func in functions.processing_functions]
    if 'vs_projection' in functions.processing_functions:
        
        click.echo(f"Applying projection {functions.processing_functions.} to the virtual staining data...")


        nuclei_prediction = projection(nuclei_prediction)
        membrane_prediction = projection(membrane_prediction, axis=1)

    # Prepare data dictionary
    data_dict = {
        "lf_image": lf_image,
        "nuclei_prediction": nuclei_prediction,
        "membrane_prediction": membrane_prediction,
    }

    # Preprocess to get the the foreground and multi-level contours
    click.echo("Preprocessing...")
    foreground_mask, contour_gradient_map = data_preprocessing(
        data_dict=data_dict,
        functions = functions
    )

    # Define path to save the tracking database and graph
    filename = str(output_dirpath).split("/")[-1].split(".")[0]
    databaset_path = output_dirpath.parent / f"{filename}_config_tracking" / f"{fov}"
    os.makedirs(databaset_path, exist_ok=True)

    # Perform tracking
    click.echo("Tracking...")
    trackin_labels, tracks_df, _ = ultrack(
        tracking_config, foreground_mask, contour_gradient_map, yx_scale, databaset_path
    )

    # Save the tracks graph to a CSV file
    csv_path = output_dirpath / Path(*position_key) / f"tracks_{fov}.csv"
    tracks_df.to_csv(csv_path, index=False)

    click.echo(f"Saved tracks to: {output_dirpath / Path(*position_key)}")

    # Save the tracking labels
    with open_ome_zarr(output_dirpath / Path(*position_key), mode="r+") as output_dataset:
        output_dataset[0][:, 0, 0] = np.asarray(trackin_labels)


@click.command()
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
@click.option(
    "-input_lf_dirpaths",
    required=True,
    type=str,
    help="Label Free Image Dirpath",
)
def track(
    input_lf_dirpaths: str,
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = None,
) -> None:

    """
    Track nuclei and membranes in using virtual staining nuclei and membranes data.

    This function applied tracking to the virtual staining data for each position in the input.
    To use this function, install the lib as pip install -e .["track"]

    Example usage:

    biahub track -i virtual_staining.zarr/*/*/* -input_lf_dirpaths lf_stabilize.zarr -o output.zarr -c config_tracking.yml

    """

    output_dirpath = Path(output_dirpath)

    settings = yaml_to_model(config_filepath, TrackingSettings)

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape

    # Estimate resources
    num_cpus, gb_ram_per_cpu = estimate_resources(shape=[T,C,Z,Y,X])

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "tracking",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 60,
        "slurm_partition": "preempted",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Prepare and submit jobs
    slurm_out_path = output_dirpath.parent / "slurm_output"
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo('Submitting SLURM jobs...')
    jobs = []

    with executor.batch():
        for input_vs_position_path in input_position_dirpaths:
            job = executor.submit(
                track_one_position,
                input_lf_dirpath=input_lf_dirpaths,
                input_vs_path=input_vs_position_path,
                output_dirpath=output_dirpath,
                z_slice=settings.z_slices,
                tracking_config=settings.get_tracking_config(),
                functions = settings.functions
            )

            jobs.append(job)


if __name__ == "__main__":
    track()
