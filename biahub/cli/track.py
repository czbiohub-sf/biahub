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
from ultrack import Tracker
from ultrack.imgproc import detect_foreground, normalize, Cellpose, inverted_edt
from ultrack.utils.array import array_apply

from biahub.analysis.AnalysisSettings import TrackingSettings, FunctionSettings
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import create_empty_hcs_zarr, yaml_to_model, _check_nan_n_zeros

def mem_nuc_contour(nuclei_prediction, membrane_prediction):
    return (np.asarray(membrane_prediction) + (1 - np.asarray(nuclei_prediction))) / 2

function_mapping = {
    "detect_foreground": detect_foreground,
    "normalize": normalize,
    "Cellpose": Cellpose(model_type = 'nuclei'),
    "inverted_edt": inverted_edt,
    "mem_nuc": mem_nuc_contour,
    "max": np.max,
    "mean": np.mean,
    "sum": np.sum,
}

def get_function(func_name):
    if func_name in function_mapping:
        return function_mapping[func_name]
    raise ValueError(f"Function '{func_name}' is not registered in the function mapping.")


def detect_empty_frames(arr):
    empty_frames_idx = []
    for f in range(arr.shape[0]):
        if np.all(arr[f] == 0):
            empty_frames_idx.append(f)
    return empty_frames_idx

def fill_empty_frames(arr, empty_frames_idx):
    prev_idx = 0
    for i in range(len(empty_frames_idx)):
        arr[empty_frames_idx[i]] = arr[empty_frames_idx[i] - 1]

        if empty_frames_idx[i] == prev_idx + 1 and i < len(empty_frames_idx) - 1 and i > 0:
            arr[empty_frames_idx[i]] = arr[empty_frames_idx[i] + 1]
        prev_idx = empty_frames_idx[i]
    return arr


def data_preprocessing(
        data_dict: dict,
        preprocessing_config: FunctionSettings,
        foreground_config: FunctionSettings,
        contour_config: FunctionSettings):

    # Fill empty frames
    empty_frames_idx = detect_empty_frames(data_dict["lf_image"])
    
    # Drop label free image
    data_dict.pop("lf_image")
    
    # Fill empty frames for tracking
    for key in data_dict:
        if empty_frames_idx:
            data_dict[key] = fill_empty_frames(data_dict[key], empty_frames_idx)

    # Preprocess inputs
    for key in preprocessing_config.input_array:
        func = get_function(preprocessing_config.func)
        data_dict[key] = array_apply(
            data_dict[key], func=func, **preprocessing_config.additional_params
        )

    # Generate foreground mask
    foreground_func = get_function(foreground_config.func)
    fg_input_arrays = [data_dict[key] for key in foreground_config.input_array]
    foreground_mask = array_apply(*fg_input_arrays, func=foreground_func, **foreground_config.additional_params)

    # Generate contour gradient map
    contour_func = get_function(contour_config.func)
    contour_input_arrays = [data_dict[key] for key in contour_config.input_array]
    contour_gradient_map = array_apply(*contour_input_arrays, func=contour_func, **contour_config.additional_params)

    return foreground_mask, contour_gradient_map

def ultrack(
    tracking_config,
    foreground_mask: ArrayLike,
    contour_gradient_map: ArrayLike,
    scale: Union[Tuple[float, float],Tuple[float, float, float]],
    databaset_path,
):
    click.echo("Tracking...")
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
    preprocessing_config: dict,
    foreground_config: dict,
    contour_config: dict,
    z_slice: tuple,
    vs_projection: str,
    tracking_config: TrackingSettings,
):
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
        "shape": (T,1, 1, Y, X),
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
    membrane_prediction = vs_dataset[0][:, channel_names.index("membrane_prediction"), z_slices]
    lf_image = im_dataset[0][:, 0, z_slices.start, :, :]

    if vs_projection:
        click.echo(f"Applying projection {vs_projection} to the virtual staining data...") 
 
        projection = get_function(vs_projection)

        nuclei_prediction = projection(nuclei_prediction, axis=1)
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
        data_dict = data_dict,
        preprocessing_config = preprocessing_config,
        foreground_config = foreground_config,
        contour_config = contour_config)

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
):

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
    gb_per_element = 4 / 2**30  # bytes_per_float32 / bytes_per_gb
    num_cpus = np.min([T * C, 16])
    input_memory = num_cpus * Z * Y * X * gb_per_element
    gb_ram_request = np.ceil(np.max([1, input_memory])).astype(int)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "tracking",
        "slurm_mem_per_cpu": f"{gb_ram_request}G",
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

    for input_vs_position_path in input_position_dirpaths:
        track_one_position(
        input_lf_dirpath=input_lf_dirpaths,
        input_vs_path=input_vs_position_path,
        output_dirpath=output_dirpath,
        preprocessing_config = settings.preprocessing_config,
        foreground_config = settings.foreground_config,
        contour_config = settings.contour_config,
        z_slice=settings.z_slices,
        vs_projection=settings.vs_projection,
        tracking_config=settings.get_tracking_config())
    

    # with executor.batch():
    #     for input_vs_position_path in input_position_dirpaths:
    #         job = executor.submit(
    #             tracking_one_position,
    #             input_lf_dirpath=input_lf_dirpaths,
    #             input_vs_path=input_vs_position_path,
    #             output_dirpath=output_dirpath,
    #             z_slice=settings.z_slices,
    #             vs_projection=settings.vs_projection,
    #             tracking_config=settings.tracking_config(),
    #         )
    #         jobs.append(job)


if __name__ == "__main__":
    track()
