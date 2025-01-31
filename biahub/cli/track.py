# %%
import os

from pathlib import Path
from typing import Dict, List, Tuple, Union

import click
import numpy as np
import submitit
import toml
import ultrack

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from ultrack import MainConfig, Tracker
from ultrack.utils.array import array_apply

from biahub.analysis.AnalysisSettings import ProcessingFunctions, TrackingSettings
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    _check_nan_n_zeros,
    create_empty_hcs_zarr,
    estimate_resources,
    update_model,
    yaml_to_model,
)


# Custom function
def mem_nuc_contour(nuclei_prediction: ArrayLike, membrane_prediction: ArrayLike) -> ArrayLike:
    """
    Computes the membrane-nucleus contour by averaging the membrane signal
    and the inverse nucleus signal.
    """
    return (np.asarray(membrane_prediction) + (1 - np.asarray(nuclei_prediction))) / 2


# List of modules to scan for functions
VALID_MODULES = {"np": np, "ultrack.imgproc": ultrack.imgproc}

# Dynamically populate FUNCTION_MAP with functions from VALID_MODULES
FUNCTION_MAP = {
    f"{module_name}.{func}": getattr(module, func)
    for module_name, module in VALID_MODULES.items()
    for func in dir(module)
    if callable(getattr(module, func))
    and not func.startswith("__")  # Only include functions, not attributes
}

# Add custom functions manually
FUNCTION_MAP["biahub.cli.track.mem_nuc_contour"] = mem_nuc_contour


def resolve_function(function_name: str):
    """
    Resolves a function from FUNCTION_MAP. Raises an error if the function is not found.
    """
    if function_name not in FUNCTION_MAP:
        raise ValueError(
            f"Invalid function '{function_name}'. Allowed functions: {list(FUNCTION_MAP.keys())}"
        )

    return FUNCTION_MAP[function_name]


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
    preprocessing_functions: Dict[str, ProcessingFunctions],
    tracking_functions: Dict[str, ProcessingFunctions],
) -> Tuple[ArrayLike, ArrayLike]:
    if "lf_image" in data_dict:
        click.echo("Checking for empty frames in the label-free image...")
        empty_frames_idx = _check_nan_n_zeros(data_dict["lf_image"])

        # drop lf_image from data_dift
        data_dict.pop("lf_image")
        for key, value in data_dict.items():
            data_dict[key] = fill_empty_frames(value, empty_frames_idx)

    # Apply preprocessing functions
    for key, func_details in preprocessing_functions.items():
        click.echo(f"Preprocessing {key} using {func_details.function}...")
        input_arrays_name = func_details.input_arrays[0]
        function = resolve_function(func_details.function)  # Uses function mapping
        input_array = data_dict[input_arrays_name]

        kwargs = func_details.kwargs if func_details.kwargs else {}
        data_dict[input_arrays_name] = array_apply(input_array, func=function, **kwargs)
    # Generate foreground mask
    click.echo("Generating foreground mask...")
    foreground_func_details = tracking_functions["foreground"]
    foreground_function = resolve_function(
        foreground_func_details.function
    )  # Uses function mapping

    fg_input_arrays = [data_dict[name] for name in foreground_func_details.input_arrays]
    fg_kwargs = foreground_func_details.kwargs if foreground_func_details.kwargs else {}

    foreground_mask = array_apply(*fg_input_arrays, func=foreground_function, **fg_kwargs)

    # Generate contour gradient map
    click.echo("Generating contour gradient map...")
    contour_func_details = tracking_functions["contourn"]
    contour_function = resolve_function(contour_func_details.function)  # Uses function mapping

    contour_input_arrays = [data_dict[name] for name in contour_func_details.input_arrays]
    contour_kwargs = contour_func_details.kwargs if contour_func_details.kwargs else {}

    contour_gradient_map = array_apply(
        *contour_input_arrays, func=contour_function, **contour_kwargs
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

    print(cfg)

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
    vs_projection_function: ProcessingFunctions,
    preprocessing_functions: Dict[str, ProcessingFunctions],
    tracking_functions: Dict[str, ProcessingFunctions],
    z_slice: tuple,
    tracking_config: MainConfig,
) -> None:
    position_key = input_vs_path.parts[-3:]
    fov = "_".join(position_key)
    z_slices = slice(z_slice[0], z_slice[1])

    click.echo(f"Processing z-stack: {z_slices}")

    click.echo(f"Processing position: {fov}")
    data_dict = {}
    if input_lf_dirpath is not None:

        click.echo(f"Loading data from: {input_lf_dirpath}...")
        input_im_path = input_lf_dirpath / Path(*position_key)
        im_dataset = open_ome_zarr(input_im_path)
        data_dict["lf_image"] = im_dataset[0][:, 0, z_slices.start, :, :]

    click.echo(f"Loading data from: {input_vs_path}...")
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

    nuclei_prediction = vs_dataset[0][:, channel_names.index("nuclei_prediction"), z_slices]
    membrane_prediction = vs_dataset[0][
        :, channel_names.index("membrane_prediction"), z_slices
    ]

    if vs_projection_function is not None:
        click.echo(
            f"Applying {vs_projection_function.function} projection to the virtual staining data..."
        )

        projection = eval(vs_projection_function.function)  # Convert string to function
        kwargs = vs_projection_function.kwargs if vs_projection_function.kwargs else {}

        nuclei_prediction = projection(nuclei_prediction, **kwargs)
        membrane_prediction = projection(membrane_prediction, **kwargs)

    # Prepare data dictionary
    data_dict["nuclei_prediction"] = nuclei_prediction
    data_dict["membrane_prediction"] = membrane_prediction

    # Preprocess to get the the foreground and multi-level contours
    click.echo("Preprocessing...")
    foreground_mask, contour_gradient_map = data_preprocessing(
        data_dict=data_dict,
        preprocessing_functions=preprocessing_functions,
        tracking_functions=tracking_functions,
    )

    # Define path to save the tracking database and graph
    filename = str(output_dirpath).split("/")[-1].split(".")[0]
    databaset_path = output_dirpath.parent / f"{filename}_config_tracking" / f"{fov}"
    os.makedirs(databaset_path, exist_ok=True)

    # Perform tracking
    click.echo("Tracking...")
    tracking_labels, tracks_df, _ = ultrack(
        tracking_config, foreground_mask, contour_gradient_map, yx_scale, databaset_path
    )

    # Save the tracks graph to a CSV file
    csv_path = output_dirpath / Path(*position_key) / f"tracks_{fov}.csv"
    tracks_df.to_csv(csv_path, index=False)

    click.echo(f"Saved tracks to: {output_dirpath / Path(*position_key)}")

    # Save the tracking labels
    with open_ome_zarr(output_dirpath / Path(*position_key), mode="r+") as output_dataset:
        output_dataset[0][:, 0, 0] = np.asarray(tracking_labels)


@click.command()
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
@click.option(
    "-input_lf_dirpaths",
    "-ilf",
    required=False,
    default=None,
    type=str,
    help="Label Free Image Dirpath, if there are blanck frames in the data.",
)
def track(
    input_lf_dirpaths: str,
    output_dirpath: str,
    config_filepath: str,
    input_position_dirpaths: str,
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
    tracking_data = settings.tracking_config

    # Create default instance
    default_config = MainConfig()

    tracking_cfg = update_model(default_config, tracking_data)

    # Get the shape of the data
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape

    # Estimate resources
    num_cpus, gb_ram_per_cpu = estimate_resources(shape=[T, C, Z, Y, X], ram_multiplier=16)

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
    click.echo(f"Preparing jobs: {slurm_args}")
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
                tracking_config=tracking_cfg,
                vs_projection_function=settings.vs_projection_function,
                preprocessing_functions=settings.preprocessing_functions,
                tracking_functions=settings.tracking_functions,
            )

            jobs.append(job)

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    log_path = Path(output_dirpath.parent / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))


if __name__ == "__main__":
    track()
