# %%
import os

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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
    Compute the membrane-nucleus contour by averaging the membrane signal and the inverse nucleus signal.

    This function generates a contour map that highlights the boundary between the nucleus and
    membrane regions in an imaging dataset.

    Parameters:
    - nuclei_prediction (ArrayLike): A NumPy array representing the predicted nucleus.
    - membrane_prediction (ArrayLike): A NumPy array representing the predicted membrane.

    Returns:
    - ArrayLike: A NumPy array representing the computed membrane-nucleus contour.

    Notes:
    - Input arrays should have the same shape to avoid broadcasting issues.
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
    Resolve a function by its name from a predefined function mapping.

    This function retrieves a callable function from the `FUNCTION_MAP` dictionary based on
    the provided function name. If the function name does not exist in the mapping, an
    error is raised.

    Parameters:
    - function_name (str): The name of the function to retrieve.

    Returns:
    - Callable: The resolved function from `FUNCTION_MAP`.

    Raises:
    - ValueError: If the provided function name is not found in `FUNCTION_MAP`.

    Notes:
    - `FUNCTION_MAP` is a dictionary containing allowed functions that can be dynamically
      resolved and applied to data.
    """

    if function_name not in FUNCTION_MAP:
        raise ValueError(
            f"Invalid function '{function_name}'. Allowed functions: {list(FUNCTION_MAP.keys())}"
        )

    return FUNCTION_MAP[function_name]


def fill_empty_frames(arr: ArrayLike, empty_frames_idx: List[int]) -> ArrayLike:
    """
    Fill empty frames in a time-series imaging dataset by propagating values from the nearest non-empty frames.

    This function identifies empty frames in a 3D (T, Y, X) or 4D (T, C, Y, X) array and fills them
    using the nearest available data. If an empty frame is found at the beginning or end, it is filled
    with the closest available non-empty frame. For empty frames in the middle, the function prioritizes
    propagation from previous frames but falls back to future frames if necessary.

    Parameters:
    - arr (ArrayLike): Input 3D (T, Y, X) or 4D (T, C, Y, X) imaging data array.
    - empty_frames_idx (List[int]): Indices of frames that are completely empty.

    Returns:
    - ArrayLike: The input array with empty frames filled.

    Notes:
    - If no empty frames are detected, the function returns the array unchanged.
    - If the first or last frame is empty, it is filled with the nearest available frame.
    - For middle frames, the function attempts to fill using the closest previous non-empty frame
      but falls back to the next available frame if necessary.
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

    """
    Preprocess imaging data for tracking by applying preprocessing functions and generating
    required inputs for tracking.

    This function applies a series of preprocessing steps to the input imaging data, including
    handling empty frames, applying user-defined preprocessing functions, and generating
    the necessary masks for object tracking.

    Parameters:
    - data_dict (dict): Dictionary containing imaging data with channel names as keys and
                         corresponding NumPy arrays as values.
    - preprocessing_functions (Dict[str, ProcessingFunctions]): Dictionary mapping preprocessing
                                                                steps to their corresponding functions.
    - tracking_functions (Dict[str, ProcessingFunctions]): Dictionary containing functions for
                                                            generating foreground and contour masks.

    Returns:
    - Tuple[ArrayLike, ArrayLike]:
        - The foreground mask used for tracking.
        - The contour gradient map used for tracking.

    Notes:
    - If the "Phase3D" key is present in `data_dict`, the function identifies and fills empty frames
      before processing.
    - The preprocessing functions are dynamically resolved and applied to the corresponding
      input arrays.
    """

    if "Phase3D" in data_dict:
        click.echo("Checking for empty frames in the label-free image...")
        empty_frames_idx = _check_nan_n_zeros(data_dict["Phase3D"])

        # drop lf_image from data_dift
        data_dict.pop("Phase3D")
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
    tracking_config: MainConfig,
    foreground_mask: ArrayLike,
    contour_gradient_map: ArrayLike,
    scale: Union[Tuple[float, float], Tuple[float, float, float]],
    databaset_path,
):
    """
    Perform object tracking using the ultrack library.

    This function tracks objects based on a provided foreground mask and contour gradient map
    using the specified tracking configuration. The results include labeled tracking data,
    a DataFrame containing track information, and a graph representation of the tracks.

    Parameters:
    - tracking_config (MainConfig): Configuration settings for the tracking process.
    - foreground_mask (ArrayLike): Binary or probability mask indicating detected objects.
    - contour_gradient_map (ArrayLike): Gradient-based contour map used for tracking refinement.
    - scale (Union[Tuple[float, float], Tuple[float, float, float]]): Scale factors for spatial
      resolution, given as (Y, X) or (Z, Y, X) depending on the dataset.
    - databaset_path (Path): Directory where tracking results and configurations will be saved.

    Returns:
    - np.ndarray: Labeled tracking results in an OME-Zarr format.
    - pd.DataFrame: DataFrame containing tracking information, including object IDs, positions,
      and frame associations.
    - networkx.Graph: Graph representation of object tracks, useful for lineage and connectivity
      analysis.

    Notes:
    - The function modifies `tracking_config` to set the working directory for results storage.
    - The function also saves the tracking configuration in a TOML file for reproducibility.
    """

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
    input_channels: Dict[str, Any],
    vs_projection_function: ProcessingFunctions,
    preprocessing_functions: Dict[str, ProcessingFunctions],
    tracking_functions: Dict[str, ProcessingFunctions],
    z_slice: tuple,
    tracking_config: MainConfig,
) -> None:
    """
    Process a single imaging position for cell tracking using virtual staining and optional label-free imaging.

    This function loads imaging data from the specified input directories, applies preprocessing steps, and
    performs object tracking using the provided configuration. It obtain the foreground and contour-based tracking
     to generate labeled tracking outputs.

    Parameters:
    - input_lf_dirpath (Path): Path to the directory containing label-free images (optional, required
                               if blank frames exist in the data).
    - input_vs_path (Path): Path to the virtual staining dataset in OME-Zarr format.
    - output_dirpath (Path): Path to save tracking results, including labels and tracks.
    - input_channels (Dict[str, Any]): Dictionary specifying channel names for label-free, virtual staining,
                                       and tracking inputs.
    - vs_projection_function (ProcessingFunctions): Function used to project virtual staining data across
                                                    the z-stack (e.g., max projection).
    - preprocessing_functions (Dict[str, ProcessingFunctions]): Dictionary of preprocessing functions applied
                                                                to the input images before tracking.
    - tracking_functions (Dict[str, ProcessingFunctions]): Dictionary of functions used for foreground mask
                                                           and contour generation.
    - z_slice (tuple): Tuple specifying the range of z-slices to process.
    - tracking_config (MainConfig): Configuration settings for the tracking algorithm.

    Returns:
    - None: The function also saves the tracking results, including labeled images and track data,
            to the specified output directory.

    Notes:
    - If blank frames exist in the data, the label-free image directory must be provided.
    - The function verifies the presence of required input channels and raises an error if any are missing.
    - Tracking is performed using the `ultrack` library, and the results are saved in an OME-Zarr format.
    - Tracks graphs are exported as CSV files.
    """

    position_key = input_vs_path.parts[-3:]
    fov = "_".join(position_key)
    z_slices = slice(z_slice[0], z_slice[1])

    click.echo(f"Processing z-stack: {z_slices}")
    click.echo(f"Processing position: {fov}")

    data_dict = {}
    # Load label free data
    if input_channels['label_free'] is not None:
        if input_lf_dirpath is not None:
            click.echo(f"Loading data from: {input_lf_dirpath}...")
            input_im_path = input_lf_dirpath / Path(*position_key)
            im_dataset = open_ome_zarr(input_im_path)
            channel_names = im_dataset.channel_names
            click.echo(f"Label Free Channel names: {channel_names}")
            for channel in input_channels["label_free"]:
                data_dict[channel] = im_dataset[0][
                    :, channel_names.index(channel), z_slices.start, :, :
                ]
        else:
            raise ValueError(
                "Label Free Image Dirpath is required if there are blanck frames in the data."
            )
    # Load virtual staining data, check if the required channels are present
    if input_channels['virtual_stain'] is None:
        raise ValueError("Virtual Staining input channels is required.")

    # Check if the tracking channel is present in the virtual staining channels
    if input_channels['tracking'] is None:
        raise ValueError("Tracking input channels is required.")
    elif len(input_channels["tracking"]) != 1:
        raise ValueError("Only one channel is allowed and required for tracking.")
    elif input_channels["tracking"][0] not in input_channels["virtual_stain"]:
        raise ValueError("Tracking channel not found in Virtual Stain input channels.")

    click.echo(f"Loading data from: {input_vs_path}...")
    vs_dataset = open_ome_zarr(input_vs_path)
    T, C, Z, Y, X = vs_dataset.data.shape
    channel_names = vs_dataset.channel_names
    click.echo(f"Virtual Stanining Channel names: {channel_names}")
    yx_scale = vs_dataset.scale[-2:]

    # Define output metadata
    output_channels = []
    for channel in input_channels["tracking"]:
        output_channels.append(f"{channel}_labels")

    output_metadata = {
        "shape": (T, 1, 1, Y, X),
        "chunks": None,
        "scale": vs_dataset.scale,
        "channel_names": output_channels,
        "dtype": np.uint32,
    }

    create_empty_hcs_zarr(
        store_path=output_dirpath, position_keys=[position_key], **output_metadata
    )

    # Load virtual staining data
    for channel in input_channels["virtual_stain"]:
        data_dict[channel] = vs_dataset[0][:, channel_names.index(channel), z_slices]

    # Apply virtual staining projection function
    if vs_projection_function is not None:
        click.echo(
            f"Applying {vs_projection_function.function} projection to the virtual staining data..."
        )

        projection = eval(vs_projection_function.function)  # Convert string to function
        kwargs = vs_projection_function.kwargs if vs_projection_function.kwargs else {}

        for input_array in vs_projection_function.input_arrays:
            data_dict[input_array] = projection(data_dict[input_array], **kwargs)

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
    "-lf_dirpaths",
    "-l",
    required=False,
    default=None,
    type=str,
    help="Label Free Image Dirpath, if there are blanck frames in the data.",
)
def track(
    lf_dirpaths: str,
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
                input_lf_dirpath=lf_dirpaths,
                input_vs_path=input_vs_position_path,
                output_dirpath=output_dirpath,
                z_slice=settings.z_slices,
                input_channels=settings.input_channels,
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
