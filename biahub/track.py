# %%
import ast
import os

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import click
import numpy as np
import pandas as pd
import submitit
import toml
import ultrack

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from ultrack import MainConfig, Tracker
from ultrack.utils import labels_to_edges
from ultrack.utils.array import array_apply

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    create_empty_hcs_zarr,
    estimate_resources,
    update_model,
    yaml_to_model,
)
from biahub.settings import ProcessingFunctions, TrackingSettings


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

    if not empty_frames_idx or not isinstance(empty_frames_idx, list):
        return arr


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


def get_empty_frames_idx_from_csv(
    blank_frame_df: pd.DataFrame, fov: str
) -> Union[List[int], None]:
    """
    Extract empty frames indices from a DataFrame containing blank frame information.
    This function retrieves the indices of empty frames for a specific field of view (FOV)
    from a DataFrame containing blank frame information.
    Parameters:
    - blank_frame_df (pd.DataFrame): DataFrame containing blank frame information.
    - fov (str): Field of view (FOV) identifier to filter the DataFrame.
    Returns:
    - List[int]: List of indices representing empty frames for the specified FOV.
    - None: If no empty frames are found for the specified FOV.
    """

    empty_frames_idx = blank_frame_df[blank_frame_df['FOV'] == fov]['t']
    if not empty_frames_idx.empty:
        t_value = empty_frames_idx.iloc[0]
        if isinstance(t_value, str) and t_value.startswith('['):
            t_value = ast.literal_eval(t_value)
        if isinstance(t_value, list):
            return [int(i) for i in t_value]
        elif t_value == 0:
            return []
    return None

def central_z_slice(z_shape: int) -> slice:
    """
    Get the central slice of the z-axis.
    Resolve the z-slice based on the provided z_slices and z_shape.

    If z_slices is None or equal to (0, 0), return a centered z-range.
    Otherwise, return a slice constructed from the provided tuple.

    Example:
        z_slices=(10, 15) → slice(10, 15)
        z_slices=(0, 0)   → slice(centered range)

    
    Returns:
        slice: Slice object representing a centered z-range.
    """
    n_slices = max(3, z_shape // 2)
    if n_slices % 2 == 0:
        n_slices += 1
    z_center = z_shape // 2
    half_window = n_slices // 2
    return slice(z_center - half_window, z_center + half_window + 1)
def resolve_z_slice(z_slices: Tuple[int, int], z_shape: int) -> slice:
    """
    Resolve the z-slice based on the provided z_slices and z_shape.
    """
    if z_slices is None or z_slices == (0, 0):
        return central_z_slice(z_shape)
    else:   
        return slice(*z_slices)

def data_preprocessing(
    data_dict: dict,
    preprocessing_functions: Dict[str, ProcessingFunctions],
    empty_frames_idx: List[int] = None,
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

    click.echo("Checking for empty frames in image...")

    if empty_frames_idx is not None and len(empty_frames_idx) > 0:
        for key, value in data_dict.items():
            click.echo(f"Filling empty frames in {key}...")
            data_dict[key] = fill_empty_frames(value, empty_frames_idx)

    z_shape = data_dict[list(data_dict.keys())[0]].shape[1]

    # Apply preprocessing functions
    for key, func_details in preprocessing_functions.items():
        click.echo(f"Preprocessing {key} using {func_details.function}...")
        if "projection" in key:
            z_slices = resolve_z_slice(func_details.z_slices, z_shape)
            projection_channel = func_details.input_channel[0]
            data_dict[projection_channel] = data_dict[projection_channel][:, z_slices, :, :]
       
        input_channel_name = func_details.input_channel[0]
        function = resolve_function(func_details.function)  # Uses function mapping
        input_channel = data_dict[input_channel_name]

        kwargs = func_details.kwargs if func_details.kwargs else {}
        data_dict[input_channel_name] = array_apply(input_channel, func=function, **kwargs)
    # Generate foreground mask
    click.echo("Generating foreground mask...")
    foreground_func_details = preprocessing_functions["foreground"]
    foreground_function = resolve_function(
        foreground_func_details.function
    )  # Uses function mapping

    fg_input_channel = [data_dict[name] for name in foreground_func_details.input_channel]
    fg_kwargs = foreground_func_details.kwargs if foreground_func_details.kwargs else {}

    foreground_mask = array_apply(*fg_input_channel, func=foreground_function, **fg_kwargs)

    # Generate contour gradient map
    click.echo("Generating contour gradient map...")
    contour_func_details = preprocessing_functions["contour"]
    contour_function = resolve_function(contour_func_details.function)  # Uses function mapping

    contour_input_channel = [data_dict[name] for name in contour_func_details.input_channel]
    contour_kwargs = contour_func_details.kwargs if contour_func_details.kwargs else {}

    contour_gradient_map = array_apply(
        *contour_input_channel, func=contour_function, **contour_kwargs
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
    input_path: Path,
    output_dirpath: Path,
    preprocessing_functions: Dict[str, ProcessingFunctions],
    input_channels: List[str],
    tracking_config: MainConfig,
    segmentation_dirpath: Path = None,
    blank_frame_csv_path: Path = None,
    mode: str = "2D",
) -> None:
    """
    Process a single imaging position for cell tracking using virtual staining and optional label-free imaging.

    This function loads imaging data from the specified input directories, applies preprocessing steps, and
    performs object tracking using the provided configuration. It obtain the foreground and contour-based tracking
     to generate labeled tracking outputs.

    Parameters:
    - input_path (Path): Path to the virtual staining dataset in OME-Zarr format.
    - output_dirpath (Path): Path to save tracking results, including labels and tracks.
    - preprocessing_functions (Dict[str, ProcessingFunctions]): Dictionary of preprocessing functions applied
                                                                to the input images before tracking.
    - tracking_config (MainConfig): Configuration settings for the tracking algorithm.
    - input_channels (List[str]): List of input channels to use for tracking.   
    Returns:
    - None: The function also saves the tracking results, including labeled images and track data,
            to the specified output directory.

    Notes:
    - If blank frames exist in the data, the label-free image directory must be provided.
    - The function verifies the presence of required input channels and raises an error if any are missing.
    - Tracking is performed using the `ultrack` library, and the results are saved in an OME-Zarr format.
    - Tracks graphs are exported as CSV files.
    """

    position_key = input_path.parts[-3:]
    fov = "_".join(position_key)
    click.echo(f"Processing FOV: {fov.replace('_', '/')}")
    input_channels_preprocessing = input_channels["preprocessing"]
    input_channels_tracking = input_channels["tracking"]
    if (len(input_channels_tracking) != 1):
        raise ValueError("Only one channel is allowed and required for tracking.")
    if input_channels_preprocessing is not None:
        if input_channels_tracking[0] not in input_channels_preprocessing:
            raise ValueError("Tracking channel not found in Preprocessing input channels.")
        else:
            input_channels = input_channels_preprocessing
    else:
        input_channels = input_channels_tracking

    if segmentation_dirpath is not None:
        click.echo(f"Loading segmentation from: {segmentation_dirpath}...")
        segmentation_path = segmentation_dirpath/ Path(*position_key)

        label_dataset = open_ome_zarr(segmentation_path)
        label_arr = np.asarray(label_dataset[0][:, :, :, :, :]).astype(np.uint32)  # Shape (T, Z, Y, X)
        # get channel in tracking input channel
        channel_names = label_dataset.channel_names
        T, Z, Y, X = label_arr.shape
        if input_channels_tracking[0] not in channel_names:
            raise ValueError("Tracking channel not found in Segmentation input channels.")
        else:
            channel_index = channel_names.index(input_channels_tracking[0])
            label_arr = label_arr[:, channel_index, :, :, :]
        # get scale
        scale = label_dataset.scale
        shape = (T, 1, Z, Y, X)
        function = resolve_function(preprocessing_functions["segmentation_to_tracking"].function)
        kwargs = preprocessing_functions["segmentation_to_tracking"].kwargs if preprocessing_functions["segmentation_to_tracking"].kwargs else {}

        foreground_mask, contour_gradient_map = function(label_arr, **kwargs)

    else:
        click.echo("No segmentation provided, using preprocessing functions to obtain foreground and contour gradient map...") 
        click.echo(f"Reading data from: {input_path}...")
        with open_ome_zarr(input_path) as dataset:
            T, C, Z, Y, X = dataset.data.shape
            channel_names = dataset.channel_names
            scale = dataset.scale

        data_dict = {}
        for channel in input_channels:
            data_dict[channel] = dataset[0][:, channel_names.index(channel), :, :, :]
        
        if mode == "2D":
            scale = scale[-2:]
            shape = (T, 1, 1, Y, X)
        else:
            scale = scale[-3:]
            shape = (T, 1, Z, Y, X)

        
        # Preprocess to get the the foreground and multi-level contours
        blank_frame_df = pd.read_csv(blank_frame_csv_path) if blank_frame_csv_path else None
        empty_frames_idx=get_empty_frames_idx_from_csv(blank_frame_df, fov)

        foreground_mask, contour_gradient_map = data_preprocessing(
            data_dict=data_dict,
            preprocessing_functions=preprocessing_functions,
            empty_frames_idx=empty_frames_idx,
        )


    output_metadata = {
        "shape": shape,
        "chunks": None,
        "scale": scale,
        "channel_names": [f"{input_channels_tracking[0]}_labels"]
        ,
        "dtype": np.uint32,
        }

    create_empty_hcs_zarr(
                    store_path=output_dirpath, position_keys=[position_key], **output_metadata)
    
    # Define path to save the tracking database and graph
    filename = str(output_dirpath).split("/")[-1].split(".")[0]
    databaset_path = output_dirpath.parent / f"{filename}_config_tracking" / f"{fov}"
    os.makedirs(databaset_path, exist_ok=True)

    # Perform tracking
    click.echo("Tracking...")
    tracking_labels, tracks_df, _ = ultrack(
        tracking_config, foreground_mask, contour_gradient_map, scale, databaset_path
    )

    # Save the tracks graph to a CSV file
    csv_path = output_dirpath / Path(*position_key) / f"tracks_{fov}.csv"
    tracks_df.to_csv(csv_path, index=False)

    click.echo(f"Saved tracks to: {output_dirpath / Path(*position_key)}")

    # Save the tracking labels
    with open_ome_zarr(output_dirpath / Path(*position_key), mode="r+") as output_dataset:
        output_dataset[0][:, 0, 0] = np.asarray(tracking_labels)

def track(
    output_dirpath: str,
    config_filepath: str,
    input_position_dirpaths: List[str],
    segmentation_dirpath: str = None,
    sbatch_filepath: str = None,
    local: bool = None,
    blank_frame_csv_path: str = None,
) -> None:

    """
    Track nuclei and membranes in using virtual staining nuclei and membranes data.
    Parameters:

    - output_dirpath (str): Path to save tracking results, including labels and tracks.
    - config_filepath (str): Path to the tracking configuration file.
    - input_position_dirpaths (str): Path to the virtual staining dataset in OME-Zarr format.
    - segmentation_paths (str): Path to the segmentation dataset in OME-Zarr format.
    - sbatch_filepath (str): Path to the SLURM submission script.
    - local (bool): If True, run the tracking locally.
    - blank_frame_csv_path (str): Path to the blank frame CSV file.
    """

    output_dirpath = Path(output_dirpath)

    settings = yaml_to_model(config_filepath, TrackingSettings)
    tracking_cfg = settings.tracking_config


    # Get the shape of the data
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape

    # Estimate resources
    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16
    )
    tracking_cfg["segmentation_config"]["n_workers"] = num_cpus
    tracking_cfg["linking_config"]["n_workers"] = num_cpus

    # Create default instance
    default_config = MainConfig()
    tracking_cfg = update_model(default_config, tracking_cfg)


    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "tracking",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 60,
        "slurm_partition": "gpu",
        "slurm_use_srun": False,
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
        for input_position_path in input_position_dirpaths:
            job = executor.submit(
                track_one_position,
                input_path=input_position_path,
                segmentation_dirpath=segmentation_dirpath,
                output_dirpath=output_dirpath,
                tracking_config=tracking_cfg,
                preprocessing_functions=settings.preprocessing_functions,
                blank_frame_csv_path=blank_frame_csv_path, 
                input_channels=settings.input_channels,
                mode=settings.mode,
            )

            jobs.append(job)

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))


@click.command("track")
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
@click.option(
    "-segmentation_dirpath",
    "-s",
    required=False,
    default=None,
    type=str,
    help=" Path to the segmentation dataset in OME-Zarr format. If not provided, no segmentation will be used.",
)
@click.option(
    "-blank_frame_csv_path",
    "-f",
    required=False,
    default=None,
    type=str,
    help=" Blank Frame CSV path, if there are blanck frames in the data and the csv was previous gerenrated.",
)
def track_cli(
    input_position_dirpaths: List[str],
    segmentation_dirpath: List[str],
    output_dirpath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = None,
    blank_frame_csv_path: str = None,
) -> None:

    """
    Track nuclei and membranes in using virtual staining nuclei and membranes data.

    This function applied tracking to the virtual staining data for each position in the input.
    To use this function, install the lib as pip install -e .["track"]

    Example usage:

    biahub track -i virtual_staining.zarr/*/*/* -l lf_stabilize.zarr -o output.zarr -c config_tracking.yml

    """

    track(
        output_dirpath=output_dirpath,
        config_filepath=config_filepath,
        input_position_dirpaths=input_position_dirpaths,
        segmentation_dirpath=segmentation_dirpath,
        sbatch_filepath=sbatch_filepath,
        local=local,
        blank_frame_csv_path=blank_frame_csv_path,
    )
