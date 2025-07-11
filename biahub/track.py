import ast
import os

from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union

import click
import numpy as np
import pandas as pd
import submitit
import toml
import ultrack

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from ultrack import MainConfig, Tracker
from ultrack.utils.array import array_apply

from biahub.cli.parsing import (
    config_filepath,
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
from biahub.settings import ProcessingInputChannel, TrackingSettings


def mem_nuc_contour(nuclei_prediction: ArrayLike, membrane_prediction: ArrayLike) -> ArrayLike:
    """
    Compute a contour map at the boundary between nuclei and membranes.

    This function enhances boundary contrast by averaging the membrane signal with the
    inverse of the nucleus signal. The result is a contour-like representation that
    highlights the interface between nuclear and membrane regions.

    Parameters
    ----------
    nuclei_prediction : ArrayLike
        Array representing the predicted nuclear signal. Values are typically in [0, 1].
    membrane_prediction : ArrayLike
        Array representing the predicted membrane signal. Values are typically in [0, 1].

    Returns
    -------
    ArrayLike
        The resulting contour map, computed as the average of membrane and (1 - nucleus).
        Output has the same shape as the input arrays.

    Notes
    -----
    - Both input arrays must have the same shape.
    - Values are not thresholded; this function assumes soft probabilities or intensities.
    - The function is compatible with NumPy or Dask arrays (via `np.asarray` conversion).

    Examples
    --------
    >>> contour = mem_nuc_contour(nuc, mem)
    >>> contour.shape == nuc.shape
    True
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
    Resolve a function by its name from the predefined FUNCTION_MAP.

    This function looks up a string identifier in a centralized dictionary of allowed
    functions and returns the corresponding callable. It is used to dynamically map
    function names (e.g., from config files) to actual Python functions.

    Parameters
    ----------
    function_name : str
        The fully qualified name of the function to retrieve
        (e.g., "np.mean", "ultrack.imgproc.gradient_magnitude").

    Returns
    -------
    Callable
        The resolved function object.

    Raises
    ------
    ValueError
        If the function name is not found in the `FUNCTION_MAP`.

    Notes
    -----
    - `FUNCTION_MAP` is a global dictionary that includes a whitelist of safe,
      user-approved or library-provided functions.
    - Additional functions (e.g., custom preprocessing functions) can be manually added
      to `FUNCTION_MAP`.
    """
    if function_name not in FUNCTION_MAP:
        raise ValueError(
            f"Invalid function '{function_name}'. Allowed functions: {list(FUNCTION_MAP.keys())}"
        )

    return FUNCTION_MAP[function_name]


def fill_empty_frames(arr: ArrayLike, empty_frames_idx: List[int]) -> ArrayLike:
    """
    Fill empty frames in a time-series imaging array using nearest available frames.

    This function modifies a temporal image stack by replacing specified empty frames
    (e.g., entirely blank or corrupted) with the nearest valid frame. For leading/trailing
    empty frames, the nearest single neighbor is used. For interior empty frames, it
    prioritizes previous-frame filling but can fall back to future frames if needed.

    Parameters
    ----------
    arr : ArrayLike
        Time-series imaging data of shape (T, Y, X) or (T, C, Y, X). The first dimension must be time.
    empty_frames_idx : list of int
        List of time indices (frame numbers) that are considered empty and should be filled.

    Returns
    -------
    ArrayLike
        The modified array with empty frames replaced in-place using temporal neighbors.

    Notes
    -----
    - This function modifies the input array in-place.
    - If no valid neighbors are available (e.g., all frames empty), the original frame is left unchanged.
    - For performance and reproducibility, the function uses a simple sequential search rather than interpolation.

    Examples
    --------
    >>> arr.shape
    (10, 1, 256, 256)  # T, C, Y, X
    >>> empty_frames_idx = [0, 4]
    >>> arr_filled = fill_empty_frames(arr, empty_frames_idx)
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
    Extract the indices of empty timepoints for a given field of view (FOV) from a DataFrame.

    This function parses a DataFrame that contains information about blank (empty) frames
    across different FOVs and returns a list of time indices corresponding to empty frames
    for the specified FOV.

    Parameters
    ----------
    blank_frame_df : pandas.DataFrame
        DataFrame containing at least two columns: 'FOV' (str) and 't' (list-like or int).
        The 't' column may contain strings representing Python lists (e.g., "[0, 3, 5]").
    fov : str
        Field of view identifier used to filter the DataFrame.

    Returns
    -------
    list of int or None
        List of integer indices representing empty timepoints for the specified FOV.
        Returns `None` if no matching FOV is found or if no empty frames are reported.

    Notes
    -----
    - The function uses `ast.literal_eval` to parse stringified lists from CSV files.
    - If `t` is the integer 0 (and not a list), it assumes no empty frames and returns `None`.

    Examples
    --------
    >>> df = pd.DataFrame({'FOV': ['A/1/1'], 't': ['[0, 2, 4]']})
    >>> get_empty_frames_idx_from_csv(df, 'A/1/1')
    [0, 2, 4]
    """

    empty_frames_idx = blank_frame_df[blank_frame_df['FOV'] == fov]['t']
    if not empty_frames_idx.empty:
        t_value = empty_frames_idx.iloc[0]
        if isinstance(t_value, str) and t_value.startswith('['):
            t_value = ast.literal_eval(t_value)
        if isinstance(t_value, list):
            return [int(i) for i in t_value]
        elif t_value == 0:
            return None
    return None


def central_z_slice(z_shape: int) -> slice:
    """
    Compute a centered Z-slice range from a 3D or 4D image volume.

    This function returns a slice object that selects a centered range of Z-planes
    from a volumetric dataset, based on the total number of Z slices. It ensures
    that the returned slice includes an odd number of planes (at least 3).

    Parameters
    ----------
    z_shape : int
        The total number of Z-planes (depth) in the dataset.

    Returns
    -------
    slice
        A slice object that extracts a centered Z-range from the dataset.

    Notes
    -----
    - Ensures that at least 3 slices are returned.
    - If the total number of slices is even, one extra slice is included to make it odd.
    - This function is typically used when `z_slices=(0, 0)` is provided,
      indicating that the user wants automatic central slicing.

    Examples
    --------
    >>> central_z_slice(21)
    slice(10 - 1, 10 + 1 + 1)  # slice(9, 12)

    >>> central_z_slice(8)
    slice(3, 6)  # (center=4, half_window=1)
    """
    n_slices = max(3, z_shape // 2)
    if n_slices % 2 == 0:
        n_slices += 1
    z_center = z_shape // 2
    half_window = n_slices // 2
    return slice(z_center - half_window, z_center + half_window + 1)


def resolve_z_slice(z_slices: Tuple[int, int], z_shape: int, mode: str = "2D") -> slice:
    """
    Resolve the z-slice range based on user-defined input and imaging mode.

    Parameters
    ----------
    z_slices : Tuple[int, int]
        Start and end indices of the z-range. If set to (0, 0), automatic center range is used.
    z_shape : int
        Total number of slices along the Z axis.
    mode : str, optional
        If "2D", returns a centered Z-range slice. If not "2D", returns full or specified range. Default is "2D".

    Returns
    -------
    slice
        A slice object representing the Z-range to extract.
    """
    if z_slices is None or z_slices == (0, 0):
        if mode == "2D":
            return central_z_slice(z_shape)
        else:
            return slice(None)
    else:
        return slice(*z_slices)


def run_ultrack(
    tracking_config: MainConfig,
    foreground_mask: ArrayLike,
    contour_gradient_map: ArrayLike,
    scale: Union[Tuple[float, float], Tuple[float, float, float]],
    databaset_path,
):
    """
    Run object tracking using the Ultrack library.

    This function performs object tracking on time-series image data using a binary
    foreground mask and a contour gradient map. It outputs labeled segmentation results,
    a track DataFrame, and a graph representing object trajectories over time.

    Parameters
    ----------
    tracking_config : MainConfig
        Ultrack configuration object defining segmentation, linking, and optimization parameters.
    foreground_mask : ArrayLike
        Binary mask (0 or 1) indicating detected object regions over time.
        Shape is typically (T, Z, Y, X) or (T, Y, X).
    contour_gradient_map : ArrayLike
        Gradient map or edge score map used to refine object boundaries and define connectivity
        for linking detected objects between timepoints.
    scale : tuple of float
        Physical resolution scale in either (Y, X) or (Z, Y, X) depending on whether the data is 2D or 3D.
    databaset_path : Path
        Directory where tracking results, configuration files, and output data will be saved.

    Returns
    -------
    labels : np.ndarray
        Labeled segmentation array of shape (T, Z, Y, X) or (T, Y, X), where each object instance
        is assigned a unique integer label across time.
    tracks_df : pandas.DataFrame
        DataFrame with tracking metadata including object ID, frame index, spatial coordinates,
        and parent-child relationships.
    graph : networkx.Graph
        Directed graph representing tracked object lineages. Nodes correspond to objects and
        edges represent links between objects across frames.

    Notes
    -----
    - `foreground_mask` must be a binary mask (e.g., after thresholding a probability map).
    - This function modifies `tracking_config` to set the working directory (`working_dir`).
    - The configuration used is saved to `config.toml` under the `databaset_path`.

    Examples
    --------
    >>> labels, tracks_df, graph = run_ultrack(
    ...     tracking_config=cfg,
    ...     foreground_mask=binary_mask,
    ...     contour_gradient_map=gradient_map,
    ...     scale=(0.5, 0.5, 1.0),
    ...     databaset_path=Path("results/posA")
    ... )
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


def run_preprocessing_pipeline(
    data_dict: Dict[str, ArrayLike],
    input_images: List[ProcessingInputChannel],
    visualize: bool = False,
) -> Dict[str, ArrayLike]:
    """
    Run a configurable preprocessing pipeline on input image channels.

    This function applies a sequence of user-defined functions (e.g., filters, transformations)
    to each specified channel in the dataset. Function pipelines are defined via the
    `ProcessingInputChannel` objects, which describe the function, its arguments, and whether
    it should be applied per timepoint.

    Parameters
    ----------
    data_dict : dict of str to ArrayLike
        A dictionary where each key is a channel name and each value is the corresponding
        multi-dimensional image array (typically of shape (T, Z, Y, X) or (T, C, Z, Y, X)).
    input_images : list of ProcessingInputChannel
        A list of input image specifications. Each `ProcessingInputChannel` defines
        which channels to process and the pipeline (functions + arguments) to apply.
    visualize : bool, optional
        If True, opens a Napari viewer to display each processed channel after its pipeline.

    Returns
    -------
    dict of str to ArrayLike
        Updated dictionary where the processed channel data has replaced the originals.

    Notes
    -----
    - The `step.input_channels` field can define which channel(s) to use as inputs
      to a function, even if different from the one being written to.
    - If `per_timepoint` is True for a step, the function will be applied frame-by-frame
      using `ultrack.utils.array.array_apply`.
    - All functions must be registered in the `FUNCTION_MAP` to be resolved.

    Examples
    --------
    >>> from biahub.settings import ProcessingInputChannel, ProcessingFunctions
    >>> import numpy as np

    >>> data_dict = {"raw": np.random.rand(10, 1, 256, 256)}  # shape (T, Z, Y, X)

    >>> input_images = [
    ...     ProcessingInputChannel(
    ...         path=None,
    ...         channels={
    ...             "raw": [
    ...                 ProcessingFunctions(
    ...                     function="np.mean",
    ...                     kwargs={"axis": 1},
    ...                     per_timepoint=False,
    ...                     input_channels=["raw"]
    ...                 )
    ...             ]
    ...         }
    ...     )
    ... ]

    >>> output = run_pipeline(data_dict, input_images)
    >>> output["raw"].shape
    (10, 256, 256)  # Z-averaged
    """
    for image in input_images:
        for channel_name, pipeline in image.channels.items():
            for step in pipeline:
                click.echo(f"Processing {channel_name} with {step.function}")
                f_name = step.function
                run_function = resolve_function(f_name)
                f_kwargs = step.kwargs
                per_timepoint = step.per_timepoint
                # if there is input channel, apply the function to the input channel otherwise apply the function to the output channel
                f_channel_name = step.input_channels
                if f_channel_name is None:
                    f_channel_name = [channel_name]
                f_data = [np.asarray(data_dict[name]) for name in f_channel_name]

                if per_timepoint:
                    result = array_apply(*f_data, func=run_function, **f_kwargs)

                else:
                    result = run_function(*f_data, **f_kwargs)

                data_dict[channel_name] = result
                if visualize:
                    import napari

                    viewer = napari.Viewer()
                    viewer.add_image(data_dict[channel_name], name=channel_name)

    return data_dict


def load_data(
    position_key: Tuple[str, str, str],
    input_images: List[ProcessingInputChannel],
    z_slices: slice,
    visualize: bool = False,
) -> Dict[str, ArrayLike]:
    """
    Load and extract specified channels from an OME-Zarr dataset for a given position.

    This function opens the OME-Zarr dataset corresponding to a specific position key,
    extracts the required channels and Z-slices as defined in the `input_images` configuration,
    and stores them in a dictionary for further processing.

    Parameters
    ----------
    position_key : tuple of str
        A tuple of three strings representing the hierarchical path to a specific position
        in the dataset (e.g., (plate, well, position)).
    input_images : list of ProcessingInputChannel
        List of input image channel configurations. Each item defines the path and channels to load.
    z_slices : slice
        Slice object specifying which Z-planes to extract (e.g., central slices for 2D mode).
    visualize : bool, optional
        If True, opens a Napari viewer and displays each loaded channel. Default is False.

    Returns
    -------
    dict of str to ArrayLike
        Dictionary mapping each channel name to its corresponding image data array.
        Each array is a Dask array of shape (T, Z, Y, X), extracted from the OME-Zarr store.

    Notes
    -----
    - Assumes that each `ProcessingInputChannel` has a valid `.path` attribute pointing to a Zarr store.
    - Uses `read_data()` to extract metadata and index the correct channel.
    - Channel names must match those present in the dataset.
    """
    data_dict = {}
    for image in input_images:
        for channel_name, _ in image.channels.items():
            # load the data from the zarr path
            if image.path is not None:
                click.echo(f"Loading data for channel {channel_name} from {image.path}")
                image_path = image.path / Path(*position_key)
                with open_ome_zarr(image_path) as dataset:
                    image_channel_names = dataset.channel_names
                    data_dict[channel_name] = dataset.data.dask_array()[
                        :, image_channel_names.index(channel_name), z_slices, :, :
                    ]
                if visualize:
                    import napari

                    viewer = napari.Viewer()
                    viewer.add_image(data_dict[channel_name], name=channel_name)
    return data_dict


def fill_empty_frames_from_csv(
    fov: str,
    data_dict: Dict[str, ArrayLike],
    blank_frame_csv_path: Path,
) -> Dict[str, ArrayLike]:
    """
    Fill empty timepoints in a multi-channel image dictionary using a CSV file of blank frames.

    This function loads a list of empty time indices from a CSV and fills them using the
    nearest non-empty frames in the data.

    Parameters
    ----------
    data_dict : dict of str to ArrayLike
        Dictionary mapping channel names to image arrays.
    csv_path : Path
        Path to the CSV file containing empty frame information.
        The CSV must contain columns: 'FOV' and 't'.
    fov : str
        Field of view identifier used to filter the CSV rows.

    Returns
    -------
    dict of str to ArrayLike
        Updated dictionary with empty frames filled in-place.
    """
    blank_frame_df = pd.read_csv(blank_frame_csv_path) if blank_frame_csv_path else None
    empty_frames_idx = get_empty_frames_idx_from_csv(blank_frame_df, fov)
    for channel_name, channel_data in data_dict.items():
        data_dict[channel_name] = fill_empty_frames(channel_data, empty_frames_idx)
    return data_dict


def data_preprocessing(
    position_key: str,
    input_images: List[ProcessingInputChannel],
    z_slices: slice,
    blank_frames_path: Path = None,
    visualize: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load, preprocess, and prepare image data for tracking.

    This function performs the full preprocessing pipeline on a single field of view (FOV).
    It loads Z-sliced image data, applies configured preprocessing functions, fills in any
    empty timepoints using a blank frame CSV (if provided), and returns the two required
    channels for tracking: the foreground mask and the contour gradient map.

    Parameters
    ----------
    position_key : str
        FOV key, typically formed as "Plate_Well_Position" (e.g., "A_1_3").
    input_images : list of ProcessingInputChannel
        Configuration defining which channels to load and how to process them.
    z_slices : slice
        Slice object selecting which Z-planes to load (e.g., central slices).
    blank_frames_path : Path, optional
        Optional path to a CSV file containing frame indices to be filled for each FOV.
        If not provided, empty frame handling is skipped.
    visualize : bool, optional
        If True, opens Napari viewer to visualize each intermediate step.

    Returns
    -------
    dict of str to np.ndarray
        Dictionary containing two arrays:
        - "foreground" : binary mask array for detected objects
        - "contour"    : gradient/edge map for contour-based refinement

    Raises
    ------
    ValueError
        If neither 'foreground' and 'contour' nor 'foreground_contour' channels are found
        in the processed data.

    Examples
    --------
    >>> z_slice = slice(10, 15)
    >>> foreground, contour = data_preprocessing(
    ...     position_key="A_1_3",
    ...     input_images=config.input_images,
    ...     z_slices=z_slice,
    ...     blank_frames_path=Path("blank_frames.csv"),
    ...     visualize=False
    ... )
    >>> foreground.shape, contour.shape
    ((10, 5, 256, 256), (10, 5, 256, 256))
    """
    fov = "_".join(position_key)
    data_dict = load_data(
        position_key=position_key,
        input_images=input_images,
        z_slices=z_slices,
        visualize=visualize,
    )
    data_dict = run_preprocessing_pipeline(data_dict, input_images, visualize=visualize)
    data_dict = fill_empty_frames_from_csv(fov, data_dict, blank_frames_path)

    # get foreground and contour
    channel_names = data_dict.keys()
    if "foreground" in channel_names and "contour" in channel_names:
        foreground_mask, contour_gradient_map = data_dict["foreground"], data_dict["contour"]
    elif "foreground_contour" in channel_names:
        foreground_mask, contour_gradient_map = data_dict["foreground_contour"]
    else:
        raise ValueError("Foreground and contour channels are required for tracking.")
    del data_dict
    return foreground_mask, contour_gradient_map


def track_one_position(
    position_key: str,
    input_images: List[ProcessingInputChannel],
    output_dirpath: Path,
    tracking_config: MainConfig,
    blank_frames_path: Path = None,
    z_slices: Tuple[int, int] = (0, 0),
    scale: Tuple[float, float, float, float, float] = (1, 1, 1, 1, 1),
) -> None:
    """
    Run tracking on a single field of view using foreground and contour channel data.

    This function loads image data, applies a preprocessing pipeline, fills blank frames if needed,
    and uses the Ultrack library to compute object tracks. It is agnostic to the imaging source —
    as long as the pipeline produces a binary foreground mask and a corresponding contour map.
    Parameters
    ----------
    position_key : str
        A string identifier for the field of view (e.g., "A_1_3"), typically composed of
        Plate, Well, and Position joined by underscores.
    input_images : list of ProcessingInputChannel
        Configuration describing which channels to load and how to preprocess them.
    output_dirpath : Path
        Output directory where labeled Zarr volumes and track CSVs will be stored.
    tracking_config : MainConfig
        Ultrack configuration containing segmentation, linking, and optimization settings.
    blank_frames_path : Path, optional
        Path to CSV file indicating empty frames for the current FOV. If None, blank frame
        filling is skipped.
    z_slices : tuple of int, optional
        Tuple specifying the range of Z-slices to load. If (0, 0), the central slice range
        will be auto-resolved. Default is (0, 0).
    scale : tuple of float, optional
        Physical scale of the dataset in (T, C, Z, Y, X) order. Tracking uses either the
        (Z, Y, X) or (Y, X) portion depending on dimensionality. Default is (1, 1, 1, 1, 1).

    Returns
    -------
    None
        Outputs are saved directly to disk:
        - Tracked object labels in Zarr format
        - CSV file containing the track graph (IDs, positions, parents)

    Notes
    -----
    - Output is saved to `{output_dirpath}/{position_key}/tracks_{position_key}.csv`.
    - The Ultrack config is also saved as TOML in a `_config_tracking/{FOV}/` subdirectory.
    - If required input channels ("foreground" and "contour") are missing, a ValueError is raised.

    Examples
    --------
    >>> track_one_position(
    ...     position_key="A_1_3",
    ...     input_images=config.input_images,
    ...     output_dirpath=Path("output.zarr"),
    ...     tracking_config=cfg,
    ...     blank_frames_path=Path("blank_frames.csv"),
    ...     z_slices=(10, 15),
    ...     scale=(1, 1, 0.5, 0.2, 0.2)
    ... )
    """

    fov = "_".join(position_key)
    click.echo(f"Processing FOV: {fov.replace('_', '/')}")
    # tracking input images
    foreground_mask, contour_gradient_map = data_preprocessing(
        position_key, input_images, z_slices, blank_frames_path
    )

    # Define path to save the tracking database and graph
    filename = str(output_dirpath).split("/")[-1].split(".")[0]
    databaset_path = output_dirpath.parent / f"{filename}_config_tracking" / f"{fov}"
    os.makedirs(databaset_path, exist_ok=True)

    # Perform tracking
    click.echo("Tracking...")
    tracking_labels, tracks_df, _ = run_ultrack(
        tracking_config, foreground_mask, contour_gradient_map, scale, databaset_path
    )

    # Save the tracks graph to a CSV file
    csv_path = output_dirpath / Path(*position_key) / f"tracks_{fov}.csv"
    os.makedirs(csv_path.parent, exist_ok=True)

    tracks_df.to_csv(csv_path, index=False)

    click.echo(f"Saved tracks to: {output_dirpath / Path(*position_key)}")

    # Save the tracking labels
    with open_ome_zarr(output_dirpath / Path(*position_key), mode="r+") as output_dataset:
        output_dataset[0][:, 0, 0] = np.asarray(tracking_labels, dtype=np.uint32)


def track(
    output_dirpath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = None,
) -> None:
    """
    Launch tracking jobs for multiple imaging positions using foreground and contour data.

    This function orchestrates the tracking of cell trajectories across multiple fields of view (FOVs).
    It supports any imaging modality, as long as preprocessing produces the required foreground mask
    and contour gradient map for tracking.
        Parameters
    ----------
    output_dirpath : str
        Path to the Zarr store where output labeled segmentations and track data will be saved.
    config_filepath : str
        Path to the YAML configuration file containing tracking and preprocessing settings.
    sbatch_filepath : str, optional
        Path to a SLURM batch script to override default SLURM job parameters. If not provided,
        defaults for CPUs, memory, and parallelism are used.
    local : bool, optional
        If True, runs all tracking jobs sequentially on the local machine instead of submitting via SLURM.

    Returns
    -------
    None
        Results are written directly to disk in the `output_dirpath`. Also logs Submitit job IDs.

    Notes
    -----
    - This function mirrors the input Zarr structure in the output store.
    - Tracking is distributed per FOV using Submitit SLURM array jobs by default.
    - Tracking configurations (`n_workers`, scale, and output shapes) are inferred from the first FOV.
    - Output logs are saved to `output_dirpath/../slurm_output/submitit_jobs_ids.log`.

    Examples
    --------
    >>> track(
    ...     output_dirpath="output.zarr",
    ...     config_filepath="config_tracking.yml",
    ...     sbatch_filepath="track_job.sbatch",
    ...     local=False,
    ... )
    """

    output_dirpath = Path(output_dirpath)

    settings = yaml_to_model(config_filepath, TrackingSettings)

    input_images_paths = [
        image.path for image in settings.input_images if image.path is not None
    ]
    if len(input_images_paths) < 1:
        raise ValueError("No input_images_paths provided")
    fov = settings.fov

    # check if all input_images_paths have the same position keys
    input_position_dirpaths = [Path(p) for p in glob(str(input_images_paths[0] / fov))]
    position_keys = [p.parts[-3:] for p in input_position_dirpaths]

    tracking_cfg = settings.tracking_config

    # Get the shape of the data
    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape
        scale = dataset.scale
        shape = (T, C, Z, Y, X)

    # Resolve z-slices
    z_slices = resolve_z_slice(settings.z_range, shape[2], settings.mode)

    # Define output metadata
    if settings.mode == "2D":
        output_shape = (T, 1, 1, Y, X)
        track_scale = scale[-2:]
    else:
        output_shape = (T, 1, Z, Y, X)
        track_scale = scale[-3:]

    output_metadata = {
        "shape": output_shape,
        "chunks": None,
        "scale": scale,
        "channel_names": [f"{settings.target_channel}_labels"],
        "dtype": np.uint32,
    }

    # Create the output zarr mirroring input_position_dirpaths
    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=position_keys,
        **output_metadata,
    )

    # Estimate resources
    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16
    )

    # Use the number of CPUs and RAM per CPU in the tracking configuration
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
        "slurm_partition": "preempted",
        "slurm_gpus_per_node": 1,
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
        for position_key in position_keys:
            job = executor.submit(
                track_one_position,
                position_key=position_key,
                output_dirpath=output_dirpath,
                tracking_config=tracking_cfg,
                input_images=settings.input_images,
                blank_frames_path=settings.blank_frames_path,
                z_slices=z_slices,
                scale=track_scale,
            )

            jobs.append(job)

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))


@click.command("track")
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
def track_cli(
    output_dirpath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = None,
) -> None:
    """
    Track objects in 2D or 3D time-lapse microscopy data using configurable preprocessing.

    This command applies preprocessing, handles optional blank frame filling, and performs
    object tracking on each position using the Ultrack library. Compatible with any image
    modality as long as it produces 'foreground' and 'contour' inputs.

    Example usage:

    biahub track -i virtual_staining.zarr/*/*/* -o output.zarr -c config_tracking.yml

    """

    track(
        output_dirpath=output_dirpath,
        config_filepath=config_filepath,
        sbatch_filepath=sbatch_filepath,
        local=local,
    )


if __name__ == "__main__":
    track_cli()  # pylint: disable=no-value-for-parameter
