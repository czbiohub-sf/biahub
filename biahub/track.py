import ast
import logging
import os

from pathlib import Path

import click
import dask.array as da
import numpy as np
import pandas as pd
import submitit

from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from numpy.typing import ArrayLike
from tqdm import tqdm

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.resolve_function import resolve_function
from biahub.cli.utils import (
    echo_resources,
    estimate_resources,
    get_submitit_cluster,
    resolve_ome_zarr_version,
    update_model,
    yaml_to_model,
)
from biahub.settings import CellposeConfig, ProcessingInputChannel, TrackingSettings

logger = logging.getLogger(__name__)

# Optical parameters for waveorder focus finding (focus_from_transverse_band),
# shared with biahub.estimate_stabilization. Used only when a config sets
# use_focus: true to resolve the in-focus z-window per FOV.
NA_DET = 1.35
LAMBDA_ILL = 0.500

# Lazy imports for ultrack - imported only when needed in specific functions

_ultrack_patched = False


def _patch_ultrack_readonly_buffer():
    """Workaround for scikit-image #6378: read-only buffer in _map_array.

    scikit-image's Cython ``_map_array`` rejects read-only memoryviews.
    Under numpy 2.x, pandas DataFrame columns and ``np.asarray`` results are
    often read-only, so every call path through ``map_array`` can trigger this.

    Patch ``skimage.util.map_array`` directly to force writable copies of
    all arrays before passing them to Cython.

    Remove when scikit-image >= 0.27 is available.
    """
    global _ultrack_patched
    if _ultrack_patched:
        return
    _ultrack_patched = True

    import skimage.util._map_array as _ma

    _original_map_array = _ma.map_array

    def _map_array_writable(input_arr, input_vals, output_vals, out=None):
        input_arr = np.array(input_arr, copy=False)
        if not input_arr.flags.writeable:
            input_arr = input_arr.copy()
        input_vals = np.array(input_vals, copy=False)
        if not input_vals.flags.writeable:
            input_vals = input_vals.copy()
        output_vals = np.array(output_vals, copy=False)
        if not output_vals.flags.writeable:
            output_vals = output_vals.copy()
        return _original_map_array(input_arr, input_vals, output_vals, out=out)

    _ma.map_array = _map_array_writable


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


CUSTOM_FUNCTIONS = {
    "biahub.track.mem_nuc_contour": mem_nuc_contour,
}


def fill_empty_frames(arr: ArrayLike, empty_frames_idx: list[int]) -> ArrayLike:
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


def get_empty_frames_idx_from_csv(blank_frame_df: pd.DataFrame, fov: str) -> list[int] | None:
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
    >>> df = pd.DataFrame({"FOV": ["A/1/1"], "t": ["[0, 2, 4]"]})
    >>> get_empty_frames_idx_from_csv(df, "A/1/1")
    [0, 2, 4]
    """
    empty_frames_idx = blank_frame_df[blank_frame_df["FOV"] == fov]["t"]
    if not empty_frames_idx.empty:
        t_value = empty_frames_idx.iloc[0]
        if isinstance(t_value, str) and t_value.startswith("["):
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


def _median_focus_plane(data, z_shape: int, pixel_size: float, channel_index: int) -> int:
    """Median in-focus z-plane across timepoints for one channel.

    Uses waveorder ``focus_from_transverse_band`` (with the module constants
    ``NA_DET``/``LAMBDA_ILL``) per timepoint; empty frames fall back to the
    central plane. ``data`` is a (T, C, Z, Y, X) array-like.
    """
    from waveorder.focus import focus_from_transverse_band

    z_focus = []
    for t in tqdm(range(data.shape[0]), desc="Finding focus"):
        zyx = np.asarray(data[t, channel_index, :, :, :])
        if zyx.sum() == 0:
            z_focus.append(z_shape // 2)
            continue
        z_f = focus_from_transverse_band(
            zyx, NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size
        )
        z_focus.append(z_shape // 2 if z_f is None else int(np.clip(z_f, 0, z_shape - 1)))
    return int(np.median(z_focus))


def resolve_z_slice(
    z_range: tuple[int, int],
    z_shape: int,
    *,
    data=None,
    pixel_size: float | None = None,
    use_focus: bool = False,
    z_total: int | None = None,
    frac_below: float = 1 / 3,
    frac_above: float = 2 / 3,
    focus_channel_index: int = 0,
) -> tuple[slice, int]:
    """
    Resolve the z-slice range based on user input and imaging mode.

    This function determines which Z-planes to extract from a 3D or 4D volume based on:
    - Focus finding, if `use_focus` is True: the in-focus plane is detected per
      timepoint (waveorder ``focus_from_transverse_band`` on ``focus_channel_index``,
      using the module constants ``NA_DET``/``LAMBDA_ILL`` and ``pixel_size``), and a window
      of ``z_total`` planes is placed around the median focus (``frac_below`` below,
      ``frac_above`` above). ``z_range`` is ignored in this mode.
    - A user-specified Z-range tuple `(start, stop)`, or
    - Automatic central slicing if `(-1, -1)` is passed.
    - All planes are returned if `None` is passed.

    Validation ensures that the resulting slice includes at least one Z-plane.
    Odd-length slices are not required.

    Parameters
    ----------
    z_range : Tuple[int, int]
        The (start, stop) indices for slicing Z-planes. If (-1, -1), central slicing will be used.
        If None, all planes are returned.
    z_shape : int
        Total number of Z-planes in the dataset (e.g., `shape[2]`).

    Returns
    -------
    slice
        A slice object selecting the requested Z-range from the dataset.
    int
        The number of Z-planes in the selected range.

    Raises
    ------
    ValueError
        If the user-provided slice range is invalid (e.g., stop <= start).

    Examples
    --------
    >>> resolve_z_slice((5, 10), z_shape=30)
    (slice(5, 10), 5)

    >>> resolve_z_slice((-1, -1), z_shape=21)
    (slice(9, 12), 3)  # 3 central slices

    >>> resolve_z_slice((-1, -1), z_shape=21)
    (slice(None), 21)
    """
    if use_focus:
        if data is None or pixel_size is None or z_total is None:
            raise ValueError(
                "use_focus=True requires data, pixel_size, and z_total to be provided."
            )
        center = _median_focus_plane(data, z_shape, pixel_size, focus_channel_index)
        start = max(0, center - int(round(frac_below * z_total)))
        stop = min(z_shape, center + int(round(frac_above * z_total)))
        if stop <= start:
            raise ValueError(
                f"Focus z-window is empty (start={start}, stop={stop}); check z_total={z_total}."
            )
        return slice(start, stop), stop - start

    if z_range == (-1, -1):
        z_slices = central_z_slice(z_shape)
        Z = z_slices.stop - z_slices.start
    elif z_range is None:
        z_slices = slice(None)
        Z = z_shape
    else:
        start, stop = z_range
        if stop <= start:
            raise ValueError(
                f"Invalid Z-slice range {z_range}: must contain at least one slice (stop > start)."
            )
        z_slices = slice(start, stop)
        Z = stop - start
    return z_slices, Z


def run_ultrack(
    tracking_config,
    database_path,
    **track_kwargs,
):
    """
    Run object tracking using the Ultrack library.

    Note: ultrack is imported lazily within this function.

    This function performs object tracking on time-series image data and outputs labeled
    segmentation results, a track DataFrame, and a graph of object trajectories over time.
    The detection inputs are forwarded to ``Tracker.track()`` via ``**track_kwargs``, so it
    accepts either the foreground+contour pair (``detection``, ``edges``) or precomputed
    instance ``labels`` (with ``sigma``), plus ``scale`` and ``overwrite``.

    Parameters
    ----------
    tracking_config : MainConfig
        Ultrack configuration object defining segmentation, linking, and optimization parameters.
    database_path : Path
        Directory where tracking results, configuration files, and output data will be saved.
    **track_kwargs
        Keyword arguments forwarded to ``Tracker.track()``. E.g. ``detection``, ``edges``
        (foreground+contour mode), or ``labels``, ``sigma`` (cellpose mode), plus ``scale``
        and ``overwrite``.

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
    - This function modifies `tracking_config` to set the working directory (`working_dir`).
    - The configuration used is saved to `config.toml` under the `database_path`.
    """
    import toml

    from ultrack import Tracker

    _patch_ultrack_readonly_buffer()

    cfg = tracking_config

    cfg.data_config.working_dir = database_path

    click.echo(str(cfg))

    tracker = Tracker(cfg)
    tracker.track(**track_kwargs)

    tracks_df, graph = tracker.to_tracks_layer()
    labels = tracker.to_zarr(
        tracks_df=tracks_df,
    )

    with open(database_path / "config.toml", mode="w") as f:
        toml.dump(cfg.dict(by_alias=True), f)

    return (
        labels,
        tracks_df,
        graph,
    )


def run_preprocessing_pipeline(
    data_dict: dict[str, ArrayLike],
    input_images: list[ProcessingInputChannel],
    visualize: bool = False,
) -> dict[str, ArrayLike]:
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
    """
    import napari

    from ultrack.utils.array import array_apply

    for image in input_images:
        for channel_name, pipeline in image.channels.items():
            for step in pipeline:
                click.echo(f"Processing {channel_name} with {step.function}")
                f_name = step.function
                run_function = resolve_function(f_name, custom_functions=CUSTOM_FUNCTIONS)
                f_kwargs = step.kwargs
                per_timepoint = step.per_timepoint
                # if there is input channel, apply the function to the input channel otherwise apply the function to the output channel
                f_channel_name = step.input_channels
                if f_channel_name is None:
                    f_channel_name = [channel_name]
                f_data = [
                    (
                        data_dict[name].compute()
                        if isinstance(data_dict[name], da.Array)
                        else np.asarray(data_dict[name])
                    )
                    for name in f_channel_name
                ]
                if per_timepoint:
                    result = array_apply(*f_data, func=run_function, **f_kwargs)

                else:
                    result = run_function(*f_data, **f_kwargs)

                data_dict[channel_name] = result
                if visualize:
                    viewer = napari.Viewer()
                    viewer.add_image(data_dict[channel_name], name=channel_name)

    return data_dict


def load_data(
    position_key: tuple[str, str, str],
    input_images: list[ProcessingInputChannel],
    z_slices: slice,
    visualize: bool = False,
) -> dict[str, ArrayLike]:
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
    import napari

    data_dict = {}
    for image in input_images:
        # load the data from the zarr path
        if image.path is not None:
            image_path = image.path / Path(*position_key)
            with open_ome_zarr(image_path) as dataset:
                image_channel_names = dataset.channel_names
                for channel_name, _ in image.channels.items():
                    click.echo(f"Loading data for channel {channel_name} from {image.path}")
                    data_dict[channel_name] = dataset.data.dask_array()[
                        :, image_channel_names.index(channel_name), z_slices, :, :
                    ]
            if visualize:
                viewer = napari.Viewer()
                viewer.add_image(data_dict[channel_name], name=channel_name)
    return data_dict


def fill_empty_frames_from_csv(
    fov: str,
    data_dict: dict[str, ArrayLike],
    blank_frame_csv_path: Path,
) -> dict[str, ArrayLike]:
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
    if blank_frame_csv_path:
        blank_frame_df = pd.read_csv(blank_frame_csv_path)
        empty_frames_idx = get_empty_frames_idx_from_csv(blank_frame_df, fov)
        for channel_name, channel_data in data_dict.items():
            data_dict[channel_name] = fill_empty_frames(channel_data, empty_frames_idx)

    return data_dict


def data_preprocessing(
    position_key: str,
    input_images: list[ProcessingInputChannel],
    z_slices: slice,
    blank_frames_path: Path = None,
    visualize: bool = False,
) -> dict[str, np.ndarray]:
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
    """
    fov = "/".join(position_key)
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


def run_cellpose_per_frame(
    images: np.ndarray,
    model_type: str = "nuclei",
    diameter: float = 80,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    gpu: bool = True,
    min_size: int = 500,
) -> np.ndarray:
    """Run cellpose on each 2D frame of a (T, Y, X) array.

    Returns an integer label array of shape (T, Y, X).

    Note: cellpose is imported lazily so that importing ``biahub.track`` does not
    require the ``segment`` extra unless cellpose segmentation is actually used.
    """
    from cellpose import models as cp_models

    model = cp_models.CellposeModel(model_type=model_type, gpu=gpu)

    T = images.shape[0]
    labels = np.zeros_like(images, dtype=np.int32)
    for t in tqdm(range(T), desc="Cellpose segmentation"):
        mask, _, _ = model.eval(
            images[t],
            diameter=diameter,
            channels=[0, 0],  # grayscale
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
        )
        labels[t] = np.asarray(mask)
    return labels


def cellpose_preprocessing(
    position_key: str,
    input_images: list[ProcessingInputChannel],
    z_slices: slice,
    blank_frames_path: Path = None,
    cellpose_config: CellposeConfig = None,
) -> np.ndarray:
    """
    Load data, fill empty frames, and run cellpose segmentation.

    Returns an integer label array of shape (T, Y, X) ready to pass
    directly to ``run_ultrack(..., labels=...)``.

    Parameters
    ----------
    position_key : str
        FOV key (plate, well, position).
    input_images : list of ProcessingInputChannel
        Channel loading configuration.
    z_slices : slice
        Z-planes to load.
    blank_frames_path : Path, optional
        CSV of blank frames.
    cellpose_config : CellposeConfig
        Cellpose model parameters including ``input_channel``.
    """
    fov = "/".join(position_key)

    data_dict = load_data(
        position_key=position_key,
        input_images=input_images,
        z_slices=z_slices,
    )
    data_dict = run_preprocessing_pipeline(data_dict, input_images)
    data_dict = fill_empty_frames_from_csv(fov, data_dict, blank_frames_path)

    # Get the channel to segment
    channel_name = cellpose_config.input_channel
    if channel_name not in data_dict:
        raise ValueError(
            f"Cellpose input channel '{channel_name}' not found in data. "
            f"Available: {list(data_dict.keys())}"
        )

    images = data_dict[channel_name]
    if isinstance(images, da.Array):
        images = images.compute()
    images = np.asarray(images)

    # Project Z if needed (T, Z, Y, X) -> (T, Y, X)
    if images.ndim == 4:
        click.echo(f"Projecting Z-dimension via mean: {images.shape} -> (T, Y, X)")
        images = images.mean(axis=1)

    click.echo(
        f"Running cellpose ({cellpose_config.model_type}, "
        f"diameter={cellpose_config.diameter}) on channel '{channel_name}'..."
    )
    cellpose_labels = run_cellpose_per_frame(
        images,
        model_type=cellpose_config.model_type,
        diameter=cellpose_config.diameter,
        cellprob_threshold=cellpose_config.cellprob_threshold,
        flow_threshold=cellpose_config.flow_threshold,
        gpu=cellpose_config.gpu,
        min_size=cellpose_config.min_size,
    )

    n_cells = [len(np.unique(cellpose_labels[t])) - 1 for t in range(cellpose_labels.shape[0])]
    click.echo(
        f"Cellpose cells per frame: mean={np.mean(n_cells):.1f}, "
        f"min={np.min(n_cells)}, max={np.max(n_cells)}"
    )
    return cellpose_labels


def track_one_position(
    position_key: str,
    input_images: list[ProcessingInputChannel],
    output_dirpath: Path,
    tracking_config,
    blank_frames_path: Path = None,
    z_slices: tuple[int, int] = (0, 0),
    scale: tuple[float, float, float, float, float] = (1, 1, 1, 1, 1),
    cellpose_config: CellposeConfig | None = None,
    use_focus: bool = False,
    z_total: int | None = None,
    focus_frac_below: float = 1 / 3,
    focus_frac_above: float = 2 / 3,
    focus_channel: str | None = None,
) -> None:
    """
    Run tracking on a single field of view.

    Supports two segmentation modes:
    - **foreground+contour** (default): the preprocessing pipeline must produce a binary
      foreground mask and a corresponding contour map, which are handed to ultrack.
    - **cellpose**: runs cellpose per-frame on a specified channel and passes the resulting
      instance labels directly to ultrack.

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
    cellpose_config : CellposeConfig, optional
        If provided, uses cellpose segmentation instead of foreground+contour.

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
    - In foreground+contour mode, missing "foreground"/"contour" channels raise a ValueError.
    """
    fov = "_".join(position_key)
    click.echo(f"Processing FOV: {fov.replace('_', '/')}")

    # Define path to save the tracking database and graph
    filename = output_dirpath.stem
    database_path = output_dirpath.parent / f"{filename}_config_tracking" / f"{fov}"
    os.makedirs(database_path, exist_ok=True)

    # Focus-based z-resolution (per FOV): find the in-focus plane and take a
    # window of z_total planes around it, overriding the passed z_slices.
    if use_focus:
        input_path = next((img.path for img in input_images if img.path is not None), None)
        if input_path is None:
            raise ValueError(
                "use_focus=True requires an input_images entry with a non-null path."
            )
        with open_ome_zarr(str(Path(input_path) / Path(*position_key)), mode="r") as ds:
            Z = ds.data.shape[2]
            pixel_size = ds.scale[-1]
            if focus_channel is None:
                focus_channel_index = 0
            elif focus_channel in ds.channel_names:
                focus_channel_index = ds.channel_names.index(focus_channel)
            else:
                raise ValueError(
                    f"focus_channel '{focus_channel}' not found in {ds.channel_names}"
                )
            z_slices, _ = resolve_z_slice(
                None,
                Z,
                data=ds.data,
                pixel_size=pixel_size,
                use_focus=True,
                z_total=z_total,
                frac_below=focus_frac_below,
                frac_above=focus_frac_above,
                focus_channel_index=focus_channel_index,
            )
        click.echo(f"Focus-resolved z-slice for {fov.replace('_', '/')}: {z_slices}")

    if cellpose_config is not None:
        # Cellpose mode: load data, fill blanks, run cellpose, pass labels to ultrack
        cellpose_labels = cellpose_preprocessing(
            position_key, input_images, z_slices, blank_frames_path, cellpose_config
        )

        click.echo("Tracking with cellpose labels...")
        tracking_labels, tracks_df, _ = run_ultrack(
            tracking_config=tracking_config,
            database_path=database_path,
            labels=cellpose_labels,
            sigma=cellpose_config.labels_sigma,
            scale=scale,
            overwrite=True,
        )
    else:
        # Foreground + contour mode
        foreground_mask, contour_gradient_map = data_preprocessing(
            position_key, input_images, z_slices, blank_frames_path
        )

        click.echo("Tracking with foreground + contour...")
        tracking_labels, tracks_df, _ = run_ultrack(
            tracking_config=tracking_config,
            database_path=database_path,
            detection=foreground_mask,
            edges=contour_gradient_map,
            scale=scale,
            overwrite=True,
        )

    # Save the tracks graph to a CSV file
    csv_path = output_dirpath / Path(*position_key) / f"tracks_{fov}.csv"
    os.makedirs(csv_path.parent, exist_ok=True)

    tracks_df.to_csv(csv_path, index=False)

    click.echo(f"Saved tracks to: {output_dirpath / Path(*position_key)}")

    # Save the tracking labels
    with open_ome_zarr(output_dirpath / Path(*position_key), mode="r+") as output_dataset:
        output_dataset[0][:, 0, 0] = np.asarray(tracking_labels, dtype=np.uint32)
    return tracking_labels, tracks_df


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    settings: TrackingSettings,
) -> tuple[int, int, int, int, int]:
    """Create the empty tracking output plate.

    Returns the (T, C, Z_out, Y, X) shape used for resource estimation.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as dataset:
        T, C, Z, Y, X = dataset.data.shape
        scale = dataset.scale

    _, Z_out = resolve_z_slice(settings.z_range, Z)

    if settings.mode == "2D":
        output_shape = (T, 1, 1, Y, X)
    else:
        output_shape = (T, 1, Z_out, Y, X)

    position_keys = [Path(p).parts[-3:] for p in input_position_dirpaths]

    input_plate = Path(input_position_dirpaths[0]).parents[2]
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=position_keys,
        channel_names=[f"{settings.target_channel}_labels"],
        shape=output_shape,
        chunks=None,
        scale=scale,
        version=resolve_ome_zarr_version(
            input_position_dirpaths[0], settings.output_ome_zarr_version
        ),
        dtype=np.uint32,
        metadata_sources=input_plate,
    )

    click.echo(f"Created {output_dirpath} ({len(position_keys)} positions)")

    return (T, C, Z_out, Y, X)


def track(
    input_position_dirpaths: list[str],
    output_dirpath: str,
    config_filepath: str,
    sbatch_filepath: str = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
    input_images_path: str | None = None,
):
    """Launch tracking jobs for multiple imaging positions.

    Parameters
    ----------
    input_position_dirpaths : list[str]
        Paths to input position directories (used for plate structure and metadata).
    output_dirpath : str
        Path to the output Zarr store.
    config_filepath : str
        Path to the tracking configuration YAML.
    sbatch_filepath : str, optional
        Path to a SLURM batch file to override defaults.
    cluster : str, optional
        Execution cluster: 'slurm', 'local', or 'debug'.
    monitor : bool, optional
        If True, monitor submitted jobs.
    init_only : bool, optional
        Only initialize the output store and exit.
    input_images_path : str, optional
        Override the first null input_images path in config (used by Nextflow).
    """
    from ultrack import MainConfig

    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    settings = yaml_to_model(config_filepath, TrackingSettings)

    if input_images_path is not None:
        for image in settings.input_images:
            if image.path is None:
                image.path = Path(input_images_path)
                break

    output_shape = _init_output_plate(input_position_dirpaths, output_dirpath, settings)
    T, C, Z_out, Y, X = output_shape

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=[T, C, Z_out, Y, X], ram_multiplier=16, max_num_cpus=16
    )
    mem_gb = num_cpus * gb_ram_per_cpu
    time_minutes = 60
    # Emit the JSON resources contract consumed by parse_resources in
    # nextflow/modules/common.nf (shared with every other step's --init).
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        return

    # Read shape/scale from the first input position for tracking parameters
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as dataset:
        T, C, Z, Y, X = dataset.data.shape
        scale = dataset.scale

    if settings.use_focus:
        # z-slice is resolved per-FOV inside track_one_position via focus finding
        z_slices = None
    else:
        z_slices, _ = resolve_z_slice(settings.z_range, Z)
    track_scale = scale[-2:] if settings.mode == "2D" else scale[-3:]

    tracking_cfg = settings.tracking_config
    tracking_cfg["segmentation_config"]["n_workers"] = num_cpus
    tracking_cfg["linking_config"]["n_workers"] = num_cpus
    default_config = MainConfig()
    tracking_cfg = update_model(default_config, tracking_cfg)

    position_keys = [Path(p).parts[-3:] for p in input_position_dirpaths]

    cellpose_cfg = (
        settings.cellpose_config if settings.segmentation_method == "cellpose" else None
    )

    slurm_args = {
        "slurm_job_name": "tracking",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 60,
        "slurm_partition": "preempted",
        "slurm_gpus_per_node": 1,
        "slurm_use_srun": False,
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
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
                cellpose_config=cellpose_cfg,
                use_focus=settings.use_focus,
                z_total=settings.z_total,
                focus_frac_below=settings.focus_frac_below,
                focus_frac_above=settings.focus_frac_above,
                focus_channel=settings.focus_channel,
            )
            jobs.append(job)

    job_ids = [job.job_id for job in jobs]
    slurm_out_path.mkdir(exist_ok=True)
    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a
    # DebugJob but execution only happens when .wait()/.done()/.result() is
    # called.  On the Nextflow path (--cluster debug) we run in-process so
    # Nextflow handles the fan-out and resource scheduling.
    if resolved_cluster == "debug":
        for job, pk in zip(jobs, position_keys, strict=True):
            job.wait()
            click.echo(f"Tracking complete: {'/'.join(pk)}")
        return

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("track")
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@click.option(
    "--input-images-path",
    default=None,
    type=click.Path(exists=True),
    help="Override the first null input_images path from config (used by Nextflow).",
)
def track_cli(
    input_position_dirpaths: list[str],
    output_dirpath: str,
    config_filepath: Path,
    sbatch_filepath: str = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    input_images_path: str | None = None,
):
    r"""Track objects in 2D or 3D time-lapse microscopy data using configurable preprocessing.

    \b
    Initialize the output plate only (Nextflow init step):
    >>> biahub track --init -i ./reconstruct.zarr/*/*/* -o ./track.zarr -c config.yml

    \b
    In-process run of a single position (Nextflow per-position worker):
    >>> biahub track --cluster debug -i ./reconstruct.zarr/B/3/000000 \
        -o ./track.zarr -c config.yml --input-images-path ./virtual-stain.zarr

    \b
    Full SLURM fan-out:
    >>> biahub track -i ./reconstruct.zarr/*/*/* -o ./track.zarr -c config.yml
    """
    track(
        input_position_dirpaths=input_position_dirpaths,
        output_dirpath=output_dirpath,
        config_filepath=config_filepath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
        input_images_path=input_images_path,
    )


if __name__ == "__main__":
    track_cli()
