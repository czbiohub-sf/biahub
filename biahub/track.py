import ast
import json
import logging
import shutil
import tempfile

from contextlib import contextmanager
from pathlib import Path

import click
import dask.array as da
import numpy as np
import pandas as pd
import submitit
import zarr

from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_labels
from numpy.typing import ArrayLike

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
from biahub.settings import ProcessingInputChannel, TrackingSettings
from biahub.utils.cluster import (
    apply_sbatch_overrides,
    echo_resources,
    estimate_resources,
    get_submitit_cluster,
    local_executor_parameters,
)
from biahub.utils.config import update_model, yaml_to_model
from biahub.utils.ngff import label_layout, labels_exist

logger = logging.getLogger(__name__)

# Provenance key on each label image: the track settings and the resolved Ultrack config.
PROVENANCE_KEY = "biahub-track"

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


def check_input_dimensionality(output_mode: str, z_size: int) -> None:
    """Refuse an input whose Z does not match ``output_mode``.

    Track neither projects nor selects Z-planes (that is `biahub project`): 2D tracking
    needs a Z=1 plate and 3D tracking a stack.
    """
    if output_mode == "2D" and z_size > 1:
        raise ValueError(
            f"output_mode is 2D but the input is 3D (Z={z_size}). Project the plate with "
            "`biahub project` and track the projected (Z=1) plate, or set output_mode: 3D."
        )
    if output_mode == "3D" and z_size == 1:
        raise ValueError(
            "output_mode is 3D but the input has one Z-plane; set output_mode: 2D."
        )


def filter_short_siblings(
    tracks_df: pd.DataFrame, labels: ArrayLike, min_length: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Remove short daughter tracks at divisions, in tracks and labels.

    At each division where exactly one daughter lives ``min_length`` frames or fewer
    and does not divide again, that daughter is dropped (its labels set to 0) and the
    other daughter is merged into the parent. If both daughters are that short, both
    are kept. This is ultrack's ``filter_short_sibling_tracks``, which counts a
    daughter of exactly ``min_length`` frames as short.
    """
    from ultrack.tracks import filter_short_sibling_tracks

    n_before = tracks_df.track_id.nunique()
    tracks_df, labels = filter_short_sibling_tracks(tracks_df, min_length, segments=labels)
    tracks_df = tracks_df.sort_values(["track_id", "t"]).reset_index(drop=True)
    click.echo(
        f"Sibling filter (min_length={min_length}): {n_before} -> "
        f"{tracks_df.track_id.nunique()} tracks"
    )
    return tracks_df, np.asarray(labels)


def run_ultrack(
    tracking_config,
    database_path,
    geff_path: Path | None = None,
    geff_kwargs: dict | None = None,
    min_sibling_length: int | None = None,
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
    geff_path : Path, optional
        If set, the tracks are also exported there as GEFF via ``Tracker.to_geff``
        (overwriting any previous export), from the same tracks table the labels are
        painted from, so the three outputs agree after the sibling filter.
    geff_kwargs : dict, optional
        Extra keyword arguments for ``Tracker.to_geff``.
    min_sibling_length : int, optional
        If set, apply :func:`filter_short_siblings` to the tracks and labels before the
        GEFF export.
    **track_kwargs
        Keyword arguments forwarded to ``Tracker.track()``. E.g. ``detection``, ``edges``
        (foreground+contour mode), or ``labels``, ``sigma`` (label-image mode), plus ``scale``
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
    if min_sibling_length is not None:
        tracks_df, labels = filter_short_siblings(tracks_df, labels, min_sibling_length)
    if geff_path is not None:
        # `tracks_df` restricts the export to these (possibly filtered) tracks and their
        # ids, which is what the labels were painted with (royerlab/ultrack: to_geff
        # from a tracks DataFrame).
        tracker.to_geff(
            str(geff_path), overwrite=True, tracks_df=tracks_df, **(geff_kwargs or {})
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
                # Feed the arrays in lazily. Materialising them here costs the full
                # (T, Z, Y, X) stack -- tens of GB for a full timelapse -- even though
                # neither branch below needs it all at once:
                #   * array_apply slices one timepoint at a time into a zarr output;
                #   * a reduction over a dask array (e.g. np.mean) streams chunk-wise
                #     and only its much smaller result is materialised, just below.
                # numpy functions dispatch on dask via __array_function__; any function
                # that is not dask-aware falls back to np.asarray internally, which is
                # exactly the eager behaviour this replaces.
                f_data = [data_dict[name] for name in f_channel_name]
                if per_timepoint:
                    result = array_apply(*f_data, func=run_function, **f_kwargs)

                else:
                    result = run_function(*f_data, **f_kwargs)

                if isinstance(result, da.Array):
                    result = result.compute()

                data_dict[channel_name] = result
                if visualize:
                    viewer = napari.Viewer()
                    viewer.add_image(data_dict[channel_name], name=channel_name)

    return data_dict


def load_data(
    position_key: tuple[str, str, str],
    input_images: list[ProcessingInputChannel],
    visualize: bool = False,
) -> dict[str, ArrayLike]:
    """
    Load and extract specified channels from an OME-Zarr dataset for a given position.

    This function opens the OME-Zarr dataset corresponding to a specific position key,
    extracts the required channels (every Z-plane) as defined in the `input_images`
    configuration, and stores them in a dictionary for further processing.

    Parameters
    ----------
    position_key : tuple of str
        A tuple of three strings representing the hierarchical path to a specific position
        in the dataset (e.g., (plate, well, position)).
    input_images : list of ProcessingInputChannel
        List of input image channel configurations. Each item defines the path and channels to load.
    visualize : bool, optional
        If True, opens a Napari viewer and displays each loaded channel. Default is False.

    Returns
    -------
    dict of str to ArrayLike
        Dictionary mapping each channel name to its corresponding image data array.
        Each array is a Dask array of shape (T, Z, Y, X), extracted from the OME-Zarr
        store, or (T, Y, X) when the store has one Z-plane (a projected plate).

    Notes
    -----
    - Only ``input_images`` entries with a non-null ``path`` are loaded here; entries
      with ``path: null`` are virtual/computed channels produced later by
      ``run_preprocessing_pipeline`` (``track`` resolves the primary null path before
      submitting, see :func:`track`).
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
                    tzyx = dataset.data.dask_array()[
                        :, image_channel_names.index(channel_name)
                    ]
                    # A projected (Z=1) plate is a 2D series: drop the singleton axis so
                    # the processing functions see (T, Y, X).
                    data_dict[channel_name] = tzyx[:, 0] if tzyx.shape[1] == 1 else tzyx
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
    fov : str
        Field of view identifier used to filter the CSV rows.
    data_dict : dict of str to ArrayLike
        Dictionary mapping channel names to image arrays.
    blank_frame_csv_path : Path
        Path to the CSV file containing empty frame information.
        The CSV must contain columns: 'FOV' and 't'.

    Returns
    -------
    dict of str to ArrayLike
        Updated dictionary with empty frames filled in-place.
    """
    if blank_frame_csv_path:
        blank_frame_df = pd.read_csv(blank_frame_csv_path)
        empty_frames_idx = get_empty_frames_idx_from_csv(blank_frame_df, fov)
        for channel_name, channel_data in data_dict.items():
            # fill_empty_frames mutates in place, so it needs a concrete array. Only
            # materialise when there is actually something to fill, to keep channels
            # that need no filling lazy.
            if empty_frames_idx and isinstance(channel_data, da.Array):
                channel_data = channel_data.compute()
            data_dict[channel_name] = fill_empty_frames(channel_data, empty_frames_idx)

    return data_dict


def _load_and_preprocess(
    position_key: str,
    input_images: list[ProcessingInputChannel],
    blank_frames_path: Path | None,
    visualize: bool = False,
) -> dict[str, ArrayLike]:
    """Load, run the processing pipeline, and fill blank frames for one FOV.

    Used by the foreground+contour path (the labels path reads a label image).
    """
    fov = "/".join(position_key)
    data_dict = load_data(
        position_key=position_key, input_images=input_images, visualize=visualize
    )
    data_dict = run_preprocessing_pipeline(data_dict, input_images, visualize=visualize)
    return fill_empty_frames_from_csv(fov, data_dict, blank_frames_path)


def detect_foreground_segmentation(
    data_dict: dict[str, ArrayLike],
) -> tuple[ArrayLike, ArrayLike]:
    """Pull the foreground mask and contour map out of a preprocessed ``data_dict``.

    Accepts either separate ``foreground`` and ``contour`` channels or a single
    ``foreground_contour`` pair. Raises if neither is present.
    """
    channel_names = data_dict.keys()
    if "foreground" in channel_names and "contour" in channel_names:
        return data_dict["foreground"], data_dict["contour"]
    if "foreground_contour" in channel_names:
        return data_dict["foreground_contour"]
    raise ValueError("Foreground and contour channels are required for tracking.")


def load_labels(
    position_dirpath: Path,
    label_name: str,
    output_mode: str,
    blank_frames_path: Path | None = None,
) -> np.ndarray:
    """Read the instance labels ``labels/<label_name>`` of one position.

    The label image is (T, 1, Z, Y, X), the image's axes with one channel. Returns
    (T, Y, X) labels for 2D tracking (the single plane of a projected plate) and
    (T, Z, Y, X) for 3D. Frames listed as blank for this FOV in ``blank_frames_path``
    are filled with the nearest non-blank frame's labels, as the images used to be
    before segmentation moved out of track, so a dropped timepoint does not break
    every track.
    """
    with open_ome_zarr(str(position_dirpath), mode="r") as position:
        if label_name not in position.label_names():
            raise ValueError(
                f"{position_dirpath} has no label image {label_name!r} (labels: "
                f"{position.label_names()}); segment the plate with `biahub segment` first."
            )
        labels = np.asarray(position.get_label(label_name)["0"][:, 0])  # (T, Z, Y, X)
    if output_mode == "2D":
        labels = labels[:, 0]
    if blank_frames_path:
        fov = "/".join(Path(position_dirpath).parts[-3:])
        empty_frames_idx = get_empty_frames_idx_from_csv(pd.read_csv(blank_frames_path), fov)
        if empty_frames_idx:
            labels = fill_empty_frames(labels, empty_frames_idx)
    n_cells = [len(np.unique(frame)) - 1 for frame in labels]
    click.echo(
        f"Labels {label_name!r}: cells per frame mean={np.mean(n_cells):.1f}, "
        f"min={np.min(n_cells)}, max={np.max(n_cells)}"
    )
    return labels


def write_tracking_labels(label_array, labels: ArrayLike, output_mode: str) -> None:
    """Write tracked labels into the (T, 1, Z, Y, X) label array of the plate track read.

    The label image has the image's axes with one channel and the plate's Z: one
    plane for a projected plate, where 2D tracking writes its (T, Y, X) labels; the
    full stack for 3D tracking, whose (T, Z, Y, X) labels fill it.
    """
    labels = np.asarray(labels, dtype=np.uint32)
    if output_mode == "2D":
        # A single-plane series that kept its Z axis.
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 3:
            raise ValueError(
                f"output_mode='2D' expects (T, Y, X) labels but tracking produced "
                f"shape {labels.shape}."
            )
        label_array[:, 0, 0] = labels
        return
    if labels.ndim != 4 or labels.shape[1] != label_array.shape[2]:
        raise ValueError(
            f"output_mode='3D' expects (T, {label_array.shape[2]}, Y, X) labels but "
            f"tracking produced shape {labels.shape}."
        )
    label_array[:, 0] = labels


@contextmanager
def _database_dirpath(output_dirpath: Path | None, fov: str):
    """Yield the Ultrack working directory (database and config) for one FOV.

    ``{output_dirpath}/{fov}`` when ``output_dirpath`` is given, and kept. Otherwise a
    fresh directory in this worker's ``$TMPDIR``, deleted on exit: one per FOV,
    because ``$TMPDIR`` can be shared by all of a user's jobs on a node.
    """
    if output_dirpath is not None:
        dirpath = Path(output_dirpath) / fov
        dirpath.mkdir(parents=True, exist_ok=True)
        yield dirpath
        return
    dirpath = Path(tempfile.mkdtemp(prefix=f"biahub-track-{fov}-"))
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath, ignore_errors=True)


def _resolved_ultrack_config(tracking_config) -> dict:
    """Return the full Ultrack config, defaults included, as JSON-safe provenance."""
    config = json.loads(json.dumps(tracking_config.dict(by_alias=True), default=str))
    # The working directory is the per-FOV database, often a deleted temp directory.
    config.get("data", {}).pop("working_dir", None)
    return config


def track_one_position(
    position_key: tuple[str, str, str],
    position_dirpath: Path,
    input_images: list[ProcessingInputChannel],
    output_dirpath: Path | None,
    tracking_config,
    label_name: str,
    blank_frames_path: Path | None = None,
    scale: tuple[float, ...] = (1, 1, 1, 1, 1),
    input_label: str | None = None,
    labels_sigma: float = 5.0,
    output_mode: str = "2D",
    min_sibling_length: int | None = None,
    tracks_stem: str = "tracks",
) -> None:
    """
    Run tracking on a single field of view.

    Supports two segmentation inputs:
    - **labels** (``input_label`` set): the instance labels of ``labels/<input_label>``
      in the position (written by `biahub segment`) are handed to ultrack, which builds
      its own segmentation hypotheses from their contours (smoothed by ``labels_sigma``).
    - **foreground+contour** (legacy): the preprocessing pipeline must produce a binary
      foreground mask and a corresponding contour map, which are handed to ultrack.

    Parameters
    ----------
    position_key : tuple of str
        (row, column, fov) of the field of view.
    position_dirpath : Path
        The ``-i`` position the outputs are written into. ``track --init`` has already
        created its empty ``labels/<label_name>`` image.
    input_images : list of ProcessingInputChannel
        Configuration describing which channels to load and how to preprocess them
        (foreground+contour); may be empty with ``input_label``.
    output_dirpath : Path or None
        Where to keep the Ultrack database and config for this FOV, in
        ``{output_dirpath}/{row}_{col}_{fov}/``. If None they live in a scratch
        directory in this worker's ``$TMPDIR`` and are deleted after export (see
        :func:`_database_dirpath`).
    tracking_config : MainConfig
        Ultrack configuration containing segmentation, linking, and optimization settings.
    label_name : str
        Name of the label image under ``{position_dirpath}/labels/``.
    blank_frames_path : Path, optional
        Path to CSV file indicating empty frames for the current FOV. If None, blank frame
        filling is skipped.
    scale : tuple of float, optional
        Physical scale used for tracking, in (Z, Y, X) or (Y, X) order (already trimmed
        by the caller to match ``output_mode``). Default is (1, 1, 1, 1, 1).
    input_label : str, optional
        Name of the label image to track, ``labels/<input_label>`` in the position. If
        None, the foreground+contour path is used.
    labels_sigma : float, optional
        Smoothing of the labels' contours before ultrack re-segments them (ultrack's
        ``labels_to_contours`` sigma). Default is 5.0.
    output_mode : str, optional
        "2D" tracks a Z=1 input and writes (T, Y, X) labels into its one plane; "3D"
        tracks the stack and writes (T, Z, Y, X) labels. Track does not project (see
        `biahub project`). Default is "2D".
    min_sibling_length : int, optional
        If set, remove daughter tracks of this many frames or fewer before writing
        (see :func:`filter_short_siblings`); the labels, the GEFF and the CSV all
        come from the filtered tracks. Default None keeps every division.
    tracks_stem : str, optional
        Stem of the GEFF and CSV file names. Default ``"tracks"``.

    Returns
    -------
    None
        Outputs are saved into ``position_dirpath``:
        - ``labels/<label_name>``: the tracked labels, painted with ``track_id``
        - ``<tracks_stem>.geff``: the track graph, linked to the labels by ``seg_id``
        - ``<tracks_stem>_<row>_<col>_<fov>.csv``: the same tracks as a CSV, kept for now
        ``tracks_stem`` is ``tracks`` unless the config names the tracked label itself
        (``output_label``), when it is ``tracks_<output_label>``.

    Notes
    -----
    - In foreground+contour mode, missing "foreground"/"contour" channels raise a ValueError.
    """
    position_dirpath = Path(position_dirpath)
    fov = "_".join(position_key)
    click.echo(f"Processing FOV: {fov.replace('_', '/')}")

    with open_ome_zarr(position_dirpath, mode="r") as position:
        time_scale = position.scale[0]
        units = {axis.name.lower(): getattr(axis, "unit", None) for axis in position.axes}

    with _database_dirpath(output_dirpath, fov) as database_path:
        geff_kwargs = {
            "zarr_format": 3,
            "include_masks": False,
            "include_overlaps": False,
            "segmentation_path": f"../labels/{label_name}",
            "scale": tuple(scale),
            "spatial_unit": units.get("x"),
            "time_scale": time_scale if units.get("t") else None,
            "time_unit": units.get("t"),
        }
        if input_label is not None:
            labels = load_labels(position_dirpath, input_label, output_mode, blank_frames_path)
            detection_kwargs = {"labels": labels, "sigma": labels_sigma}
            click.echo(f"Tracking the labels of labels/{input_label}...")
        else:
            # Load, run the processing pipeline, and fill blank frames.
            data_dict = _load_and_preprocess(position_key, input_images, blank_frames_path)
            foreground_mask, contour_gradient_map = detect_foreground_segmentation(data_dict)
            detection_kwargs = {"detection": foreground_mask, "edges": contour_gradient_map}
            click.echo("Tracking with foreground + contour...")

        tracking_labels, tracks_df, _ = run_ultrack(
            tracking_config=tracking_config,
            database_path=database_path,
            geff_path=position_dirpath / f"{tracks_stem}.geff",
            geff_kwargs=geff_kwargs,
            min_sibling_length=min_sibling_length,
            scale=scale,
            overwrite=True,
            **detection_kwargs,
        )

        # Kept alongside the GEFF for downstream readers that still expect a CSV.
        tracks_df.to_csv(position_dirpath / f"{tracks_stem}_{fov}.csv", index=False)

        with open_ome_zarr(position_dirpath, mode="r+") as position:
            label_image = position.get_label(label_name)
            write_tracking_labels(label_image["0"], tracking_labels, output_mode)

    # Record the full Ultrack config, so provenance does not depend on the (usually
    # deleted) database directory.
    label_group = zarr.open_group(position_dirpath / "labels" / label_name, mode="r+")
    provenance = dict(label_group.attrs.get(PROVENANCE_KEY, {}))
    provenance["ultrack_config"] = _resolved_ultrack_config(tracking_config)
    label_group.attrs[PROVENANCE_KEY] = provenance

    click.echo(f"Saved labels and tracks to: {position_dirpath}")


def _init_labels(
    input_position_dirpaths: list[Path],
    settings: TrackingSettings,
) -> tuple[int, int, int, int, int]:
    """Create the empty ``labels/<tracked_label>`` image in every input position.

    The input's dimensionality must match ``settings.output_mode`` (see
    :func:`check_input_dimensionality`), and with ``settings.labels`` every position
    must already have that label image (``biahub segment --init`` creates it, so a
    pipeline missing its segment step fails here). The tracked labels are (T, Z, Y, X)
    with the plate's Z: one plane for a projected plate, the stack for 3D tracking.
    Positions that already have the output label are left untouched, and the track
    settings are recorded on each label image.

    Returns the input (T, C, Z, Y, X) shape used for resource estimation.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as dataset:
        Z = dataset.data.shape[2]
    check_input_dimensionality(settings.output_mode, Z)

    if settings.labels is not None:
        missing = [
            str(p) for p in input_position_dirpaths if not labels_exist([p], settings.labels)
        ]
        if missing:
            raise ValueError(
                f"labels: {settings.labels!r} names a label image that "
                f"{len(missing)} position(s) do not have (first: {missing[0]}); run "
                "`biahub segment` on the plate first (its model name is the label name)."
            )

    shape, layout = label_layout(input_position_dirpaths[0])
    create_empty_labels(
        Path(input_position_dirpaths[0]).parents[2],
        [Path(p).parts[-3:] for p in input_position_dirpaths],
        [settings.tracked_label],
        # The settings as written; the worker adds the resolved Ultrack config under
        # the same key once it has tracked (track_one_position).
        label_metadata={PROVENANCE_KEY: settings.model_dump(mode="json")},
        **layout,
    )
    click.echo(
        f"Created labels/{settings.tracked_label} in "
        f"{Path(input_position_dirpaths[0]).parents[2]} "
        f"({len(input_position_dirpaths)} positions)"
    )
    return shape


def track(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path | None = None,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
    input_images_path: str | None = None,
):
    """Launch tracking jobs for multiple imaging positions.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to input position directories (used for plate structure and metadata).
    config_filepath : Path
        Path to the tracking configuration YAML.
    output_dirpath : Path, optional
        Where to keep the per-FOV Ultrack databases (``{output_dirpath}/{row}_{col}_{fov}/``)
        and the SLURM logs (``{output_dirpath}/slurm_output``). If None, each worker
        builds its database in its own ``$TMPDIR`` and deletes it after export, and the
        SLURM logs go to ``./slurm_output``. The labels and tracks themselves are
        always written into the ``-i`` positions.
    sbatch_filepath : str, optional
        Path to a SLURM batch file to override defaults.
    cluster : str, optional
        Execution cluster: 'slurm', 'local', or 'debug'.
    monitor : bool, optional
        If True, monitor submitted jobs.
    init_only : bool, optional
        Only create the empty label images in the ``-i`` positions and exit.
    input_images_path : str, optional
        Explicit pixel-data source filling the first null ``input_images`` path
        (Nextflow). If unset, that null path falls back to the ``-i`` plate (see Notes).

    Notes
    -----
    Outputs go into each ``-i`` position: ``labels/tracks_<labels>`` (OME-NGFF label
    image; ``labels/<target_channel>`` on the foreground+contour path),
    ``tracks.geff`` and ``tracks_<row>_<col>_<fov>.csv``. The ``-i`` plate is the plate
    track reads, so 2D tracking writes into the projected (Z=1) plate, next to the
    ``labels/<labels>`` image `biahub segment` wrote there.

    Two input sources: ``-i`` (``input_position_dirpaths``) gives the plate structure,
    shape/scale and the label image, while ``input_images[].path`` gives pixel data
    (foreground+contour). The first null ``path`` is the primary source
    (``--input-images-path`` if given, else the ``-i`` plate); later null entries are
    virtual/computed channels.
    """
    from ultrack import MainConfig

    if output_dirpath is not None:
        output_dirpath = Path(output_dirpath)
        slurm_out_path = output_dirpath / "slurm_output"
    else:
        # Not $TMPDIR: submitit's folder must be visible from the compute nodes.
        slurm_out_path = Path("slurm_output").absolute()
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    settings = yaml_to_model(config_filepath, TrackingSettings)

    # Resolve the primary (first null) input_images path: an explicit
    # --input-images-path wins, else fall back to the -i input plate root.
    input_plate = Path(input_position_dirpaths[0]).parents[2]
    primary_path = Path(input_images_path) if input_images_path is not None else input_plate
    for image in settings.input_images:
        if image.path is None:
            image.path = primary_path
            break

    # Workers skip the scaffold --init already made: they all share the -i plate.
    if init_only or not labels_exist(input_position_dirpaths, settings.tracked_label):
        _init_labels(input_position_dirpaths, settings)
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as dataset:
        T, C, Z, Y, X = dataset.data.shape
        scale = dataset.scale
    check_input_dimensionality(settings.output_mode, Z)

    _, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16
    )
    time_minutes = 60
    mem_gb = num_cpus * gb_ram_per_cpu
    # Emit the JSON resources contract consumed by parse_resources in
    # nextflow/modules/common.nf (shared with every other step's --init).
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        click.echo(f"Initialized tracking ({len(input_position_dirpaths)} positions)")
        return

    track_scale = scale[-2:] if settings.output_mode == "2D" else scale[-3:]

    tracking_cfg = settings.tracking_config
    tracking_cfg["segmentation_config"]["n_workers"] = num_cpus
    tracking_cfg["linking_config"]["n_workers"] = num_cpus
    default_config = MainConfig()
    tracking_cfg = update_model(default_config, tracking_cfg)

    position_keys = [Path(p).parts[-3:] for p in input_position_dirpaths]

    slurm_args = {
        "slurm_job_name": "track",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": time_minutes,
        # Ultrack is CPU-bound (segmentation is `biahub segment`'s, on the GPU); it
        # cannot resume, but a position tracks in minutes, so a preemption costs little.
        # Override via --sbatch-filepath if a different partition is needed.
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }

    if sbatch_filepath:
        slurm_args = apply_sbatch_overrides(slurm_args, sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    if resolved_cluster == "local":
        # Explicit --sbatch-filepath values (e.g. '#LOCAL --timeout-min') win over
        # the computed local executor parameters.
        slurm_args = {**local_executor_parameters(time_minutes, gpu=False), **slurm_args}
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for position_key, position_dirpath in zip(
            position_keys, input_position_dirpaths, strict=True
        ):
            job = executor.submit(
                track_one_position,
                position_key=position_key,
                position_dirpath=Path(position_dirpath),
                output_dirpath=output_dirpath,
                tracking_config=tracking_cfg,
                label_name=settings.tracked_label,
                input_images=settings.input_images,
                blank_frames_path=settings.blank_frames_path,
                scale=track_scale,
                input_label=settings.labels,
                labels_sigma=settings.labels_sigma,
                output_mode=settings.output_mode,
                min_sibling_length=settings.min_sibling_length,
                tracks_stem=settings.tracks_stem,
            )
            jobs.append(job)

    job_ids = [job.job_id for job in jobs]
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
@config_filepath()
@output_dirpath(
    required=False,
    help="Directory to keep the per-FOV Ultrack databases and the SLURM logs in. "
    "If omitted, each worker builds its database in its own $TMPDIR and deletes it "
    "after export, and the SLURM logs go to ./slurm_output.",
)
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@click.option(
    "--input-images-path",
    default=None,
    type=click.Path(exists=True),
    help="Pixel-data source filling the first null input_images path (used by "
    "Nextflow). If omitted, that null path falls back to the -i input plate.",
)
def track_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path | None = None,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    input_images_path: str | None = None,
):
    """Track the instance labels of a label image (`biahub segment`) with ultrack.

    Reads labels/<labels> from each -i position and writes into it: the tracked labels
    as an OME-NGFF label image (labels/tracks_<labels>), the tracks as tracks.geff,
    and a tracks CSV. 2D tracking reads (and writes into) the projected Z=1 plate
    `biahub project` makes; the Ultrack databases are temporary unless -o names a
    directory to keep them in. Track does not segment or project.

    \b
    Full SLURM fan-out:
    >>> biahub track -i ./assemble_proj.zarr/*/*/* -c config.yml

    \b
    Keep the Ultrack databases (e.g. to re-solve with other weights):
    >>> biahub track -i ./assemble_proj.zarr/*/*/* -o ./track -c config.yml

    \b
    Create the empty label images only (Nextflow init step):
    >>> biahub track --init -i ./assemble_proj.zarr/*/*/* -c config.yml

    \b
    In-process run of a single position (Nextflow per-position worker):
    >>> biahub track --cluster debug -i ./assemble_proj.zarr/B/3/000000 -c config.yml
    """  # noqa: D301
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
