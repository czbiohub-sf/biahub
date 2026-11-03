"""What happens to the tracking inputs before linking.

Blank frames, ultrack's foreground/contour inputs, and downsampling for trackers trained
on small nuclei.
"""

import ast

from pathlib import Path

import click
import dask.array as da
import numpy as np
import pandas as pd

from numpy.typing import ArrayLike

from biahub.cli.resolve_function import resolve_function
from biahub.settings import ProcessingInputChannel


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


def _downsample(
    labels: np.ndarray, images: np.ndarray, factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink (T, Y, X) labels and images by an integer factor.

    Labels are strided (every ``factor``-th pixel), which keeps the label ids; images are
    block-averaged. Objects smaller than the stride can vanish (they stay untracked).
    """
    T, Y, X = labels.shape
    Yc, Xc = Y // factor * factor, X // factor * factor
    small_images = (
        np.asarray(images)[:, :Yc, :Xc]
        .reshape(T, Yc // factor, factor, Xc // factor, factor)
        .mean(axis=(2, 4))
        .astype(np.float32)
    )
    return np.ascontiguousarray(labels[:, :Yc:factor, :Xc:factor]), small_images
