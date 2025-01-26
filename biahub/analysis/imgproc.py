from typing import Literal, Sequence

import click
import numpy as np

from biahub.analysis.AnalysisSettings import ProcessingFunctions


def process_czyx(
    czyx_data: np.ndarray,
    processing_functions: list[ProcessingFunctions],
) -> np.ndarray:
    """
    Process a CZYX image using processing functions

    Parameters
    ----------
    czyx_data : np.ndarray
        A CZYX image to process
    processing_functions : list[ProcessingFunctions]
        A list of processing functions to apply with their configurations

    Returns
    -------
    np.ndarray
        A processed CZYX image
    """
    # Apply processing functions
    for proc in processing_functions:

        func = proc.function  # ImportString automatically resolves the function
        kwargs = proc.kwargs
        c_idx = proc.channel

        click.echo(f'Processing with {func.__name__} with kwargs {kwargs} to channel {c_idx}')
        # TODO should we ha
        czyx_data = func(czyx_data, **kwargs)

    return czyx_data


def binning_czyx(
    czyx_data: np.ndarray,
    binning_factor_zyx: Sequence[int] = [1, 2, 2],
    mode: Literal['sum', 'mean'] = 'sum',
) -> np.ndarray:
    """
    Binning via summing or averaging pixels within bin windows

    Parameters
    ----------
    czyx_data : np.ndarray
        Input array to bin in CZYX format
    binning_factor_zyx : Sequence[int]
        Binning factor in each dimension (Z, Y, X). Can be list or tuple.
    mode : str
        'sum' for sum binning or 'mean' for mean binning

    Returns
    -------
    np.ndarray
        Binned array with shape (C, new_Z, new_Y, new_X) with same dtype as input.
        For sum mode, values are normalized to span [0, dtype.max] for integer types
        or [0, 65535] for float types.
        For mean mode, values are averaged within bins.
    """
    # Calculate new dimensions after binning
    C = czyx_data.shape[0]
    new_z = czyx_data.shape[1] // binning_factor_zyx[0]
    new_y = czyx_data.shape[2] // binning_factor_zyx[1]
    new_x = czyx_data.shape[3] // binning_factor_zyx[2]

    # Use float32 for intermediate calculations to avoid overflow
    output = np.zeros((C, new_z, new_y, new_x), dtype=np.float32)

    for c in range(C):
        # Reshape to group pixels that will be binned together
        reshaped = (
            czyx_data[c]
            .astype(np.float32)
            .reshape(
                new_z,
                binning_factor_zyx[0],
                new_y,
                binning_factor_zyx[1],
                new_x,
                binning_factor_zyx[2],
            )
        )

        if mode == 'sum':
            output[c] = reshaped.sum(axis=(1, 3, 5))
            # Normalize sum to [0, max_val] where max_val is dtype dependent
            if output[c].max() > 0:  # Avoid division by zero
                if np.issubdtype(czyx_data.dtype, np.integer):
                    max_val = np.iinfo(czyx_data.dtype).max
                else:
                    max_val = np.iinfo(np.uint16).max  # Normalize floats to uint16 range
                output[c] = (
                    (output[c] - output[c].min())
                    * max_val
                    / (output[c].max() - output[c].min())
                )
        elif mode == 'mean':
            output[c] = reshaped.mean(axis=(1, 3, 5))
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sum' or 'mean'.")

    # For mean mode and integer dtypes, scale to dtype range
    if mode == 'mean' and np.issubdtype(czyx_data.dtype, np.integer):
        dtype_info = np.iinfo(czyx_data.dtype)
        output = output * dtype_info.max / output.max()

    # Convert back to original dtype
    return output.astype(czyx_data.dtype)
