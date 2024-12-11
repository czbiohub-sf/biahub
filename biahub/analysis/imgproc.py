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
        Binned array with shape (C, new_Z, new_Y, new_X)
    """
    # Calculate new dimensions after binning
    C = czyx_data.shape[0]
    new_z = czyx_data.shape[1] // binning_factor_zyx[0]
    new_y = czyx_data.shape[2] // binning_factor_zyx[1]
    new_x = czyx_data.shape[3] // binning_factor_zyx[2]

    # Process each channel separately
    output = np.zeros((C, new_z, new_y, new_x), dtype=czyx_data.dtype)

    for c in range(C):
        # Reshape to group pixels that will be binned together
        reshaped = czyx_data[c].reshape(
            new_z,
            binning_factor_zyx[0],
            new_y,
            binning_factor_zyx[1],
            new_x,
            binning_factor_zyx[2],
        )

        if mode == 'sum':
            output[c] = reshaped.sum(axis=(1, 3, 5))
        elif mode == 'mean':
            output[c] = reshaped.mean(axis=(1, 3, 5))
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sum' or 'mean'.")

    return output
