"""NumPy helpers shared by the step CLIs.

Cropping a volume to a slicing spec, and spotting frames that carry no data.
"""

import numpy as np


def copy_n_paste(zyx_data: np.ndarray, zyx_slicing_params: list) -> np.ndarray:
    """
    Load a zyx array and crop given a list of ZYX slices().

    Parameters
    ----------
    zyx_data : np.ndarray
        data to copy
    zyx_slicing_params : list
        list of slicing parameters for z,y,x
        Each element is a single slice object [z_slice, y_slice, x_slice]

    Returns
    -------
    np.ndarray
        crop of the input zyx_data given the slicing parameters
    """
    # Replace NaN values with zeros
    zyx_data = np.nan_to_num(zyx_data, nan=0)
    zyx_data_sliced = zyx_data[
        zyx_slicing_params[0],
        zyx_slicing_params[1],
        zyx_slicing_params[2],
    ]
    return zyx_data_sliced


def copy_n_paste_czyx(czyx_data: np.ndarray, czyx_slicing_params: list) -> np.ndarray:
    """
    Load a zyx array and crop given a list of ZYX slices().

    Parameters
    ----------
    czyx_data : np.ndarray
        data to copy
    czyx_slicing_params : list
        list of slicing parameters for z,y,x
        Each element is a single slice object [z_slice, y_slice, x_slice]

    Returns
    -------
    np.ndarray
        crop of the input czyx_data given the slicing parameters
    """
    czyx_data_sliced = czyx_data[
        :,
        czyx_slicing_params[0],
        czyx_slicing_params[1],
        czyx_slicing_params[2],
    ]
    return czyx_data_sliced


def _check_nan_n_zeros(input_array: np.ndarray) -> bool:
    """
    Check if data are all zeros or nan.

    Parameters
    ----------
    input_array : np.ndarray
        Input array (N-dimensional).

    Returns
    -------
    bool
        True if the array is entirely zeros or NaNs, False otherwise.
    """
    return np.all(np.isnan(input_array)) or np.all(input_array == 0)


def get_empty_frame_indices(input_array: np.ndarray) -> list[int]:
    """
    Get the indices of the empty frames in a 3D array.

    Parameters
    ----------
    input_array : np.ndarray
        Input array (3D).

    Returns
    -------
    List[int]
        List of Z indices that are entirely zeros or NaNs.
    """
    indices = []

    if len(input_array.shape) == 3:  # 3D array (e.g., Z, Y, X)
        for z in range(input_array.shape[0]):
            if _check_nan_n_zeros(input_array[z, :, :]):
                indices.append(z)  # Add Z index if it's empty
        return indices

    else:
        raise ValueError("Input array must be 3D.")
