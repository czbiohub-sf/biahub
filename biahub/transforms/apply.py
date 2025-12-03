from typing import Tuple

import ants
import click
import numpy as np
import scipy.ndimage

from biahub.registration.utils import convert_transform_to_ants


def apply_stabilization_transform(
    zyx_data: np.ndarray,
    list_of_shifts: list[np.ndarray],
    t_idx: int,
    output_shape: tuple[int, int, int] = None,
):
    """
    Apply stabilization transformations to 3D or 4D volumetric data.

    This function applies a time-indexed stabilization transformation to a single 3D (Z, Y, X) volume
    or a 4D (C, Z, Y, X) volume using a precomputed list of transformations.

    Parameters:
    - zyx_data (np.ndarray): Input 3D (Z, Y, X) or 4D (C, Z, Y, X) volumetric data.
    - list_of_shifts (list[np.ndarray]): List of transformation matrices (one per time index).
    - t_idx (int): Time index corresponding to the transformation to apply.
    - output_shape (tuple[int, int, int], optional): Desired shape of the output stabilized volume.
                                                     If None, the shape of `zyx_data` is used.
                                                     Defaults to None.

    Returns:
    - np.ndarray: The stabilized 3D (Z, Y, X) or 4D (C, Z, Y, X) volume.

    Notes:
    - If `zyx_data` is 4D, the function recursively applies stabilization to each channel (C).
    - Uses ANTsPy for applying the transformation to the input data.
    - Handles `NaN` values in the input by replacing them with 0 before applying the transformation.
    - Echoes the transformation matrix for debugging purposes when verbose logging is enabled.
    """

    if output_shape is None:
        output_shape = zyx_data.shape[-3:]

    # Get the transformation matrix for the current time index
    tx_shifts = convert_transform_to_ants(list_of_shifts[t_idx])

    if zyx_data.ndim == 4:
        stabilized_czyx = np.zeros((zyx_data.shape[0],) + output_shape, dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            stabilized_czyx[c] = apply_stabilization_transform(
                zyx_data[c], list_of_shifts, t_idx, output_shape
            )
        return stabilized_czyx
    else:
        click.echo(f'shifting matrix with t_idx:{t_idx} \n{list_of_shifts[t_idx]}')
        target_zyx_ants = ants.from_numpy(np.zeros((output_shape), dtype=np.float32))

        zyx_data = np.nan_to_num(zyx_data, nan=0)
        zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
        stabilized_zyx = tx_shifts.apply_to_image(
            zyx_data_ants, reference=target_zyx_ants
        ).numpy()

    return stabilized_zyx


def apply_affine_transform(
    zyx_data: np.ndarray,
    matrix: np.ndarray,
    output_shape_zyx: Tuple,
    method="ants",
    interpolation: str = "linear",
    crop_output_slicing: bool = None,
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    zyx_data : np.ndarray
        3D input array to be transformed
    matrix : np.ndarray
        3D Homogenous transformation matrix
    output_shape_zyx : Tuple
        output target zyx shape
    method : str, optional
        method to use for transformation, by default 'ants'
    interpolation: str, optional
        interpolation mode for ants, by default "linear"
    crop_output : bool, optional
        crop the output to the largest interior rectangle, by default False

    Returns
    -------
    np.ndarray
        registered zyx data
    """

    Z, Y, X = output_shape_zyx
    if crop_output_slicing is not None:
        Z_slice, Y_slice, X_slice = crop_output_slicing
        Z = Z_slice.stop - Z_slice.start
        Y = Y_slice.stop - Y_slice.start
        X = X_slice.stop - X_slice.start

    # TODO: based on the signature of this function, it should not be called on 4D array
    if zyx_data.ndim == 4:
        registered_czyx = np.zeros((zyx_data.shape[0], Z, Y, X), dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            registered_czyx[c] = apply_affine_transform(
                zyx_data[c],
                matrix,
                output_shape_zyx,
                method=method,
                interpolation=interpolation,
                crop_output_slicing=crop_output_slicing,
            )
        return registered_czyx
    else:
        # Convert nans to 0
        zyx_data = np.nan_to_num(zyx_data, nan=0)

        # NOTE: default set to ANTS apply_affine method until we decide we get a benefit from using cupy
        # The ants method on CPU is 10x faster than scipy on CPU. Cupy method has not been bencharked vs ANTs

        if method == "ants":
            # The output has to be a ANTImage Object
            empty_target_array = np.zeros((output_shape_zyx), dtype=np.float32)
            target_zyx_ants = ants.from_numpy(empty_target_array)

            T_ants = convert_transform_to_ants(matrix)

            zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
            registered_zyx = T_ants.apply_to_image(
                zyx_data_ants, reference=target_zyx_ants, interpolation=interpolation
            ).numpy()

        elif method == "scipy":
            registered_zyx = scipy.ndimage.affine_transform(zyx_data, matrix, output_shape_zyx)

        else:
            raise ValueError(f"Unknown method {method}")

        # Crop the output to the largest interior rectangle
        if crop_output_slicing is not None:
            registered_zyx = registered_zyx[Z_slice, Y_slice, X_slice]

    return registered_zyx
