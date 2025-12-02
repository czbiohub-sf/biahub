from typing import Tuple

import ants
import numpy as np
import scipy.ndimage

from biahub.registration.utils import convert_transform_to_ants

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

