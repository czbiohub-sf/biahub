"""
Registration utility functions.

Helper functions shared across registration methods (beads, ANTs, phase correlation).
Includes:
- Transform validation and interpolation across timepoints.
- Approximate transform computation from voxel sizes and rotation/flip parameters.
- Transform I/O (save/load from disk, convert between numpy and ANTs formats).
- Volume utilities: LIR cropping, affine application, padding, and shape matching.

Key conventions
---------------
- Coordinates and shapes are in ZYX order for 3D data.
- Transforms are 4x4 homogeneous matrices.
"""

import ants
import click
import largestinteriorrectangle as lir
import numpy as np
import scipy

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

from biahub.core.transform import Transform

# TODO: see if at some point these globals should be hidden or exposed.
NA_DETECTION_SOURCE = 1.35
NA_DETECTION_TARGET = 1.35
WAVELENGTH_EMISSION_SOURCE_CHANNEL = 0.45  # in um
WAVELENGTH_EMISSION_TARGET_CHANNEL = 0.6  # in um
FOCUS_SLICE_ROI_WIDTH = 150  # size of central ROI used to find focal slice


def get_aprox_transform(
    mov_shape: tuple[int, int, int],
    ref_shape: tuple[int, int, int],
    pre_affine_90degree_rotation: int = -1,
    pre_affine_fliplr: bool = False,
    verbose: bool = False,
    ref_voxel_size: tuple[float, float, float] = (0.174, 0.1494, 0.1494),
    mov_voxel_size: tuple[float, float, float] = (0.174, 0.1494, 0.1494),
) -> Transform:

    mov_Z, mov_Y, mov_X = mov_shape
    ref_Z, ref_Y, ref_X = ref_shape

    # Calculate scaling factors for displaying data     source_channel_voxel_size, target_channel_voxel_size
    scaling_factor_z = mov_voxel_size[-3] / ref_voxel_size[-3]
    scaling_factor_yx = mov_voxel_size[-1] / ref_voxel_size[-1]
    click.echo(
        f"Z scaling factor: {scaling_factor_z:.3f}; XY scaling factor: {scaling_factor_yx:.3f}\n"
    )

    scaling_affine = get_3D_rescaling_matrix(
        (ref_Z, ref_Y, ref_X),
        (scaling_factor_z, scaling_factor_yx, scaling_factor_yx),
        (ref_Z, ref_Y, ref_X),
    )
    rotate90_affine = get_3D_rotation_matrix(
        (mov_Z, mov_Y, mov_X),
        90 * pre_affine_90degree_rotation,
        (ref_Z, ref_Y, ref_X),
    )

    # Apply flip transformation if requested (flip happens first)
    if pre_affine_fliplr:
        fliplr_affine = get_3D_fliplr_matrix(
            (mov_Z, mov_Y, mov_X),
            (ref_Z, ref_Y, ref_X),
        )
    else:
        fliplr_affine = np.eye(4)

    compound_affine = np.linalg.inv(scaling_affine @ rotate90_affine @ fliplr_affine)

    return Transform(matrix=compound_affine)


def convert_transform_to_ants(T_numpy: np.ndarray):
    """Homogeneous 3D transformation matrix from numpy to ants.

    Parameters
    ----------
    numpy_transform :4x4 homogenous matrix

    Returns
    -------
    Ants transformation matrix object
    """
    assert T_numpy.shape == (4, 4)

    T_ants_style = T_numpy[:, :-1].ravel()
    T_ants_style[-3:] = T_numpy[:3, -1]
    T_ants = ants.new_ants_transform(
        transform_type="AffineTransform",
    )
    T_ants.set_parameters(T_ants_style)

    return T_ants


def convert_transform_to_numpy(T_ants):
    """
    Convert the ants transformation matrix to numpy 3D homogenous transform.

    Modified from Jordao's dexp code

    Parameters
    ----------
    T_ants : Ants transfromation matrix object

    Returns
    -------
    np.array
        Converted Ants to numpy array

    """
    T_numpy = T_ants.parameters.reshape((3, 4), order="F")
    T_numpy[:, :3] = T_numpy[:, :3].transpose()
    T_numpy = np.vstack((T_numpy, np.array([0, 0, 0, 1])))

    # Reference:
    # https://sourceforge.net/p/advants/discussion/840261/thread/9fbbaab7/
    # https://github.com/netstim/leaddbs/blob/a2bb3e663cf7fceb2067ac887866124be54aca7d/helpers/ea_antsmat2mat.m
    # T = original translation offset from A
    # T = T + (I - A) @ centering

    T_numpy[:3, -1] += (np.eye(3) - T_numpy[:3, :3]) @ T_ants.fixed_parameters

    return T_numpy


def find_lir(registered_zyx: np.ndarray, plot: bool = False) -> tuple:
    registered_zyx = np.asarray(registered_zyx, dtype=bool)

    # Find the lir in YX at Z//2
    registered_yx = registered_zyx[registered_zyx.shape[0] // 2].copy()
    coords_yx = lir.lir(registered_yx)
    coords_yx = list(map(int, coords_yx))

    x, y, width, height = coords_yx
    x_start, x_stop = x, x + width
    y_start, y_stop = y, y + height
    x_slice = slice(x_start, x_stop)
    y_slice = slice(y_start, y_stop)

    # Iterate over ZX and ZY slices to find optimal Z cropping params
    _coords = []
    for _x in (x_start, x_start + (x_stop - x_start) // 2, x_stop - 1):
        registered_zy = registered_zyx[:, y_slice, _x].copy()
        coords_zy = lir.lir(registered_zy)
        _, z, _, depth = coords_zy
        z_start, z_stop = z, z + depth
        _coords.append((z_start, z_stop))
    for _y in (y_start, y_start + (y_stop - y_start) // 2, y_stop - 1):
        registered_zx = registered_zyx[:, _y, x_slice].copy()
        coords_zx = lir.lir(registered_zx)
        _, z, _, depth = coords_zx
        z_start, z_stop = z, z + depth
        _coords.append((z_start, z_stop))

    _coords = np.asarray(_coords)
    z_start = int(_coords.max(axis=0)[0])
    z_stop = int(_coords.min(axis=0)[1])
    z_slice = slice(z_start, z_stop)

    if plot:
        xy_corners = ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
        rectangle_yx = plt.Polygon(
            xy_corners,
            closed=True,
            fill=None,
            edgecolor="r",
        )
        # Add the rectangle to the plot
        _, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(registered_yx)
        ax[0].add_patch(rectangle_yx)

        zx_corners = ((x, z), (x + width, z), (x + width, z + depth), (x, z + depth))
        rectangle_zx = plt.Polygon(
            zx_corners,
            closed=True,
            fill=None,
            edgecolor="r",
        )
        ax[1].imshow(registered_zx)
        ax[1].add_patch(rectangle_zx)
        plt.savefig("./lir.png")

    return (z_slice, y_slice, x_slice)


def find_overlapping_volume(
    input_zyx_shape: tuple,
    target_zyx_shape: tuple,
    transformation_matrix: np.ndarray,
    method: str = "LIR",
    plot: bool = False,
) -> tuple:
    """
    Find the overlapping rectangular volume after registration of two 3D datasets.

    Parameters
    ----------
    input_zyx_shape : Tuple
        shape of input array
    target_zyx_shape : Tuple
        shape of target array
    transformation_matrix : np.ndarray
        affine transformation matrix
    method : str, optional
        method of finding the overlapping volume, by default 'LIR'

    Returns
    -------
    Tuple
        ZYX slices of the overlapping volume after registration

    """
    # Make dummy volumes
    moving_volume = np.ones(tuple(input_zyx_shape), dtype=np.float32)
    fixed_volume = np.ones(tuple(target_zyx_shape), dtype=np.float32)

    # Convert to ants objects
    fixed_volume_ants = ants.from_numpy(fixed_volume)
    moving_volume_ants = ants.from_numpy(moving_volume)

    tform_ants = convert_transform_to_ants(transformation_matrix)

    # Now apply the transform using this grid
    registered_volume = tform_ants.apply_to_image(
        moving_volume_ants, reference=fixed_volume_ants
    ).numpy()
    if method == "LIR":
        click.echo("Starting Largest interior rectangle (LIR) search")
        mask = (registered_volume > 0) & (fixed_volume > 0)
        z_slice, y_slice, x_slice = find_lir(mask, plot=plot)

    else:
        raise ValueError(f"Unknown method {method}")

    return (z_slice, y_slice, x_slice)


def rescale_voxel_size(affine_matrix, input_scale):
    return np.linalg.norm(affine_matrix, axis=1) * input_scale


def get_3D_rescaling_matrix(start_shape_zyx, scaling_factor_zyx=(1, 1, 1), end_shape_zyx=None):
    center_Y_start, center_X_start = np.array(start_shape_zyx)[-2:] / 2
    if end_shape_zyx is None:
        center_Y_end, center_X_end = (center_Y_start, center_X_start)
    else:
        center_Y_end, center_X_end = np.array(end_shape_zyx)[-2:] / 2

    scaling_matrix = np.array(
        [
            [scaling_factor_zyx[-3], 0, 0, 0],
            [
                0,
                scaling_factor_zyx[-2],
                0,
                -center_Y_start * scaling_factor_zyx[-2] + center_Y_end,
            ],
            [
                0,
                0,
                scaling_factor_zyx[-1],
                -center_X_start * scaling_factor_zyx[-1] + center_X_end,
            ],
            [0, 0, 0, 1],
        ]
    )
    return scaling_matrix


def get_3D_rotation_matrix(
    start_shape_zyx: tuple, angle: float = 0.0, end_shape_zyx: tuple = None
) -> np.ndarray:
    """
    Rotate Transformation Matrix.

    Parameters
    ----------
    start_shape_zyx : Tuple
        Shape of the input
    angle : float, optional
        Angles of rotation in degrees
    end_shape_zyx : Tuple, optional
       Shape of output space

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    # TODO: make this 3D?
    center_Y_start, center_X_start = np.array(start_shape_zyx)[-2:] / 2
    if end_shape_zyx is None:
        center_Y_end, center_X_end = (center_Y_start, center_X_start)
    else:
        center_Y_end, center_X_end = np.array(end_shape_zyx)[-2:] / 2

    theta = np.radians(angle)

    rotation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [
                0,
                np.cos(theta),
                -np.sin(theta),
                -center_Y_start * np.cos(theta)
                + np.sin(theta) * center_X_start
                + center_Y_end,
            ],
            [
                0,
                np.sin(theta),
                np.cos(theta),
                -center_Y_start * np.sin(theta)
                - center_X_start * np.cos(theta)
                + center_X_end,
            ],
            [0, 0, 0, 1],
        ]
    )
    return rotation_matrix


def get_3D_fliplr_matrix(start_shape_zyx: tuple, end_shape_zyx: tuple = None) -> np.ndarray:
    """
    Get 3D left-right flip transformation matrix.

    Parameters
    ----------
    start_shape_zyx : tuple
        Shape of the source volume (Z, Y, X).
    end_shape_zyx : tuple, optional
        Shape of the target volume (Z, Y, X). If None, uses start_shape_zyx.

    Returns
    -------
    np.ndarray
        4x4 transformation matrix for left-right flip.
    """
    center_X_start = start_shape_zyx[-1] / 2
    if end_shape_zyx is None:
        center_X_end = center_X_start
    else:
        center_X_end = end_shape_zyx[-1] / 2

    # Flip matrix: reflects across X axis and translates to maintain center
    flip_matrix = np.array(
        [
            [1, 0, 0, 0],  # Z unchanged
            [0, 1, 0, 0],  # Y unchanged
            [0, 0, -1, 2 * center_X_end],  # X flipped and translated
            [0, 0, 0, 1],  # Homogeneous coordinate
        ]
    )
    return flip_matrix


def apply_affine_transform(
    zyx_data: np.ndarray,
    matrix: np.ndarray,
    output_shape_zyx: tuple,
    method="ants",
    interpolation: str = "linear",
    crop_output_slicing: bool = None,
) -> np.ndarray:
    """_summary_.

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


def pad_to_shape(
    arr: ArrayLike,
    shape: tuple[int, ...],
    mode: str,
    verbose: bool = False,
    **kwargs,
) -> ArrayLike:
    """
    Pad or crop array to match provided shape.

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    shape : Tuple[int]
        Output shape.
    mode : str
        Padding mode (see np.pad).
    verbose : bool
        If True, print verbose output.
    kwargs : dict
        Additional keyword arguments for np.pad.

    Returns
    -------
    ArrayLike
        Padded array.
    """
    assert arr.ndim == len(shape)

    dif = tuple(s - a for s, a in zip(shape, arr.shape, strict=True))
    assert all(d >= 0 for d in dif)

    pad_width = [[s // 2, s - s // 2] for s in dif]

    if verbose:
        click.echo(
            f"padding: input shape {arr.shape}, output shape {shape}, padding {pad_width}"
        )

    return np.pad(arr, pad_width=pad_width, mode=mode, **kwargs)


def center_crop(
    arr: ArrayLike,
    shape: tuple[int, ...],
    verbose: bool = False,
) -> ArrayLike:
    """
    Crop the center of `arr` to match provided shape.

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    shape : Tuple[int, ...]
    """
    assert arr.ndim == len(shape)

    starts = tuple((cur_s - s) // 2 for cur_s, s in zip(arr.shape, shape, strict=True))

    assert all(s >= 0 for s in starts)

    slicing = tuple(slice(s, s + d) for s, d in zip(starts, shape, strict=True))
    if verbose:
        click.echo(
            f"center crop: input shape {arr.shape}, output shape {shape}, slicing {slicing}"
        )
    return arr[slicing]


def match_shape(
    img: ArrayLike,
    shape: tuple[int, ...],
    verbose: bool = False,
) -> ArrayLike:
    """
    Pad or crop array to match provided shape.

    Parameters
    ----------
    img : ArrayLike
        Input array.
    shape : Tuple[int, ...]
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    ArrayLike
        Padded or cropped array.
    """
    if np.any(shape > img.shape):
        padded_shape = np.maximum(img.shape, shape)
        img = pad_to_shape(img, padded_shape, mode="reflect")

    if np.any(shape < img.shape):
        img = center_crop(img, shape)

    if verbose:
        click.echo(f"matched shape: input shape {img.shape}, output shape {shape}")

    return img
