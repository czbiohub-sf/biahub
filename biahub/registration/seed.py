"""The approximate seed: a transform from the two stores' shapes and voxel sizes.

`get_aprox_transform` (the engine's `transform.seed_from_shapes`) and the matrix builders
it composes -- rescaling, a 90-degree rotation and a left-right flip, each about the
volume's centre. ZYX order; 4x4 homogeneous matrices.
"""

import click
import numpy as np

from biahub.core.transform import Transform


def get_aprox_transform(
    mov_shape: tuple[int, int, int],
    ref_shape: tuple[int, int, int],
    mov_voxel_size: tuple[float, ...],
    ref_voxel_size: tuple[float, ...],
    pre_affine_90degree_rotation: int = -1,
    pre_affine_fliplr: bool = False,
    verbose: bool = False,
) -> Transform:
    """Return the approximate transform between two stores from their shapes and voxels.

    Rescales the moving grid by the voxel-size ratio (z, and yx from the last axis),
    after an optional 90-degree rotation and left-right flip, onto the reference grid.
    The voxel sizes are the stores' own (their OME-NGFF scale; only the last three axes
    are read): there is no default instrument.
    """
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
