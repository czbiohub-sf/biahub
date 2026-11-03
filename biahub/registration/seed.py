"""The approximate transform from the two stores' shapes and voxel sizes.

`get_aprox_transform` (`transform.initial: {from: shapes}`) composes
`Transform.from_scaling`, `from_rotation_yx` and `from_flip_x`. ZYX order.
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

    ref_shape_zyx = (ref_Z, ref_Y, ref_X)
    mov_shape_zyx = (mov_Z, mov_Y, mov_X)
    scaling_affine = Transform.from_scaling(
        (scaling_factor_z, scaling_factor_yx, scaling_factor_yx), ref_shape_zyx, ref_shape_zyx
    ).matrix
    rotate90_affine = Transform.from_rotation_yx(
        90 * pre_affine_90degree_rotation, mov_shape_zyx, ref_shape_zyx
    ).matrix

    # Apply flip transformation if requested (flip happens first)
    if pre_affine_fliplr:
        fliplr_affine = Transform.from_flip_x(mov_shape_zyx, ref_shape_zyx).matrix
    else:
        fliplr_affine = np.eye(4)

    compound_affine = np.linalg.inv(scaling_affine @ rotate90_affine @ fliplr_affine)

    return Transform(matrix=compound_affine)
