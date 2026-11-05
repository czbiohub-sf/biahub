"""The approximate seed from the two stores' shapes and voxel sizes (registration/seed.py)."""

import numpy as np
import pytest


def test_the_approximate_initial_transform_takes_the_voxel_sizes_it_is_given():
    # No built-in voxel sizes: the caller reads them from each store's metadata. Same
    # grids and no rotation or flip leave the moving volume where it is.
    from biahub.registration.seed import get_aprox_transform

    shape, um = (30, 100, 100), (1.0, 0.5, 0.5)
    same = get_aprox_transform(
        mov_shape=shape,
        ref_shape=shape,
        mov_voxel_size=um,
        ref_voxel_size=um,
        pre_affine_90degree_rotation=0,
    )
    np.testing.assert_allclose(same.matrix, np.eye(4), atol=1e-12)
    # the moving voxels twice the reference's in z: one moving plane spans two reference ones
    coarse = get_aprox_transform(
        mov_shape=shape,
        ref_shape=shape,
        mov_voxel_size=(2.0, 0.5, 0.5),
        ref_voxel_size=um,
        pre_affine_90degree_rotation=0,
    )
    np.testing.assert_allclose(np.diag(coarse.matrix)[:3], (0.5, 1.0, 1.0))
    with pytest.raises(TypeError, match="voxel_size"):
        get_aprox_transform(mov_shape=shape, ref_shape=shape)
