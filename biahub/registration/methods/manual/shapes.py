"""The shapes method: the registration from the two grids (shapes and voxel sizes).

A manual method: the person states the setup (how the camera is turned and flipped);
the stores give the rest. The moving grid, flipped left-right (`fliplr`) and turned by `rotate90` quarter turns in
yx, then rescaled by the voxel-size ratio (z, and yx from the last axis) onto the
reference grid. The voxel sizes are the stores' own (their OME-NGFF scale; only the last
three axes are read): there is no default instrument. No pixels are read, so it is the
same at every timepoint and does not depend on where it starts. ZYX order.

On its own (`method: shapes`), as the first block of a stack (the next block starts from
it), or as the run's initial transform (`transform.initial: {from: shapes}`).
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.estimators import EstimatorBase
from biahub.settings import ShapesSettings


class ShapesEstimator(EstimatorBase):
    """The transform between the moving and reference grids, from their metadata."""

    # the answer is the grids': repair from other initial transforms would return it again
    uses_initial_transform = False

    def __init__(
        self,
        mov_voxel_size: tuple,
        ref_voxel_size: tuple,
        rotate90: int,
        fliplr: bool,
        transform_type: str = "affine",
    ):
        self.mov_voxel_size = tuple(mov_voxel_size)
        self.ref_voxel_size = tuple(ref_voxel_size)
        self.rotate90 = rotate90
        self.fliplr = fliplr
        self.transform_type = transform_type

    @classmethod
    def from_settings(
        cls,
        settings: ShapesSettings,
        transform_type: str,
        mov_voxel_size: tuple,
        ref_voxel_size: tuple,
    ) -> ShapesEstimator:
        return cls(
            mov_voxel_size, ref_voxel_size, settings.rotate90, settings.fliplr, transform_type
        )

    def inverse_matrix(self, mov_shape: tuple, ref_shape: tuple) -> np.ndarray:
        """Return the reference -> moving matrix (the direction on disk) for two shapes."""
        mov_shape, ref_shape = tuple(mov_shape[-3:]), tuple(ref_shape[-3:])
        scale_z = self.mov_voxel_size[-3] / self.ref_voxel_size[-3]
        scale_yx = self.mov_voxel_size[-1] / self.ref_voxel_size[-1]
        flip = Transform.from_flip_x(mov_shape, ref_shape).matrix if self.fliplr else np.eye(4)
        forward = (
            Transform.from_scaling((scale_z, scale_yx, scale_yx), ref_shape, ref_shape).matrix
            @ Transform.from_rotation_yx(90 * self.rotate90, mov_shape, ref_shape).matrix
            @ flip
        )
        return np.linalg.inv(forward)

    def transform_for(self, mov_shape: tuple, ref_shape: tuple) -> Transform:
        """Return the forward (moving -> reference) transform for two shapes."""
        return Transform.from_inverse(
            self.inverse_matrix(mov_shape, ref_shape), self.transform_type
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, initial_transform: Transform | None = None
    ) -> Transform:
        return self.transform_for(np.shape(mov), np.shape(ref))
