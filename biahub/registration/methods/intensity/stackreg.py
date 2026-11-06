"""Stackreg (pystackreg) 2D rigid / translation registration."""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike
from pystackreg import StackReg

from biahub.core.transform import Transform
from biahub.registration.estimators import EstimatorBase


class StackregEstimator(EstimatorBase):
    """TransformEstimator using pystackreg (2D rigid/translation registration).

    `StackReg.register(ref, mov)` returns a matrix in (X, Y) axis order -- not this
    codebase's (Y, X) convention -- and in the reference -> moving ("inverse") direction;
    both confirmed empirically against a known synthetic shift (0 error after swapping
    axes and inverting; ~0.44-1.15 relative error for every other combination). Swap
    axes and invert before returning, to satisfy the TransformEstimator contract (true
    forward, moving -> reference, (Y, X)).
    """

    # Ignores the initial transform: repair from other initial transforms would return
    # the same answer.
    uses_initial_transform = False

    _AXIS_SWAP = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    def __init__(self, transformation: int = StackReg.TRANSLATION):
        self.transformation = transformation

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, initial_transform: Transform | None = None
    ) -> Transform:
        # pystackreg's register() takes no initial guess -- correlation-based, like PCC.
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        sr = StackReg(self.transformation)
        xy_inverse_matrix = np.asarray(sr.register(ref, mov))
        yx_inverse = self._AXIS_SWAP @ xy_inverse_matrix @ self._AXIS_SWAP
        return Transform.from_inverse(yx_inverse)  # stackreg registers reference -> moving
