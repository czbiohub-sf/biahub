"""Focus-finding stabilization: z from the in-focus slice, yx from stackreg on that slice.

The legacy `estimate-stabilization --method focus-finding` in the engine's shape. Each
volume's in-focus slice is found with waveorder's transverse-band criterion on a centre
crop; the z drift is the difference of the two focus indices and the yx drift is a
pystackreg translation between the two in-focus slices (same crop, clipped at zero, as
the legacy code did). The seed is ignored: focus finding is absolute and stackreg is
correlation-based, like PCC.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from numpy.typing import ArrayLike
from waveorder.focus import focus_from_transverse_band

from biahub.core.transform import Transform
from biahub.registration.estimators import EstimationError
from biahub.registration.methods.stackreg import StackregEstimator
from biahub.settings import FocusSettings

FocusAxes = Literal["z", "xy", "xyz"]


def center_crop_yx(zyx: np.ndarray, crop_xy: tuple[int, int]) -> np.ndarray:
    """Centre crop of at most `crop_xy` (X, Y) pixels, the legacy `center_crop_xy` convention."""
    _, y, x = zyx.shape
    cx, cy = min(crop_xy[0], x), min(crop_xy[1], y)
    return zyx[:, y // 2 - cy // 2 : y // 2 + cy // 2, x // 2 - cx // 2 : x // 2 + cx // 2]


def find_focus(
    zyx: np.ndarray, pixel_size: float, na_det: float = 1.35, lambda_ill: float = 0.5
) -> int:
    """Index of the in-focus slice, or `EstimationError` when the volume has none.

    An empty field of view (all zeros) has no focus; the legacy code wrote 0 and forward
    filled it from the neighbours -- in the engine that is the repair pass's job.
    """
    if not np.any(zyx):
        raise EstimationError("empty field of view, no focus to find")
    z = focus_from_transverse_band(
        zyx, NA_det=na_det, lambda_ill=lambda_ill, pixel_size=pixel_size
    )
    if z is None:
        raise EstimationError("no in-focus slice found")
    return int(z)


class FocusEstimator:
    """TransformEstimator for same-channel drift: focus index for z, stackreg for yx.

    `axes` selects what is estimated: "z" (focus only), "xy" (stackreg on the in-focus
    slices; the focus is still found to pick the slice) or "xyz". The returned transform
    is a pure translation in the forward (moving -> reference) direction.
    """

    def __init__(
        self,
        pixel_size: float,
        axes: FocusAxes = "xyz",
        center_crop_xy: tuple[int, int] = (800, 800),
        na_det: float = 1.35,
        lambda_ill: float = 0.5,
    ):
        if axes not in ("z", "xy", "xyz"):
            raise ValueError(f"axes must be 'z', 'xy' or 'xyz', got {axes!r}")
        self.pixel_size = float(pixel_size)
        self.axes = axes
        self.center_crop_xy = (int(center_crop_xy[0]), int(center_crop_xy[1]))
        self.na_det = na_det
        self.lambda_ill = lambda_ill
        self._stackreg = StackregEstimator()

    @classmethod
    def from_settings(cls, focus_settings: FocusSettings, pixel_size: float) -> FocusEstimator:
        return cls(
            pixel_size=pixel_size,
            axes=focus_settings.axes,
            center_crop_xy=tuple(focus_settings.center_crop_xy),
            na_det=focus_settings.na_det,
            lambda_ill=focus_settings.lambda_ill,
        )

    def _focus(self, zyx: np.ndarray) -> int:
        return find_focus(
            center_crop_yx(zyx, self.center_crop_xy),
            self.pixel_size,
            self.na_det,
            self.lambda_ill,
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        z_mov, z_ref = self._focus(mov), self._focus(ref)

        matrix = np.eye(4)
        if "z" in self.axes:
            matrix[0, 3] = z_ref - z_mov
        if "xy" in self.axes:
            mov_yx = np.clip(
                center_crop_yx(mov[z_mov : z_mov + 1], self.center_crop_xy)[0], 0, None
            )
            ref_yx = np.clip(
                center_crop_yx(ref[z_ref : z_ref + 1], self.center_crop_xy)[0], 0, None
            )
            matrix[1:, 1:] = self._stackreg.estimate(mov_yx, ref_yx).matrix
        return Transform(matrix, transform_type="euclidean")
