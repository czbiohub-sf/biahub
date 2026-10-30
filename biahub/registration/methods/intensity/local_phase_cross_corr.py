"""Local phase cross-correlation: the `local` option of the phase_cross_corr method.

Tile shifts between the warped moving volume and the reference, fitted in yx; z stays the
seed's. Its own file because it measures tiles with the evaluation's `tile_residuals`,
while the evaluation uses the phase-correlation primitive in `phase_cross_corr.py`: kept
apart, the imports run one way.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.estimators import EstimationError, fit_pairs
from biahub.registration.metrics import tile_residuals
from biahub.settings import LocalPhaseCrossCorrSettings


class LocalPCCEstimator:
    """TransformEstimator from local phase cross-correlation: tile shifts fitted in yx.

    The moving volume, warped by the current transform, is phase-correlated with the
    reference on a grid of tiles of their max projections (`metrics.tile_residuals`);
    the shifts are fitted with the transform type in yx and the fit corrects the current
    transform. Max projections hold no depth, so z stays the seed's.

    Each of `rounds` measures the tiles with the current transform and corrects it, so a
    shift larger than a tile's reach in the first round is taken out by the next.
    """

    def __init__(
        self,
        settings: LocalPhaseCrossCorrSettings,
        transform_type: str,
        ref_um: tuple[float, float, float],
    ):
        self.settings = settings
        self.transform_type = transform_type
        self.ref_um = np.asarray(ref_um, float)

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        s = self.settings
        current = seed if seed is not None else Transform.identity(3)
        for _ in range(s.rounds):
            tiles = tile_residuals(
                current,
                mov,
                ref,
                tuple(self.ref_um),
                compare=s.compare,
                tile_px=s.tile_px,
                grid=s.grid,
                min_ncc=s.min_ncc,
            )
            if tiles["n_tiles"] < s.min_tiles:
                raise EstimationError(
                    f"local phase_cross_corr: {tiles['n_tiles']} tiles correlated (need {s.min_tiles}): too "
                    "little shared structure, or the seed is further off than a tile reaches"
                )
            current = self._correct(current, tiles["residuals"])
        return current

    def _correct(self, current: Transform, residuals: dict) -> Transform:
        """Return `current` corrected by the yx fit of the tile shifts."""
        reference = np.asarray(residuals["position_um"])[:, 1:]
        mapped = reference - np.asarray(residuals["diff_um"])[:, 1:]  # where content landed
        yx = fit_pairs(mapped, reference, self.transform_type)  # 3 x 3, in µm
        correction_um = np.eye(4)
        correction_um[1:3, 1:3] = yx[:2, :2]
        correction_um[1:3, 3] = yx[:2, 2]
        to_vox = np.diag([*(1.0 / self.ref_um), 1.0])
        from_vox = np.diag([*self.ref_um, 1.0])
        correction = to_vox @ correction_um @ from_vox  # on the reference grid
        return Transform(
            correction @ np.asarray(current.matrix), transform_type=self.transform_type
        )
