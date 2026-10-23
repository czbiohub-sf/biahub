"""The tiles method: tile shifts on max projections, fitted in yx.

The moving volume, warped by the current transform, is phase-correlated with the reference
on a grid of tiles of their max projections (`evaluate.tile_residuals`); the tile shifts
are fitted with the transform type in yx and the fit corrects the current transform. Max
projections hold no depth, so z is left as the seed has it: a later block (e.g. nuclei)
or the seed brings z.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.estimators import EstimationError
from biahub.registration.evaluate import tile_residuals
from biahub.registration.methods.vote_icp import fit_model
from biahub.settings import TilesSettings


class TilesEstimator:
    """TransformEstimator from tile shifts (yx only; z stays the seed's).

    Each of `rounds` measures the tiles with the current transform and corrects it, so a
    shift larger than a tile's reach in the first round is taken out by the next.
    """

    def __init__(
        self,
        settings: TilesSettings,
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
                    f"tiles: {tiles['n_tiles']} tiles correlated (need {s.min_tiles}): too "
                    "little shared structure, or the seed is further off than a tile reaches"
                )
            current = self._correct(current, tiles["residuals"])
        return current

    def _correct(self, current: Transform, residuals: dict) -> Transform:
        """Return `current` corrected by the yx fit of the tile shifts."""
        reference = np.asarray(residuals["position_um"])[:, 1:]
        mapped = reference - np.asarray(residuals["diff_um"])[:, 1:]  # where content landed
        yx = fit_model(mapped, reference, self.transform_type)  # 3 x 3, in µm
        correction_um = np.eye(4)
        correction_um[1:3, 1:3] = yx[:2, :2]
        correction_um[1:3, 3] = yx[:2, 2]
        to_vox = np.diag([*(1.0 / self.ref_um), 1.0])
        from_vox = np.diag([*self.ref_um, 1.0])
        correction = to_vox @ correction_um @ from_vox  # on the reference grid
        return Transform(
            correction @ np.asarray(current.matrix), transform_type=self.transform_type
        )
