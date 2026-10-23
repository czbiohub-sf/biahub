"""The nuclei method: match nucleus centres between the channels and fit what they hold.

Nuclei seen in both channels (DAPI or H2B onto virtual-stained nuclei, ...) are detected in
each volume, paired by mutual nearest neighbour in yx within a radius that shrinks, and
fitted. A cell layer is thin, so its nuclei constrain yx and a z offset but no z scale or
tilt: the fit is the transform type in yx plus the median z shift, in µm.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.estimators import EstimationError
from biahub.registration.evaluate import _mutual_nearest, detect_nuclei
from biahub.registration.methods.vote_icp import fit_model
from biahub.settings import NucleiSettings


class NucleiEstimator:
    """TransformEstimator from matched nucleus centres (yx fit plus a z offset).

    Starts from the seed (the moving centres are mapped through it to pair them), so in a
    stack it refines the block before; the final fit pairs the moving centres themselves.

    The seed must be within the first match radius. Somewhat further off, nuclei can pair
    with a neighbour instead and the fit agrees with the wrong pairs; far off, too few
    pair and the estimate fails. Put a coarse block before this one when the seed may be
    off by more than the first radius.
    """

    def __init__(
        self,
        settings: NucleiSettings,
        transform_type: str,
        mov_um: tuple[float, float, float],
        ref_um: tuple[float, float, float],
    ):
        self.settings = settings
        self.transform_type = transform_type
        self.mov_um = np.asarray(mov_um, float)
        self.ref_um = np.asarray(ref_um, float)

    def _detect(self, volume: ArrayLike, voxel_um: np.ndarray) -> np.ndarray:
        s = self.settings
        marks = detect_nuclei(
            volume,
            tuple(voxel_um),
            min_area_um2=s.min_area_um2,
            max_area_um2=s.max_area_um2,
            min_solidity=s.min_solidity,
            smooth_um=s.smooth_um,
        )
        return marks.points_vox

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov_pts, ref_pts = self._detect(mov, self.mov_um), self._detect(ref, self.ref_um)
        need = self.settings.min_matches
        if min(len(mov_pts), len(ref_pts)) < need:
            raise EstimationError(
                f"nuclei: {len(mov_pts)} moving and {len(ref_pts)} reference nuclei "
                f"detected (need {need})"
            )
        current = seed if seed is not None else Transform.identity(3)
        mov_um, ref_um = mov_pts * self.mov_um, ref_pts * self.ref_um
        for radius in self.settings.match_radii_um:
            mapped_um = current.apply_points(mov_pts) * self.ref_um
            i, j = _mutual_nearest(mapped_um[:, 1:], ref_um[:, 1:], radius)
            if len(i) < need:
                raise EstimationError(
                    f"nuclei: {len(i)} pairs within {radius} µm (need {need}); if the "
                    "seed may be further off, put a coarse block before this one"
                )
            current = self._fit(mov_um[i], ref_um[j])
        return current

    def _fit(self, mov_um: np.ndarray, ref_um: np.ndarray) -> Transform:
        """Return the forward voxel transform: yx of the transform type, a z offset."""
        yx = fit_model(mov_um[:, 1:], ref_um[:, 1:], self.transform_type)  # 3 x 3, µm
        in_um = np.eye(4)
        in_um[1:3, 1:3] = yx[:2, :2]
        in_um[1:3, 3] = yx[:2, 2]
        in_um[0, 3] = float(np.median(ref_um[:, 0] - mov_um[:, 0]))
        to_ref_vox = np.diag([*(1.0 / self.ref_um), 1.0])
        from_mov_vox = np.diag([*self.mov_um, 1.0])
        return Transform(to_ref_vox @ in_um @ from_mov_vox, transform_type=self.transform_type)
