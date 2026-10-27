"""The nuclei method: landmark-based, with nuclei for landmarks.

The same parts as beads (`NodeGraphEstimator`): a detector per channel -- here nuclei
(`evaluate.detect_nuclei`) -- the beads' graph matchers, and a fit. Nuclei seen in both
channels (DAPI or H2B onto virtual-stained nuclei, ...) are detected on the moving volume
warped by the current transform and on the reference, matched, and fitted. A cell layer
is thin, so its nuclei constrain yx and a z offset but no z scale or tilt: the fit is the
transform type in yx plus the median z shift.

The seed must bring the nuclei within the matcher's reach (a few µm); put a coarse block
before this one when it may be further off.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.estimators import ScoreFn, fit_pairs
from biahub.registration.evaluate import detect_nuclei
from biahub.registration.methods.landmark.bead_matching import NodeGraphEstimator
from biahub.settings import AffineTransformSettings, BeadsMatchSettings, NucleiSettings


class NucleiNodeDetector:
    """Detects nucleus centres (voxels, z y x) as nodes for graph matching."""

    def __init__(self, settings: NucleiSettings, voxel_um: tuple[float, float, float]):
        self.settings = settings
        self.voxel_um = tuple(voxel_um)

    def detect(self, array: ArrayLike) -> np.ndarray:
        s = self.settings
        return detect_nuclei(
            array,
            self.voxel_um,
            min_area_um2=s.min_area_um2,
            max_area_um2=s.max_area_um2,
            min_solidity=s.min_solidity,
            smooth_um=s.smooth_um,
        ).points_vox


def yx_and_z_shift_fit(transform_type: str, voxel_um: tuple[float, float, float]):
    """Return a fit for matched nodes on one grid: the type in yx (in µm), the z shift."""
    um = np.asarray(voxel_um, float)

    def fit(matches, mov_nodes, ref_nodes) -> Transform:
        src = np.asarray(mov_nodes)[matches[:, 0]] * um
        dst = np.asarray(ref_nodes)[matches[:, 1]] * um
        yx = fit_pairs(src[:, 1:], dst[:, 1:], transform_type)
        in_um = np.eye(4)
        in_um[1:3, 1:3] = yx[:2, :2]
        in_um[1:3, 3] = yx[:2, 2]
        in_um[0, 3] = float(np.median(dst[:, 0] - src[:, 0]))
        to_vox, from_vox = np.diag([*(1.0 / um), 1.0]), np.diag([*um, 1.0])
        return Transform(to_vox @ in_um @ from_vox, transform_type=transform_type)

    return fit


def nuclei_estimator(
    settings: NucleiSettings,
    transform_type: str,
    ref_um: tuple[float, float, float],
    score_fn: ScoreFn | None = None,
) -> NodeGraphEstimator:
    """Build the nuclei method: nucleus detectors, the configured matcher, a yx + z fit.

    Both channels are detected on the reference grid (the moving volume is warped onto
    it first), so `ref_um` is the voxel size for both.
    """
    matching = BeadsMatchSettings(
        algorithm=settings.algorithm,
        source_peaks_settings=None,
        target_peaks_settings=None,
        hungarian_match_settings=settings.hungarian_match_settings,
        spectral_match_settings=settings.spectral_match_settings,
        match_descriptor_settings=settings.match_descriptor_settings,
        filter_matches_settings=settings.filter_matches_settings,
    )
    return NodeGraphEstimator(
        mov_detector=NucleiNodeDetector(settings, ref_um),
        ref_detector=NucleiNodeDetector(settings, ref_um),
        beads_match_settings=matching,
        affine_transform_settings=AffineTransformSettings(transform_type=transform_type),
        iterations=settings.iterations,
        score_fn=score_fn,
        fit=yx_and_z_shift_fit(transform_type, ref_um),
    )
