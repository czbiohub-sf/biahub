"""Transform estimators: pluggable strategies for computing a Transform between a moving and a reference array.

A `TransformEstimator` only promises `estimate(mov, ref) -> Transform` -- how it gets
there (point matching, iterative optimization, correlation, a user-supplied matrix) is
private to the implementation. Applying and scoring a Transform are separate,
estimator-independent concerns (see `biahub.core.transform.Transform.apply`).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from numpy.typing import ArrayLike

from biahub.characterize_psf import detect_peaks
from biahub.core.transform import Transform
from biahub.registration.beads import matches_from_beads, transform_from_matches
from biahub.settings import AffineTransformSettings, BeadsMatchSettings, DetectPeaksSettings


@runtime_checkable
class TransformEstimator(Protocol):
    """Computes the Transform that maps `mov` onto `ref`."""

    def estimate(self, mov: ArrayLike, ref: ArrayLike) -> Transform: ...


@runtime_checkable
class NodeDetector(Protocol):
    """Extracts point coordinates (nodes) from an array."""

    def detect(self, array: ArrayLike) -> ArrayLike: ...


class BeadNodeDetector:
    """Detects bead centroids as local-maxima peaks -- today's only node source."""

    def __init__(self, settings: DetectPeaksSettings):
        self.settings = settings

    def detect(self, array: ArrayLike) -> ArrayLike:
        return detect_peaks(
            np.asarray(array),
            block_size=self.settings.block_size,
            threshold_abs=self.settings.threshold_abs,
            nms_distance=self.settings.nms_distance,
            min_distance=self.settings.min_distance,
        )


class NodeGraphEstimator:
    """TransformEstimator over matched point correspondences.

    Composes a `NodeDetector` (beads today; segmentation centroids or other node
    sources later) with the existing graph-matching + transform-fitting steps.
    """

    def __init__(
        self,
        mov_detector: NodeDetector,
        ref_detector: NodeDetector,
        beads_match_settings: BeadsMatchSettings,
        affine_transform_settings: AffineTransformSettings,
    ):
        self.mov_detector = mov_detector
        self.ref_detector = ref_detector
        self.beads_match_settings = beads_match_settings
        self.affine_transform_settings = affine_transform_settings

    @classmethod
    def from_beads_settings(
        cls,
        beads_match_settings: BeadsMatchSettings,
        affine_transform_settings: AffineTransformSettings,
    ) -> NodeGraphEstimator:
        """Build the estimator using today's beads settings shape.

        Separate source/target peak-detection settings, both bead-based.
        """
        return cls(
            mov_detector=BeadNodeDetector(beads_match_settings.source_peaks_settings),
            ref_detector=BeadNodeDetector(beads_match_settings.target_peaks_settings),
            beads_match_settings=beads_match_settings,
            affine_transform_settings=affine_transform_settings,
        )

    def estimate(self, mov: ArrayLike, ref: ArrayLike) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        mov_nodes = self.mov_detector.detect(mov)
        ref_nodes = self.ref_detector.detect(ref)
        matches = matches_from_beads(mov_nodes, ref_nodes, self.beads_match_settings)
        fwd_transform, _inv_transform = transform_from_matches(
            matches,
            mov_nodes,
            ref_nodes,
            self.affine_transform_settings,
            ndim=mov.ndim,
        )
        return fwd_transform
