"""Transform estimators: pluggable strategies for computing a Transform between a moving and a reference array.

A `TransformEstimator` only promises `estimate(mov, ref) -> Transform` -- how it gets
there (point matching, iterative optimization, correlation, a user-supplied matrix) is
private to the implementation. Applying and scoring a Transform are separate,
estimator-independent concerns (see `biahub.core.transform.Transform.apply`).
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

from numpy.typing import ArrayLike

from biahub.characterize_psf import detect_peaks
from biahub.core.transform import Transform
from biahub.registration.beads import matches_from_beads, transform_from_matches
from biahub.registration.phase_cross_correlation import (
    phase_cross_corr,
    phase_cross_corr_padding,
)
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    PhaseCrossCorrSettings,
)


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


class PCCEstimator:
    """TransformEstimator using phase cross-correlation (rigid translation only).

    `phase_cross_corr(ref, mov)`'s shift is already the forward (moving -> reference)
    translation in the array's own axis order (see issue #356 for a case where an
    existing caller of this function builds the wrong-axis matrix by hand instead).
    """

    def __init__(
        self,
        function_type: Literal["custom", "custom_padding"] = "custom",
        normalization: Literal["magnitude", "classic"] | None = None,
        maximum_shift: float = 1.2,
    ):
        self.function_type = function_type
        self.normalization = normalization
        self.maximum_shift = maximum_shift

    @classmethod
    def from_settings(cls, settings: PhaseCrossCorrSettings) -> PCCEstimator:
        return cls(
            function_type=settings.function_type,
            normalization=settings.normalization,
            maximum_shift=settings.maximum_shift,
        )

    def estimate(self, mov: ArrayLike, ref: ArrayLike) -> Transform:
        mov = np.asarray(mov).astype(np.float32)
        ref = np.asarray(ref).astype(np.float32)
        if self.function_type == "custom_padding":
            shift, _corr = phase_cross_corr_padding(
                ref, mov, maximum_shift=self.maximum_shift, normalization=self.normalization
            )
        else:
            shift, _corr = phase_cross_corr(ref, mov, normalization=self.normalization)
        return Transform.from_translation(shift)
