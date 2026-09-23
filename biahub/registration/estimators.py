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
from pystackreg import StackReg

from biahub.characterize_psf import detect_peaks
from biahub.core.transform import Transform
from biahub.registration.ants import estimate as ants_estimate
from biahub.registration.beads import matches_from_beads, transform_from_matches
from biahub.registration.manual import user_assisted_registration
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


class AntsEstimator:
    """TransformEstimator using ANTs intensity-based optimization.

    `biahub.registration.ants.estimate()`'s `fwd_transform` is, despite its name,
    empirically the reference -> moving ("pull") direction, not moving -> reference --
    confirmed against a real volume (rel. err 0.48 applied directly, vs 0.02 inverted;
    baseline unregistered error is 0.35, so using it directly is worse than doing
    nothing). Its own `inv_transform` isn't a genuine inverse either: ANTs doesn't write
    a separately-inverted file for a single affine/similarity stage, so it reads back
    nearly identical to `fwd_transform`. Invert `fwd_transform` ourselves so this
    satisfies the TransformEstimator contract (true forward, moving -> reference).
    """

    def __init__(self, ants_kwargs: dict | None = None, verbose: bool = False):
        self.ants_kwargs = ants_kwargs
        self.verbose = verbose

    def estimate(self, mov: ArrayLike, ref: ArrayLike) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        pull_transform, _unused = ants_estimate(
            ref=ref, mov=mov, verbose=self.verbose, ants_kwargs=self.ants_kwargs
        )
        return pull_transform.invert()


class ManualEstimator:
    """TransformEstimator via user-assisted (napari) point annotation.

    Interactive: `estimate()` opens a napari viewer and blocks on `input()` until you
    annotate matching points. `user_assisted_registration` builds its transform
    correctly (skimage point-fit composed with the pre-alignment matrix, true
    moving -> reference), then explicitly inverts it before returning -- so, like
    ants.py:estimate(), its return value is the reference -> moving ("pull") direction.
    Invert once more here to satisfy the TransformEstimator contract.
    """

    def __init__(
        self,
        source_channel_name: str,
        target_channel_name: str,
        source_channel_voxel_size: tuple[float, float, float],
        target_channel_voxel_size: tuple[float, float, float],
        similarity: bool = False,
        pre_affine_90degree_rotation: int = 0,
        pre_affine_fliplr: bool = False,
    ):
        self.source_channel_name = source_channel_name
        self.target_channel_name = target_channel_name
        self.source_channel_voxel_size = source_channel_voxel_size
        self.target_channel_voxel_size = target_channel_voxel_size
        self.similarity = similarity
        self.pre_affine_90degree_rotation = pre_affine_90degree_rotation
        self.pre_affine_fliplr = pre_affine_fliplr

    def estimate(self, mov: ArrayLike, ref: ArrayLike) -> Transform:
        (pull_matrix,) = user_assisted_registration(
            source_channel_volume=np.asarray(mov),
            source_channel_name=self.source_channel_name,
            source_channel_voxel_size=self.source_channel_voxel_size,
            target_channel_volume=np.asarray(ref),
            target_channel_name=self.target_channel_name,
            target_channel_voxel_size=self.target_channel_voxel_size,
            similarity=self.similarity,
            pre_affine_90degree_rotation=self.pre_affine_90degree_rotation,
            pre_affine_fliplr=self.pre_affine_fliplr,
        )
        return Transform(matrix=np.asarray(pull_matrix)).invert()


class StackregEstimator:
    """TransformEstimator using pystackreg (2D rigid/translation registration).

    `StackReg.register(ref, mov)` returns a matrix in (X, Y) axis order -- not this
    codebase's (Y, X) convention -- and in the reference -> moving ("pull") direction;
    both confirmed empirically against a known synthetic shift (0 error after swapping
    axes and inverting; ~0.44-1.15 relative error for every other combination). Swap
    axes and invert before returning, to satisfy the TransformEstimator contract (true
    forward, moving -> reference, (Y, X)).
    """

    _AXIS_SWAP = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    def __init__(self, transformation: int = StackReg.TRANSLATION):
        self.transformation = transformation

    def estimate(self, mov: ArrayLike, ref: ArrayLike) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        sr = StackReg(self.transformation)
        xy_pull_matrix = np.asarray(sr.register(ref, mov))
        yx_matrix = self._AXIS_SWAP @ xy_pull_matrix @ self._AXIS_SWAP
        return Transform(matrix=yx_matrix).invert()
