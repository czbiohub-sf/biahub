"""Transform estimators: pluggable strategies for computing a Transform between a moving and a reference array.

A `TransformEstimator` only promises `estimate(mov, ref) -> Transform` -- how it gets
there (point matching, iterative optimization, correlation, a user-supplied matrix) is
private to the implementation. Applying and scoring a Transform are separate,
estimator-independent concerns (see `biahub.core.transform.Transform.apply`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from numpy.typing import ArrayLike
from pystackreg import StackReg
from scipy.spatial import cKDTree

from biahub.characterize_psf import detect_peaks
from biahub.core.transform import Transform
from biahub.registration.ants import DEFAULT_ANTS_KWARGS, preprocess_zyx
from biahub.registration.ants import estimate as ants_estimate
from biahub.registration.beads import (
    matches_from_beads,
    score_transform,
    transform_from_matches,
)
from biahub.registration.manual import user_assisted_registration
from biahub.registration.metrics import normalized_mutual_information, residual_score
from biahub.registration.phase_cross_correlation import (
    phase_cross_corr,
    phase_cross_corr_padding,
)
from biahub.registration.pointcloud import (
    fit_affine,
    translation_matrix,
    vote_drift,
    vote_icp_register,
)
from biahub.settings import (
    AffineTransformSettings,
    AntsRegistrationSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    PhaseCrossCorrSettings,
    SeedCorrectionSettings,
    VoteIcpSettings,
)


@runtime_checkable
class TransformEstimator(Protocol):
    """Computes the Transform that maps `mov` onto `ref`.

    `seed`, when given, is an initial guess in this same contract's direction (true
    forward, moving -> reference) -- e.g. a previous timepoint's accepted result via
    `SeedPolicy`. Estimators that don't use a seed (correlation-based methods: PCC,
    stackreg) ignore it.
    """

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform: ...


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


class EstimationError(RuntimeError):
    """`estimate()` could not produce a transform (too few nodes or matches, degenerate fit)."""


ScoreFn = Callable[[Transform, np.ndarray, np.ndarray], float]
NodeMatcher = Callable[
    [np.ndarray, np.ndarray], np.ndarray
]  # (mov_nodes, ref_nodes) -> (N, 2)


def beads_score_fn(beads_match_settings: BeadsMatchSettings) -> ScoreFn:
    """Build the score function the beads settings ask for (`qc_settings.score_metric`)."""
    metric = beads_match_settings.qc_settings.score_metric
    if metric == "overlap":
        return lambda transform, mov, ref: score_transform(
            transform, mov, ref, beads_match_settings
        )
    if metric == "residual":
        return lambda transform, mov, ref: residual_score(
            transform, mov, ref, beads_match_settings
        )
    if metric == "mutual_information":
        return normalized_mutual_information
    raise ValueError(f"unknown score_metric {metric!r}")


class NodeGraphEstimator:
    """TransformEstimator over matched point correspondences.

    Composes a `NodeDetector` (beads today; segmentation centroids or other node
    sources later) with graph matching and transform fitting.

    One pass warps `mov` by the current guess (`Transform.apply`), detects nodes in the
    warped frame, matches, fits the residual correction and composes it back
    (`correction @ seed`: seed first, then correction). Nodes are detected in the warped
    frame, so each pass sees a better-aligned image than the last; `iterations > 1` with
    a `score_fn` repeats the pass from the previous result and returns the best-scoring
    transform (never a later, worse one). Without a `score_fn` the last pass is returned.
    """

    def __init__(
        self,
        mov_detector: NodeDetector,
        ref_detector: NodeDetector,
        beads_match_settings: BeadsMatchSettings,
        affine_transform_settings: AffineTransformSettings,
        iterations: int = 1,
        score_fn: ScoreFn | None = None,
        matcher: NodeMatcher | None = None,
    ):
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        self.mov_detector = mov_detector
        self.ref_detector = ref_detector
        self.beads_match_settings = beads_match_settings
        self.affine_transform_settings = affine_transform_settings
        self.iterations = iterations
        self.score_fn = score_fn
        self.matcher = matcher or (
            lambda mov_nodes, ref_nodes: matches_from_beads(
                mov_nodes, ref_nodes, beads_match_settings
            )
        )

    @classmethod
    def from_beads_settings(
        cls,
        beads_match_settings: BeadsMatchSettings,
        affine_transform_settings: AffineTransformSettings,
        iterations: int | None = None,
    ) -> TransformEstimator:
        """Bead-peak detection on both sides, scored by bead overlap.

        `iterations` defaults to `beads_match_settings.qc_settings.iterations`. With
        `spectral_arm` on, returns a `CompetingEstimator` of the configured matcher and a
        spectral-acquire-then-refine cascade; the higher-scoring arm wins.
        """
        iterations = (
            beads_match_settings.qc_settings.iterations if iterations is None else iterations
        )

        score_fn = beads_score_fn(beads_match_settings)

        def node_graph(settings: BeadsMatchSettings, n_iterations: int) -> NodeGraphEstimator:
            return cls(
                mov_detector=BeadNodeDetector(settings.source_peaks_settings),
                ref_detector=BeadNodeDetector(settings.target_peaks_settings),
                beads_match_settings=settings,
                affine_transform_settings=affine_transform_settings,
                iterations=n_iterations,
                score_fn=score_fn,
            )

        if beads_match_settings.estimation_mode == "vote_icp":
            configured: TransformEstimator = VoteIcpEstimator.from_beads_settings(
                beads_match_settings, score_fn=score_fn
            )
        else:
            configured = node_graph(beads_match_settings, iterations)
        if beads_match_settings.seed_correction_settings.mode != "none":
            configured = ChainedEstimator(
                [VoteSeedCorrection.from_beads_settings(beads_match_settings), configured],
                score_fn=score_fn,
            )
        if beads_match_settings.spectral_arm == "off":
            return configured
        spectral_settings = beads_match_settings.model_copy(deep=True)
        spectral_settings.algorithm = "spectral"
        cascade = ChainedEstimator(
            [
                node_graph(spectral_settings, iterations),
                node_graph(beads_match_settings, iterations),
            ],
            score_fn=score_fn,
        )
        return CompetingEstimator(
            {beads_match_settings.algorithm: configured, "spectral": cascade},
            score_fn=score_fn,
            escalate_below=(
                beads_match_settings.qc_settings.score_threshold
                if beads_match_settings.spectral_arm == "on_low_score"
                else None
            ),
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        current = seed
        best: Transform | None = None
        best_score = -np.inf
        for _ in range(self.iterations):
            try:
                current = self._single_pass(mov, ref, current)
            except EstimationError:
                # A later pass that cannot match (e.g. the previous pass drifted) must not
                # throw away an earlier usable result.
                if best is not None:
                    break
                raise
            if self.score_fn is None:
                best = current
                continue
            score = self.score_fn(current, mov, ref)
            if np.isfinite(score) and score > best_score:
                best, best_score = current, score
        if best is None:
            raise EstimationError(
                f"no finite score in {self.iterations} iteration(s): nodes not detectable "
                "after warping"
            )
        return best

    def _single_pass(
        self, mov: np.ndarray, ref: np.ndarray, seed: Transform | None
    ) -> Transform:
        mov_for_detection = seed.apply(mov, reference=ref) if seed is not None else mov
        mov_nodes = np.asarray(self.mov_detector.detect(mov_for_detection))
        ref_nodes = np.asarray(self.ref_detector.detect(ref))
        if len(mov_nodes) < 3 or len(ref_nodes) < 3:
            raise EstimationError(
                f"too few nodes to fit a transform: {len(mov_nodes)} moving, "
                f"{len(ref_nodes)} reference (need >= 3 each)"
            )
        matches = np.asarray(self.matcher(mov_nodes, ref_nodes))
        if matches.ndim != 2 or len(matches) < 3:
            raise EstimationError(
                f"too few matches to fit a transform: {len(matches)} from "
                f"{len(mov_nodes)} x {len(ref_nodes)} nodes (need >= 3)"
            )
        correction, _inv_correction = transform_from_matches(
            matches,
            mov_nodes,
            ref_nodes,
            self.affine_transform_settings,
            ndim=mov.ndim,
        )
        if not np.all(np.isfinite(correction.matrix)):
            raise EstimationError(
                f"degenerate fit from {len(matches)} matches (non-finite matrix)"
            )
        return correction @ seed if seed is not None else correction


class ChainedEstimator:
    """Run estimators in sequence, each seeded by the previous result.

    Acquire-then-refine: e.g. a spectral matcher that can find the correspondence from a
    poor seed, followed by the Hungarian matcher that is more precise once close. With a
    `score_fn`, the best-scoring stage output is returned (a refinement that makes things
    worse is dropped); without one, the last stage's output.
    """

    def __init__(self, stages: list[TransformEstimator], score_fn: ScoreFn | None = None):
        if not stages:
            raise ValueError("ChainedEstimator needs at least one stage")
        self.stages = list(stages)
        self.score_fn = score_fn

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        current = seed
        best: Transform | None = None
        best_score = -np.inf
        for stage in self.stages:
            try:
                current = stage.estimate(mov, ref, seed=current)
            except EstimationError:
                if best is not None:
                    break
                raise
            if self.score_fn is None:
                best = current
                continue
            score = self.score_fn(current, mov, ref)
            if np.isfinite(score) and score > best_score:
                best, best_score = current, score
        if best is None:
            raise EstimationError("no stage of the chain produced a finite score")
        return best


class CompetingEstimator:
    """Run several estimators on the same input and keep the best-scoring result.

    An arm that raises `EstimationError` is skipped; only when every arm fails does the
    whole estimate fail, with each arm's reason. With `escalate_below`, arms after the
    first run only while the best score so far is below it. `last_winner` names the arm
    whose result was returned.
    """

    def __init__(
        self,
        arms: dict[str, TransformEstimator],
        score_fn: ScoreFn,
        escalate_below: float | None = None,
    ):
        if not arms:
            raise ValueError("CompetingEstimator needs at least one arm")
        self.arms = dict(arms)
        self.score_fn = score_fn
        self.escalate_below = escalate_below
        self.last_winner: str | None = None
        self.last_scores: dict[str, float] = {}

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        best: Transform | None = None
        best_score = -np.inf
        self.last_winner = None
        self.last_scores = {}
        failures: dict[str, str] = {}
        for index, (name, arm) in enumerate(self.arms.items()):
            if (
                index > 0
                and self.escalate_below is not None
                and best is not None
                and best_score >= self.escalate_below
            ):
                break
            try:
                transform = arm.estimate(mov, ref, seed=seed)
            except EstimationError as e:
                failures[name] = str(e)
                continue
            score = self.score_fn(transform, mov, ref)
            self.last_scores[name] = float(score)
            if np.isfinite(score) and score > best_score:
                best, best_score, self.last_winner = transform, score, name
        if best is None:
            raise EstimationError(
                "every arm failed: " + "; ".join(f"{k}: {v}" for k, v in failures.items())
            )
        return best


class VoteIcpEstimator:
    """TransformEstimator over peak clouds by iterated displacement voting.

    Reach stage: dense detection on the raw moving volume (`vote_peaks_settings`), voted
    ICP from the seed with a wide, shrinking capture radius. Precision stage: re-run at
    short range from that result with the pipeline's own moving detection, kept only on a
    strict score win. Raises `EstimationError` when voting abstains before any fit.
    """

    def __init__(
        self,
        dense_detector: NodeDetector,
        precise_detector: NodeDetector,
        ref_detector: NodeDetector,
        settings: VoteIcpSettings,
        score_fn: ScoreFn,
    ):
        self.dense_detector = dense_detector
        self.precise_detector = precise_detector
        self.ref_detector = ref_detector
        self.settings = settings
        self.score_fn = score_fn

    @classmethod
    def from_beads_settings(
        cls, beads_match_settings: BeadsMatchSettings, score_fn: ScoreFn
    ) -> VoteIcpEstimator:
        settings = beads_match_settings.vote_icp_settings
        return cls(
            dense_detector=BeadNodeDetector(settings.vote_peaks_settings),
            precise_detector=BeadNodeDetector(beads_match_settings.source_peaks_settings),
            ref_detector=BeadNodeDetector(beads_match_settings.target_peaks_settings),
            settings=settings,
            score_fn=score_fn,
        )

    def _register(self, mov_peaks, ref_peaks, pull_seed, initial_radius):
        s = self.settings
        return vote_icp_register(
            mov_peaks=mov_peaks,
            ref_peaks=ref_peaks,
            initial_transform=pull_seed,
            initial_capture_radius=initial_radius,
            min_capture_radius=s.min_capture_radius,
            radius_decay=s.radius_decay,
            cluster_radius=s.cluster_radius,
            min_votes=s.min_votes,
            max_iterations=s.max_iterations,
            convergence_translation=s.convergence_translation,
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        ndim = mov.ndim
        pull_seed = (seed.invert() if seed is not None else Transform.identity(ndim)).matrix
        ref_peaks = np.asarray(self.ref_detector.detect(ref))
        dense_peaks = np.asarray(self.dense_detector.detect(mov))
        if min(len(dense_peaks), len(ref_peaks)) < self.settings.min_votes:
            raise EstimationError(
                f"vote_icp: {len(dense_peaks)} dense moving / {len(ref_peaks)} reference "
                f"peaks (need >= {self.settings.min_votes})"
            )
        pull, info = self._register(
            dense_peaks, ref_peaks, pull_seed, self.settings.initial_capture_radius
        )
        if pull is None:
            raise EstimationError(
                f"vote_icp: voting abstained after {info['iterations']} iteration(s)"
            )
        best = Transform(
            pull, transform_type=seed.transform_type if seed else "affine"
        ).invert()
        best_score = self.score_fn(best, mov, ref)

        precise_peaks = np.asarray(self.precise_detector.detect(mov))
        if len(precise_peaks) >= self.settings.min_votes:
            precise_pull, _info = self._register(
                precise_peaks, ref_peaks, pull, self.settings.min_capture_radius
            )
            if precise_pull is not None:
                precise = Transform(precise_pull, transform_type=best.transform_type).invert()
                precise_score = self.score_fn(precise, mov, ref)
                if np.isfinite(precise_score) and precise_score > best_score:
                    best = precise
        return best


class VoteSeedCorrection:
    """A stage that corrects a seed by bead displacement voting, for a chain's first slot.

    Warps `mov` by the seed, detects beads densely and votes for the residual drift;
    "voteseed" composes the densest cluster's mean displacement into the seed, "votefit"
    also fits an affine on the cluster's pairs. Every candidate competes against the
    unchanged seed on the median nearest-neighbour distance of the pipeline's own
    detection, so a bad vote leaves the seed unchanged. Returns the (possibly unchanged)
    seed; without a seed, the identity.
    """

    def __init__(
        self,
        settings: SeedCorrectionSettings,
        mov_detector: NodeDetector,
        ref_detector: NodeDetector,
    ):
        self.settings = settings
        self.mov_detector = mov_detector
        self.ref_detector = ref_detector
        self.dense_detector = BeadNodeDetector(settings.vote_peaks_settings)
        self.last_note: str = ""

    @classmethod
    def from_beads_settings(
        cls, beads_match_settings: BeadsMatchSettings
    ) -> VoteSeedCorrection:
        return cls(
            settings=beads_match_settings.seed_correction_settings,
            mov_detector=BeadNodeDetector(beads_match_settings.source_peaks_settings),
            ref_detector=BeadNodeDetector(beads_match_settings.target_peaks_settings),
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        seed = seed if seed is not None else Transform.identity(mov.ndim)
        settings = self.settings
        ref_peaks = np.asarray(self.ref_detector.detect(ref))
        if len(ref_peaks) < settings.min_votes:
            self.last_note = f"only {len(ref_peaks)} ref peaks; seed unchanged"
            return seed
        ref_tree = cKDTree(ref_peaks)

        def nn_median(transform: Transform) -> float:
            peaks = np.asarray(self.mov_detector.detect(transform.apply(mov, reference=ref)))
            if len(peaks) == 0:
                return np.inf
            return float(np.median(ref_tree.query(peaks)[0]))

        warped = seed.apply(mov, reference=ref)
        dense_peaks = np.asarray(self.dense_detector.detect(warped))
        drift, pairs, n_votes = vote_drift(
            dense_peaks,
            ref_peaks,
            settings.capture_radius,
            settings.cluster_radius,
            settings.min_votes,
        )
        if drift is None:
            self.last_note = (
                f"only {len(dense_peaks)} dense peaks / too few votes; seed unchanged"
            )
            return seed
        # Work in the pull convention the vote was measured in: the seed maps reference
        # coordinates to moving sample coordinates, so undoing an image drift of +d in the
        # reference frame composes as pull @ T(-d). Both signs compete with the unchanged seed.
        pull = seed.invert().matrix
        candidates = {
            "keep": seed,
            "minus": Transform(pull @ translation_matrix(-drift)).invert(),
            "plus": Transform(pull @ translation_matrix(drift)).invert(),
        }
        nn = {name: nn_median(t) for name, t in candidates.items()}
        best = min(nn, key=nn.get)
        corrected = candidates[best]
        note = f"drift={np.round(drift, 1).tolist()} votes={n_votes} pick={best}"
        if settings.mode == "votefit" and len(pairs) >= 4:
            # The cluster's votes are correspondences (warped moving peak p, reference peak
            # q) in the uncorrected seed's warped frame; re-warping should put p at q, i.e.
            # sample point C(q) = p, so the fit composes onto the seed as pull @ C.
            p = np.asarray([a for a, _ in pairs], dtype=float)
            q = np.asarray([b for _, b in pairs], dtype=float)
            correction = (
                fit_affine(q, p)
                if len(pairs) >= 6
                else translation_matrix((p - q).mean(axis=0))
            )
            fitted = Transform(pull @ correction).invert()
            nn_fitted = nn_median(fitted)
            if np.isfinite(nn_fitted) and nn_fitted < nn[best]:
                corrected = fitted
                note += f"; fit accepted n={len(pairs)} nn {nn[best]:.1f}->{nn_fitted:.1f}"
            else:
                note += f"; fit rejected n={len(pairs)} nn {nn[best]:.1f}->{nn_fitted:.1f}"
        self.last_note = note
        return corrected


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

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        # PCC is correlation-based -- it finds the peak directly, no seed needed.
        mov = np.asarray(mov).astype(np.float32)
        ref = np.asarray(ref).astype(np.float32)
        if self.function_type == "custom_padding":
            shift, _corr = phase_cross_corr_padding(
                ref, mov, maximum_shift=self.maximum_shift, normalization=self.normalization
            )
        else:
            shift, _corr = phase_cross_corr(ref, mov, normalization=self.normalization)
        return Transform.from_translation(shift)


_ANTS_TRANSFORM_TYPE = {
    "euclidean": "Rigid",
    "rigid": "Rigid",
    "similarity": "Similarity",
    "affine": "Affine",
}


class AntsEstimator:
    """TransformEstimator using ANTs intensity-based optimization.

    One pass: pre-warp `mov` by the seed into `ref`'s frame, prepare both volumes
    (`ants.preprocess_zyx`: optional crop to their overlap, reference mask, clip, Sobel),
    register, and compose the correction back through the crop offset --
    `shift(+offset) @ correction @ shift(-offset) @ seed`. This is the legacy
    `ants.estimate_czyx` pipeline in the engine's forward (moving -> reference)
    convention.

    `ants.estimate()`'s `fwd_transform` is, despite its name, the reference -> moving
    ("pull") direction (see `tests/test_registration_estimators.py`); it is inverted here.
    """

    def __init__(
        self,
        ants_kwargs: dict | None = None,
        crop: bool = False,
        ref_mask_radius: float | None = None,
        clip: bool = False,
        sobel_filter: bool = False,
        verbose: bool = False,
    ):
        self.ants_kwargs = dict(ants_kwargs) if ants_kwargs else dict(DEFAULT_ANTS_KWARGS)
        self.crop = crop
        self.ref_mask_radius = ref_mask_radius
        self.clip = clip
        self.sobel_filter = sobel_filter
        self.verbose = verbose

    @classmethod
    def from_settings(
        cls,
        ants_registration_settings: AntsRegistrationSettings,
        affine_transform_settings: AffineTransformSettings,
        verbose: bool = False,
    ) -> AntsEstimator:
        """Preprocessing from the ANTs settings, transform family from the affine settings."""
        ants_kwargs = dict(DEFAULT_ANTS_KWARGS)
        ants_kwargs["type_of_transform"] = _ANTS_TRANSFORM_TYPE[
            affine_transform_settings.transform_type
        ]
        return cls(
            ants_kwargs=ants_kwargs,
            crop=ants_registration_settings.crop,
            ref_mask_radius=ants_registration_settings.ref_mask_radius,
            clip=ants_registration_settings.clip,
            sobel_filter=ants_registration_settings.sobel_filter,
            verbose=verbose,
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        aligned = seed.apply(mov, reference=ref) if seed is not None else mov
        ref_prepared, mov_prepared, offset = preprocess_zyx(
            aligned,
            ref,
            crop=self.crop,
            ref_mask_radius=self.ref_mask_radius,
            clip=self.clip,
            sobel_filter=self.sobel_filter,
        )
        pull_correction, _unused = ants_estimate(
            ref=ref_prepared,
            mov=mov_prepared,
            verbose=self.verbose,
            ants_kwargs=self.ants_kwargs,
        )
        correction = pull_correction.invert()
        if np.any(offset):
            correction = (
                Transform.from_translation(offset)
                @ correction
                @ Transform.from_translation(-offset)
            )
        return correction @ seed if seed is not None else correction


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

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        # No seed support -- napari display could pre-align with it, but that's not
        # implemented, and the point-fit itself doesn't take an initial guess.
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

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        # pystackreg's register() takes no initial guess -- correlation-based, like PCC.
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        sr = StackReg(self.transformation)
        xy_pull_matrix = np.asarray(sr.register(ref, mov))
        yx_matrix = self._AXIS_SWAP @ xy_pull_matrix @ self._AXIS_SWAP
        return Transform(matrix=yx_matrix).invert()
