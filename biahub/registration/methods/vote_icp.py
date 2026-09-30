"""Point-cloud registration primitives for sparse bead fields.

Numpy/scipy algorithms over (N, 3) ZYX peak coordinates, kept apart from the volume code
in `methods.beads` so each stage is testable on its own. Returned transforms are
homogeneous (D+1, D+1) matrices in the pull convention used by `vote_icp_register`'s
`initial_transform`: they map reference-space points to moving-space sample coordinates.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from biahub.core.transform import Transform
from biahub.registration.estimators import (
    EstimationError,
    NodeDetector,
    ScoreFn,
)
from biahub.registration.methods.beads import BeadNodeDetector
from biahub.settings import BeadsMatchSettings, SeedCorrectionSettings, VoteIcpSettings


def translation_matrix(displacement: ArrayLike) -> np.ndarray:
    matrix = np.eye(len(displacement) + 1)
    matrix[:-1, -1] = displacement
    return matrix


def fit_affine(src: ArrayLike, dst: ArrayLike) -> np.ndarray:
    """Least-squares affine C with C(src_i) ~ dst_i, as a homogeneous matrix (N >= D + 1)."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    ndim = src.shape[1]
    design = np.hstack([src, np.ones((len(src), 1))])
    coeffs, *_ = np.linalg.lstsq(design, dst, rcond=None)
    matrix = np.eye(ndim + 1)
    matrix[:ndim, :ndim] = coeffs[:ndim].T
    matrix[:ndim, ndim] = coeffs[ndim]
    return matrix


def vote_icp_register(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    initial_transform: ArrayLike,
    initial_capture_radius: float = 80.0,
    min_capture_radius: float = 10.0,
    radius_decay: float = 0.5,
    cluster_radius: float = 10.0,
    min_votes: int = 3,
    max_iterations: int = 20,
    convergence_translation: float = 0.5,
) -> tuple[np.ndarray | None, dict]:
    """Register two peak clouds by iterated displacement voting.

    An ICP whose correspondence stage is a vote: each iteration maps the moving peaks
    into reference space with the current transform, lets every mapped peak vote for
    its displacement to each reference peak within the capture radius, keeps the densest
    `cluster_radius`-ball of votes as the inlier correspondences and re-fits on them (a
    full affine with >= D + 3 pairs, a pure translation otherwise). The capture radius
    shrinks geometrically, floored at `min_capture_radius`, so early iterations have the
    reach to recover a large residual and late ones only see unambiguous neighbours.

    Unlike one-to-one matching the vote never commits a peak to a single partner, so a
    split or fuzzy detection just casts extra votes outside the winning cluster.

    `initial_transform` and the returned matrix map reference-space points to
    moving-space sample coordinates. Returns `(None, info)` when voting abstained before
    any fit; `info` holds iterations, converged, n_inliers and the final capture radius.
    """
    mov_peaks = np.asarray(mov_peaks, dtype=float)
    ref_peaks = np.asarray(ref_peaks, dtype=float)
    matrix = np.asarray(initial_transform, dtype=float)
    ref_tree = cKDTree(ref_peaks)
    ndim = mov_peaks.shape[1]

    def map_to_ref(m):
        inverse = np.linalg.inv(m)
        return mov_peaks @ inverse[:ndim, :ndim].T + inverse[:ndim, ndim]

    mapped = map_to_ref(matrix)
    radius = initial_capture_radius
    info = {"iterations": 0, "converged": False, "n_inliers": 0, "capture_radius": radius}
    fitted_once = False
    for iteration in range(1, max_iterations + 1):
        votes, pair_indices = [], []
        for i, p in enumerate(mapped):
            for j in ref_tree.query_ball_point(p, r=radius):
                votes.append(ref_peaks[j] - p)
                pair_indices.append((i, j))
        votes = np.asarray(votes)
        if len(votes) < min_votes:
            break
        vote_tree = cKDTree(votes)
        counts = np.array(
            [len(vote_tree.query_ball_point(v, r=cluster_radius)) for v in votes]
        )
        members = vote_tree.query_ball_point(votes[np.argmax(counts)], r=cluster_radius)
        inlier_mov = np.array([pair_indices[m][0] for m in members])
        inlier_ref = np.array([pair_indices[m][1] for m in members])
        # A full affine needs a margin over its degrees of freedom; with fewer inliers the
        # translation update is the only fit that cannot hallucinate.
        if len(members) >= ndim + 3:
            new_matrix = fit_affine(ref_peaks[inlier_ref], mov_peaks[inlier_mov])
        else:
            drift = votes[members].mean(axis=0)
            new_matrix = matrix @ translation_matrix(-drift)
        new_mapped = map_to_ref(new_matrix)
        step = float(np.mean(np.linalg.norm(new_mapped - mapped, axis=1)))
        matrix, mapped, fitted_once = new_matrix, new_mapped, True
        info.update(iterations=iteration, n_inliers=int(len(members)), capture_radius=radius)
        radius = max(min_capture_radius, radius * radius_decay)
        if step < convergence_translation:
            info["converged"] = True
            break
    if not fitted_once:
        return None, info
    return matrix, info


def vote_drift(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    capture_radius: float,
    cluster_radius: float,
    min_votes: int,
) -> tuple[np.ndarray | None, list[tuple[np.ndarray, np.ndarray]], int]:
    """One round of displacement voting between already-aligned peak clouds.

    Returns the mean displacement of the densest vote cluster, that cluster's
    (moving peak, reference peak) pairs, and the cluster size -- or `(None, [], 0)`
    when fewer than `min_votes` votes were cast.
    """
    mov_peaks = np.asarray(mov_peaks, dtype=float)
    ref_peaks = np.asarray(ref_peaks, dtype=float)
    ref_tree = cKDTree(ref_peaks)
    votes, pairs = [], []
    for p in mov_peaks:
        for j in ref_tree.query_ball_point(p, r=capture_radius):
            votes.append(ref_peaks[j] - p)
            pairs.append((p, ref_peaks[j]))
    if len(votes) < min_votes:
        return None, [], 0
    votes = np.asarray(votes)
    vote_tree = cKDTree(votes)
    counts = np.array([len(vote_tree.query_ball_point(v, r=cluster_radius)) for v in votes])
    members = vote_tree.query_ball_point(votes[np.argmax(counts)], r=cluster_radius)
    return votes[members].mean(axis=0), [pairs[m] for m in members], int(counts.max())


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
