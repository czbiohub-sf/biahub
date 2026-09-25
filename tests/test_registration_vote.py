import numpy as np
import pytest

from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform
from biahub.registration.beads import score_transform
from biahub.registration.estimators import (
    BeadNodeDetector,
    ChainedEstimator,
    EstimationError,
    NodeGraphEstimator,
    TransformEstimator,
    VoteIcpEstimator,
    VoteSeedCorrection,
)
from biahub.registration.pointcloud import vote_drift, vote_icp_register
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    SeedCorrectionSettings,
    VoteIcpSettings,
)


def test_vote_icp_register_recovers_a_large_offset_with_clutter():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 200, size=(20, 3))
    offset = np.array([30.0, -25.0, 40.0])
    mov = np.vstack([ref + offset, rng.uniform(0, 200, size=(6, 3))])  # + clutter

    pull, info = vote_icp_register(mov, ref, np.eye(4), initial_capture_radius=80.0)

    assert pull is not None and info["converged"]
    # pull maps reference coordinates to moving sample coordinates: ref + offset.
    np.testing.assert_allclose(pull[:3, 3], offset, atol=0.5)
    np.testing.assert_allclose(pull[:3, :3], np.eye(3), atol=0.02)


def test_vote_icp_register_abstains_without_votes():
    ref = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 50.0, 0.0]])
    mov = ref + 500.0  # far outside any capture radius
    pull, info = vote_icp_register(mov, ref, np.eye(4), initial_capture_radius=20.0)
    assert pull is None and info["iterations"] == 0


def test_vote_drift_finds_the_consensus_displacement():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 200, size=(15, 3))
    mov = np.vstack([ref - np.array([5.0, 7.0, -3.0]), rng.uniform(0, 200, size=(5, 3))])
    drift, pairs, n_votes = vote_drift(
        mov, ref, capture_radius=40.0, cluster_radius=3.0, min_votes=3
    )
    np.testing.assert_allclose(drift, [5.0, 7.0, -3.0], atol=0.5)
    assert len(pairs) >= 10 and n_votes >= 10


def _synthetic_bead_volume(rng, shape, n_beads=25, sigma=2.0, amplitude=500.0, noise_std=5.0):
    margin = 8
    centers = rng.uniform([margin] * 3, np.asarray(shape) - margin, size=(n_beads, 3))
    grid = np.indices(shape, dtype=float)
    volume = np.zeros(shape, dtype=np.float32)
    for c in centers:
        d2 = sum((g - ci) ** 2 for g, ci in zip(grid, c, strict=True))
        volume += (amplitude * np.exp(-d2 / (2 * sigma**2))).astype(np.float32)
    volume += rng.normal(0, noise_std, size=shape).astype(np.float32)
    return volume


PEAKS = DetectPeaksSettings(
    threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
)


def _beads_settings(**overrides):
    return BeadsMatchSettings(
        source_peaks_settings=PEAKS,
        target_peaks_settings=PEAKS,
        # Synthetic beads sit ~15 vox apart in a 60-vox field, so the vote cluster must be tight.
        vote_icp_settings=VoteIcpSettings(
            vote_peaks_settings=PEAKS, initial_capture_radius=40.0, cluster_radius=3.0
        ),
        **overrides,
    )


def _score(settings):
    return lambda transform, mov, ref: score_transform(transform, mov, ref, settings)


def test_vote_icp_estimator_recovers_an_offset_the_hungarian_matcher_cannot():
    rng = np.random.default_rng(11)
    shape = (40, 60, 60)
    ref = _synthetic_bead_volume(rng, shape)
    applied_zyx = (8, -12, 16)
    mov = ndi_shift(ref, shift=applied_zyx, order=1, mode="constant", cval=0.0)
    settings = _beads_settings()
    affine = AffineTransformSettings(transform_type="euclidean")

    with pytest.raises(EstimationError):
        NodeGraphEstimator(
            mov_detector=BeadNodeDetector(PEAKS),
            ref_detector=BeadNodeDetector(PEAKS),
            beads_match_settings=settings,
            affine_transform_settings=affine,
        ).estimate(mov, ref)

    estimator = VoteIcpEstimator.from_beads_settings(settings, score_fn=_score(settings))
    assert isinstance(estimator, TransformEstimator)
    coarse = estimator.estimate(mov, ref)
    # Voting acquires the bulk offset with a full affine on integer peak positions; the
    # matcher refines it, which is how the two are chained in practice.
    np.testing.assert_allclose(coarse.translation, [-a for a in applied_zyx], atol=2.0)

    refined = ChainedEstimator(
        [
            estimator,
            NodeGraphEstimator(
                mov_detector=BeadNodeDetector(PEAKS),
                ref_detector=BeadNodeDetector(PEAKS),
                beads_match_settings=settings,
                affine_transform_settings=affine,
            ),
        ],
        score_fn=_score(settings),
    ).estimate(mov, ref)
    truth = np.array([-a for a in applied_zyx], dtype=float)
    np.testing.assert_allclose(refined.translation, truth, atol=1.5)
    assert np.abs(refined.translation - truth).max() < np.abs(coarse.translation - truth).max()


def test_from_beads_settings_wires_vote_icp_mode_and_seed_correction():
    plain = NodeGraphEstimator.from_beads_settings(
        _beads_settings(), AffineTransformSettings()
    )
    vote = NodeGraphEstimator.from_beads_settings(
        _beads_settings(estimation_mode="vote_icp"), AffineTransformSettings()
    )
    corrected = NodeGraphEstimator.from_beads_settings(
        _beads_settings(seed_correction_settings=SeedCorrectionSettings(mode="votefit")),
        AffineTransformSettings(),
    )
    assert isinstance(plain, NodeGraphEstimator)
    assert isinstance(vote, VoteIcpEstimator)
    assert isinstance(corrected, ChainedEstimator)
    assert isinstance(corrected.stages[0], VoteSeedCorrection)
    assert isinstance(corrected.stages[1], NodeGraphEstimator)


def test_vote_seed_correction_moves_a_drifted_seed_toward_the_truth_and_never_worsens_it():
    rng = np.random.default_rng(11)
    shape = (40, 60, 60)
    ref = _synthetic_bead_volume(rng, shape)
    applied_zyx = (8, -12, 16)
    mov = ndi_shift(ref, shift=applied_zyx, order=1, mode="constant", cval=0.0)
    truth = Transform.from_translation([-a for a in applied_zyx])
    correction = VoteSeedCorrection(
        SeedCorrectionSettings(
            mode="votefit", vote_peaks_settings=PEAKS, capture_radius=40.0, cluster_radius=3.0
        ),
        mov_detector=BeadNodeDetector(PEAKS),
        ref_detector=BeadNodeDetector(PEAKS),
    )

    drifted = Transform.from_translation([0.0, 0.0, 0.0])
    corrected = correction.estimate(mov, ref, seed=drifted)
    assert (
        np.abs(corrected.translation - truth.translation).max()
        < np.abs(drifted.translation - truth.translation).max()
    )
    assert "pick=" in correction.last_note

    already_right = correction.estimate(mov, ref, seed=truth)
    np.testing.assert_allclose(already_right.translation, truth.translation, atol=1.0)


def test_chained_estimator_keeps_an_earlier_scored_stage_when_a_later_stage_fails():
    class _Fixed:
        def __init__(self, transform, fail=False):
            self.transform, self.fail = transform, fail

        def estimate(self, mov, ref, seed=None):
            if self.fail:
                raise EstimationError("no nodes")
            return self.transform

    first = Transform.from_translation([1.0, 0.0, 0.0])
    chain = ChainedEstimator(
        [_Fixed(first), _Fixed(None, fail=True)], score_fn=lambda t, m, r: 1.0
    )
    assert chain.estimate(np.zeros((2, 2, 2)), np.zeros((2, 2, 2))) is first
