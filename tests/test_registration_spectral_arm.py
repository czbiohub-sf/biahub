import numpy as np
import pytest

from scipy.ndimage import shift as ndi_shift
from scipy.spatial.transform import Rotation

from biahub.core.graph_matching import Graph, GraphMatcher
from biahub.core.transform import Transform
from biahub.registration.estimators import (
    BeadNodeDetector,
    ChainedEstimator,
    CompetingEstimator,
    EstimationError,
    NodeGraphEstimator,
    TransformEstimator,
)
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    SpectralMatchSettings,
)

IDENTITY = Transform.identity(3)


def _correspondence_recall(matches, n_true):
    """Fraction of the first n_true moving points matched to their true partner (i == j)."""
    correct = sum(1 for i, j in matches if i == j and i < n_true)
    return correct / n_true


def test_spectral_matching_recovers_correspondence_under_rotation_and_large_shift():
    rng = np.random.default_rng(0)
    ref_nodes = rng.uniform(0, 100, size=(18, 3))
    rotation = Rotation.from_euler("z", 12, degrees=True).as_matrix()
    mov_nodes = (ref_nodes - 50) @ rotation.T + 50 + np.array([30.0, -25.0, 15.0])
    # Add unmatched clutter on both sides.
    mov_all = np.vstack([mov_nodes, rng.uniform(0, 100, size=(4, 3))])
    ref_all = np.vstack([ref_nodes, rng.uniform(0, 100, size=(4, 3))])

    spectral = GraphMatcher(algorithm="spectral", spectral_sigma=3.0, spectral_rel_cut=0.5)
    hungarian = GraphMatcher(algorithm="hungarian", cross_check=True)

    spectral_matches = spectral.match(Graph.from_nodes(mov_all), Graph.from_nodes(ref_all))
    hungarian_matches = hungarian.match(
        Graph.from_nodes(mov_all, mode="knn", k=6), Graph.from_nodes(ref_all, mode="knn", k=6)
    )

    spectral_recall = _correspondence_recall(spectral_matches, len(ref_nodes))
    hungarian_recall = _correspondence_recall(hungarian_matches, len(ref_nodes))
    assert spectral_recall >= 0.8
    # The Hungarian cost is position-dominated; a 40-voxel offset costs it most of the pairs.
    assert spectral_recall > hungarian_recall


def test_spectral_matching_refuses_an_affinity_matrix_it_cannot_hold():
    nodes = np.random.default_rng(1).uniform(0, 100, size=(250, 3))
    matcher = GraphMatcher(algorithm="spectral")
    with pytest.raises(ValueError, match="affinity matrix"):
        matcher.match(Graph.from_nodes(nodes), Graph.from_nodes(nodes))


class _FixedEstimator:
    def __init__(self, transform, fail=False):
        self.transform = transform
        self.fail = fail
        self.seeds = []

    def estimate(self, mov, ref, seed=None):
        self.seeds.append(seed)
        if self.fail:
            raise EstimationError("stage failed")
        return self.transform


def _closeness_to(target):
    def score_fn(transform, mov, ref):
        return -float(np.abs(transform.translation - target).sum())

    return score_fn


def test_chained_estimator_seeds_each_stage_from_the_previous_and_keeps_the_best_stage():
    coarse = _FixedEstimator(Transform.from_translation([9.0, 0.0, 0.0]))
    worse_refine = _FixedEstimator(Transform.from_translation([4.0, 0.0, 0.0]))
    chain = ChainedEstimator(
        [coarse, worse_refine], score_fn=_closeness_to(np.array([10.0, 0, 0]))
    )
    seed = Transform.from_translation([1.0, 1.0, 1.0])

    result = chain.estimate(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), seed=seed)

    assert coarse.seeds == [seed]
    assert worse_refine.seeds == [coarse.transform]
    assert result is coarse.transform, "a refinement that scores worse must be dropped"
    assert isinstance(chain, TransformEstimator)


def test_competing_estimator_keeps_the_best_arm_and_skips_failed_ones():
    target = np.array([10.0, 0.0, 0.0])
    arms = {
        "broken": _FixedEstimator(IDENTITY, fail=True),
        "far": _FixedEstimator(Transform.from_translation([2.0, 0.0, 0.0])),
        "close": _FixedEstimator(Transform.from_translation([9.0, 0.0, 0.0])),
    }
    competing = CompetingEstimator(arms, score_fn=_closeness_to(target))

    result = competing.estimate(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))

    assert result is arms["close"].transform
    assert competing.last_winner == "close"
    assert set(competing.last_scores) == {"far", "close"}


def test_competing_estimator_escalates_only_below_the_threshold():
    good = _FixedEstimator(Transform.from_translation([10.0, 0.0, 0.0]))
    second = _FixedEstimator(Transform.from_translation([10.0, 0.0, 0.0]))
    competing = CompetingEstimator(
        {"first": good, "second": second},
        score_fn=_closeness_to(np.array([10.0, 0, 0])),
        escalate_below=-1.0,
    )
    competing.estimate(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))
    assert second.seeds == [], "first arm scored 0.0 >= -1.0, so the second must not run"


def test_competing_estimator_fails_with_every_arms_reason_when_all_fail():
    competing = CompetingEstimator(
        {"a": _FixedEstimator(IDENTITY, fail=True), "b": _FixedEstimator(IDENTITY, fail=True)},
        score_fn=_closeness_to(np.zeros(3)),
    )
    with pytest.raises(EstimationError, match="a: stage failed; b: stage failed"):
        competing.estimate(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))


def _synthetic_bead_volume(rng, shape, n_beads=15, sigma=2.0, amplitude=500.0, noise_std=5.0):
    margin = 8
    centers = rng.uniform([margin] * 3, np.asarray(shape) - margin, size=(n_beads, 3))
    grid = np.indices(shape, dtype=float)
    volume = np.zeros(shape, dtype=np.float32)
    for c in centers:
        d2 = sum((g - ci) ** 2 for g, ci in zip(grid, c, strict=True))
        volume += (amplitude * np.exp(-d2 / (2 * sigma**2))).astype(np.float32)
    volume += rng.normal(0, noise_std, size=shape).astype(np.float32)
    return volume


def _beads_settings(**overrides):
    peaks = DetectPeaksSettings(
        threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    return BeadsMatchSettings(
        source_peaks_settings=peaks,
        target_peaks_settings=peaks,
        spectral_match_settings=SpectralMatchSettings(),
        **overrides,
    )


def test_from_beads_settings_builds_the_competing_cascade_when_the_spectral_arm_is_on():
    plain = NodeGraphEstimator.from_beads_settings(
        _beads_settings(), AffineTransformSettings()
    )
    competing = NodeGraphEstimator.from_beads_settings(
        _beads_settings(spectral_arm="always"), AffineTransformSettings()
    )
    gated = NodeGraphEstimator.from_beads_settings(
        _beads_settings(spectral_arm="on_low_score"), AffineTransformSettings()
    )

    assert isinstance(plain, NodeGraphEstimator)
    assert isinstance(competing, CompetingEstimator) and list(competing.arms) == [
        "hungarian",
        "spectral+hungarian",
    ]
    assert isinstance(competing.arms["spectral+hungarian"], ChainedEstimator)
    assert competing.escalate_below is None
    assert gated.escalate_below == _beads_settings().qc_settings.score_threshold


def test_spectral_arm_recovers_an_offset_beyond_the_hungarian_capture_range():
    """An offset the Hungarian matcher alone cannot match from an identity seed is
    recovered once the spectral arm acquires the correspondence first."""
    rng = np.random.default_rng(11)
    shape = (40, 60, 60)
    ref = _synthetic_bead_volume(rng, shape, n_beads=30)
    applied_zyx = (8, -12, 16)
    mov = ndi_shift(ref, shift=applied_zyx, order=1, mode="constant", cval=0.0)
    settings = _beads_settings(spectral_arm="always")
    affine = AffineTransformSettings(transform_type="euclidean")

    hungarian_only = NodeGraphEstimator(
        mov_detector=BeadNodeDetector(settings.source_peaks_settings),
        ref_detector=BeadNodeDetector(settings.target_peaks_settings),
        beads_match_settings=settings,
        affine_transform_settings=affine,
    )
    with pytest.raises(EstimationError):
        hungarian_only.estimate(mov, ref)

    competing = NodeGraphEstimator.from_beads_settings(settings, affine)
    transform = competing.estimate(mov, ref)

    assert competing.last_winner == "spectral+hungarian"
    # Capture-range test, not a precision test: on integer peak positions the matcher's
    # fit lands within ~1 vox (the same precision the legacy pipeline has).
    np.testing.assert_allclose(transform.translation, [-a for a in applied_zyx], atol=1.5)
