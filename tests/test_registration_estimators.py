import numpy as np

from biahub.registration.beads import matches_from_beads, transform_from_matches
from biahub.registration.estimators import (
    BeadNodeDetector,
    NodeDetector,
    NodeGraphEstimator,
    TransformEstimator,
)
from biahub.settings import AffineTransformSettings, BeadsMatchSettings, DetectPeaksSettings


class _FixedNodeDetector:
    """Returns a canned point set regardless of the array -- isolates matching +
    transform fitting from real peak detection for tests."""

    def __init__(self, points: np.ndarray):
        self.points = points

    def detect(self, array):
        return self.points


def _make_matched_clouds(rng, n=30, translation=(5.0, -3.0, 2.0)):
    mov_nodes = rng.uniform(0, 100, size=(n, 3))
    ref_nodes = mov_nodes + np.asarray(translation)
    return mov_nodes, ref_nodes


def test_node_graph_estimator_satisfies_protocols():
    assert isinstance(BeadNodeDetector(DetectPeaksSettings()), NodeDetector)
    estimator = NodeGraphEstimator.from_beads_settings(
        BeadsMatchSettings(), AffineTransformSettings()
    )
    assert isinstance(estimator, TransformEstimator)


def test_node_graph_estimator_matches_direct_call():
    rng = np.random.default_rng(0)
    mov_nodes, ref_nodes = _make_matched_clouds(rng)

    beads_match_settings = BeadsMatchSettings()
    affine_transform_settings = AffineTransformSettings(transform_type="euclidean")

    estimator = NodeGraphEstimator(
        mov_detector=_FixedNodeDetector(mov_nodes),
        ref_detector=_FixedNodeDetector(ref_nodes),
        beads_match_settings=beads_match_settings,
        affine_transform_settings=affine_transform_settings,
    )
    mov_volume = np.zeros((10, 10, 10))
    ref_volume = np.zeros((10, 10, 10))
    transform = estimator.estimate(mov_volume, ref_volume)

    matches = matches_from_beads(mov_nodes, ref_nodes, beads_match_settings)
    expected_transform, _ = transform_from_matches(
        matches, mov_nodes, ref_nodes, affine_transform_settings, ndim=3
    )

    np.testing.assert_allclose(transform.matrix, expected_transform.matrix)


def test_node_graph_estimator_recovers_known_translation():
    rng = np.random.default_rng(1)
    translation = np.array([5.0, -3.0, 2.0])
    mov_nodes, ref_nodes = _make_matched_clouds(rng, translation=translation)

    estimator = NodeGraphEstimator(
        mov_detector=_FixedNodeDetector(mov_nodes),
        ref_detector=_FixedNodeDetector(ref_nodes),
        beads_match_settings=BeadsMatchSettings(),
        affine_transform_settings=AffineTransformSettings(transform_type="euclidean"),
    )
    transform = estimator.estimate(np.zeros((10, 10, 10)), np.zeros((10, 10, 10)))

    np.testing.assert_allclose(transform.matrix[:3, 3], translation, atol=1e-6)
    np.testing.assert_allclose(transform.matrix[:3, :3], np.eye(3), atol=1e-6)
