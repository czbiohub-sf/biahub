import numpy as np

from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform
from biahub.registration.beads import matches_from_beads, transform_from_matches
from biahub.registration.estimators import (
    AntsEstimator,
    BeadNodeDetector,
    ManualEstimator,
    NodeDetector,
    NodeGraphEstimator,
    PCCEstimator,
    StackregEstimator,
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


def test_pcc_estimator_satisfies_protocol():
    assert isinstance(PCCEstimator(), TransformEstimator)


def test_pcc_estimator_recovers_known_translation_and_warps_mov_onto_ref():
    rng = np.random.default_rng(2)
    shape = (30, 40, 50)
    ref = rng.random(shape).astype(np.float32)

    applied_zyx = (4, -6, 9)  # distinct per-axis values so an axis mixup would show
    mov = ndi_shift(ref, shift=applied_zyx, order=0, mode="constant", cval=0.0)

    transform = PCCEstimator().estimate(mov, ref)

    # transform is forward (moving -> reference): applying it to mov via the scipy
    # backend should warp mov's content back onto ref.
    margin = int(max(abs(a) for a in applied_zyx)) + 1
    interior = tuple(slice(margin, s - margin) for s in shape)
    warped = transform.apply(mov, reference=ref, order=0, backend="scipy")
    np.testing.assert_allclose(warped[interior], ref[interior], atol=1e-3)


def _synthetic_blob_volume(rng, shape, n_blobs=15, sigma=3.0, noise_std=5.0):
    """Sharp Gaussian blobs (fake beads), not random noise -- ANTs' intensity-based
    optimizer needs real structure to converge on; blurred noise wasn't enough."""
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    margin = int(sigma * 3)
    centers = rng.uniform([margin] * 3, np.asarray(shape) - margin, size=(n_blobs, 3))
    volume = np.zeros(shape, dtype=np.float32)
    for cz, cy, cx in centers:
        volume += 500 * np.exp(
            -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        )
    volume += rng.normal(0, noise_std, size=shape).astype(np.float32)
    return volume


def test_ants_estimator_satisfies_protocol():
    assert isinstance(AntsEstimator(), TransformEstimator)


def test_ants_estimator_seed_is_mechanically_used():
    """Regression test: seeds must be inverted (forward -> pull) before being handed to
    ants.registration's initial_transform, and passing a seed must not silently fall
    back to ants.registration's own default (SyN, deformable -- Transform.from_ants
    can't parse it), which happened once already when ants_kwargs was built as `{}`
    instead of preserving DEFAULT_ANTS_KWARGS.

    With near-zero optimizer iterations, the result should be (near-)indistinguishable
    from the seed itself, per the interactive verification this is based on.
    """
    rng = np.random.default_rng(10)
    shape = (24, 48, 48)
    ref = _synthetic_blob_volume(rng, shape)
    mov = ndi_shift(ref, shift=(2, -3, 4), order=1, mode="constant", cval=0.0)

    huge_seed = Transform.from_translation([15.0, -10.0, 8.0])  # nowhere near the truth
    estimator = AntsEstimator(
        ants_kwargs={
            "type_of_transform": "Similarity",
            "aff_iterations": (1, 1, 1),
            "aff_shrink_factors": (6, 3, 1),
            "aff_smoothing_sigmas": (2, 1, 0),
        }
    )
    result = estimator.estimate(mov, ref, seed=huge_seed)
    # 1 iteration/level still moves slightly -- what matters is the result tracks the
    # seed (~15, ~-10, ~8), not the true answer (-2, 3, -4).
    np.testing.assert_allclose(result.matrix[:3, 3], huge_seed.matrix[:3, 3], atol=1.0)


def test_ants_estimator_with_seed_still_recovers_translation():
    rng = np.random.default_rng(5)
    shape = (24, 48, 48)
    ref = _synthetic_blob_volume(rng, shape)
    applied_zyx = (2, -3, 4)
    mov = ndi_shift(ref, shift=applied_zyx, order=1, mode="constant", cval=0.0)

    close_seed = Transform.from_translation([-a for a in applied_zyx])
    transform = AntsEstimator().estimate(mov, ref, seed=close_seed)
    np.testing.assert_allclose(transform.matrix[:3, 3], [-a for a in applied_zyx], atol=0.5)


def test_ants_estimator_recovers_known_translation_and_warps_mov_onto_ref():
    rng = np.random.default_rng(5)
    shape = (24, 48, 48)
    ref = _synthetic_blob_volume(rng, shape)

    applied_zyx = (2, -3, 4)
    mov = ndi_shift(ref, shift=applied_zyx, order=1, mode="constant", cval=0.0)

    transform = AntsEstimator().estimate(mov, ref)
    np.testing.assert_allclose(transform.matrix[:3, 3], [-a for a in applied_zyx], atol=0.5)

    margin = 6
    interior = tuple(slice(margin, s - margin) for s in shape)
    warped = transform.apply(mov, reference=ref, order=1, backend="scipy")

    def relerr(a, b):
        return np.abs(a[interior] - b[interior]).mean() / (np.abs(b[interior]).mean() + 1e-8)

    assert relerr(warped, ref) < relerr(mov, ref)
    assert relerr(warped, ref) < 0.05


def test_manual_estimator_satisfies_protocol():
    estimator = ManualEstimator(
        source_channel_name="GFP",
        target_channel_name="Phase3D",
        source_channel_voxel_size=(1.0, 1.0, 1.0),
        target_channel_voxel_size=(1.0, 1.0, 1.0),
    )
    assert isinstance(estimator, TransformEstimator)


def test_manual_estimator_inverts_user_assisted_registrations_pull_output(monkeypatch):
    # user_assisted_registration returns a pull (reference -> moving) matrix, per its own
    # explicit internal .invert() before returning -- see estimators.py's docstring.
    pull_matrix = np.eye(4)
    pull_matrix[:3, 3] = [1.0, 2.0, 3.0]
    captured_kwargs = {}

    def fake_user_assisted_registration(**kwargs):
        captured_kwargs.update(kwargs)
        return [pull_matrix.tolist()]

    monkeypatch.setattr(
        "biahub.registration.estimators.user_assisted_registration",
        fake_user_assisted_registration,
    )

    mov = np.zeros((5, 5, 5))
    ref = np.zeros((5, 5, 5))
    estimator = ManualEstimator(
        source_channel_name="GFP",
        target_channel_name="Phase3D",
        source_channel_voxel_size=(1.0, 1.0, 1.0),
        target_channel_voxel_size=(0.5, 0.5, 0.5),
        similarity=True,
    )
    transform = estimator.estimate(mov, ref)

    np.testing.assert_allclose(transform.matrix, np.linalg.inv(pull_matrix))
    assert captured_kwargs["source_channel_name"] == "GFP"
    assert captured_kwargs["target_channel_name"] == "Phase3D"
    assert captured_kwargs["target_channel_voxel_size"] == (0.5, 0.5, 0.5)
    assert captured_kwargs["similarity"] is True
    np.testing.assert_array_equal(captured_kwargs["source_channel_volume"], mov)
    np.testing.assert_array_equal(captured_kwargs["target_channel_volume"], ref)


def _synthetic_blob_image(rng, shape, n_blobs=10, sigma=3.0, noise_std=5.0):
    yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    margin = int(sigma * 3)
    centers = rng.uniform([margin] * 2, np.asarray(shape) - margin, size=(n_blobs, 2))
    image = np.zeros(shape, dtype=np.float32)
    for cy, cx in centers:
        image += 500 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
    image += rng.normal(0, noise_std, size=shape).astype(np.float32)
    return image


def test_stackreg_estimator_satisfies_protocol():
    assert isinstance(StackregEstimator(), TransformEstimator)


def test_stackreg_estimator_recovers_known_translation_and_warps_mov_onto_ref():
    rng = np.random.default_rng(6)
    shape = (48, 48)
    ref = _synthetic_blob_image(rng, shape)

    applied_yx = (3, -5)  # distinct per-axis values so an axis mixup would show
    mov = ndi_shift(ref, shift=applied_yx, order=1, mode="constant", cval=0.0)

    transform = StackregEstimator().estimate(mov, ref)
    np.testing.assert_allclose(transform.matrix[:2, 2], [-a for a in applied_yx], atol=0.5)

    margin = 8
    interior = tuple(slice(margin, s - margin) for s in shape)
    warped = transform.apply(mov, reference=ref, order=1, backend="scipy")

    def relerr(a, b):
        return np.abs(a[interior] - b[interior]).mean() / (np.abs(b[interior]).mean() + 1e-8)

    assert relerr(warped, ref) < relerr(mov, ref)
    assert relerr(warped, ref) < 0.05
