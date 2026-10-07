import os

from pathlib import Path

import numpy as np
import pytest

from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform
from biahub.registration.estimators import EstimationError, NodeDetector, TransformEstimator
from biahub.registration.methods.ants import DEFAULT_ANTS_KWARGS, AntsEstimator
from biahub.registration.methods.beads import (
    BeadNodeDetector,
    NodeGraphEstimator,
    matches_from_beads,
    transform_from_matches,
)
from biahub.registration.methods.focus import FocusEstimator, find_focus
from biahub.registration.methods.manual import ManualEstimator
from biahub.registration.methods.pcc import PCCEstimator
from biahub.registration.methods.stackreg import StackregEstimator
from biahub.registration.metrics import correlation_score
from biahub.settings import (
    AffineTransformSettings,
    AntsRegistrationSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    FocusSettings,
)


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


def test_node_graph_estimator_keeps_the_best_scoring_iteration_not_the_last():
    rng = np.random.default_rng(3)
    translation = np.array([5.0, -3.0, 2.0])
    mov_nodes, ref_nodes = _make_matched_clouds(rng, translation=translation)
    # Canned detectors return the same nodes every pass, so pass 2 re-applies the same
    # correction on top of pass 1 and drifts to ~2x the true translation -- exactly the
    # "later iteration is worse" case keep-best must reject.
    scored = []

    def score_fn(transform, mov, ref):
        score = -float(np.abs(transform.translation - translation).sum())
        scored.append(score)
        return score

    estimator = NodeGraphEstimator(
        mov_detector=_FixedNodeDetector(mov_nodes),
        ref_detector=_FixedNodeDetector(ref_nodes),
        beads_match_settings=BeadsMatchSettings(),
        affine_transform_settings=AffineTransformSettings(transform_type="euclidean"),
        iterations=2,
        score_fn=score_fn,
    )
    transform = estimator.estimate(np.zeros((10, 10, 10)), np.zeros((10, 10, 10)))

    # pass-1 refinement, then its input (scored once), then the worse pass-2 refinement
    assert len(scored) == 3 and scored[0] > scored[2]
    np.testing.assert_allclose(transform.translation, translation, atol=1e-6)


def _drifting_estimator(translation, score_fn, iterations):
    """Canned nodes: every pass re-applies the same correction (drifts by `translation`)."""
    rng = np.random.default_rng(3)
    mov_nodes, ref_nodes = _make_matched_clouds(rng, translation=translation)
    return NodeGraphEstimator(
        mov_detector=_FixedNodeDetector(mov_nodes),
        ref_detector=_FixedNodeDetector(ref_nodes),
        beads_match_settings=BeadsMatchSettings(),
        affine_transform_settings=AffineTransformSettings(transform_type="euclidean"),
        iterations=iterations,
        score_fn=score_fn,
    )


def test_node_graph_estimator_keeps_its_input_when_the_refinement_scores_lower():
    # The seed is already right; the pass's refinement drifts away and scores lower, so
    # (as in the legacy pipeline) the pass keeps its input.
    translation = np.array([5.0, -3.0, 2.0])
    seed = Transform.from_translation(translation)

    def score_fn(transform, mov, ref):
        return -float(np.abs(transform.translation - translation).sum())

    estimator = _drifting_estimator(translation, score_fn, iterations=1)
    transform = estimator.estimate(np.zeros((10, 10, 10)), np.zeros((10, 10, 10)), seed=seed)
    np.testing.assert_allclose(transform.matrix, seed.matrix)


def test_node_graph_estimator_stops_at_a_perfect_score():
    passes = []
    estimator = _drifting_estimator(
        np.array([1.0, 0.0, 0.0]), lambda t, m, r: 1.0, iterations=3
    )
    original = estimator._single_pass

    def counting_pass(*args, **kwargs):
        passes.append(1)
        return original(*args, **kwargs)

    estimator._single_pass = counting_pass
    estimator.estimate(np.zeros((10, 10, 10)), np.zeros((10, 10, 10)))
    assert len(passes) == 1


def test_node_graph_estimator_fails_clearly_with_too_few_nodes():
    estimator = NodeGraphEstimator(
        mov_detector=_FixedNodeDetector(np.zeros((2, 3))),
        ref_detector=_FixedNodeDetector(np.zeros((30, 3))),
        beads_match_settings=BeadsMatchSettings(),
        affine_transform_settings=AffineTransformSettings(),
    )
    with pytest.raises(EstimationError, match="2 moving, 30 reference"):
        estimator.estimate(np.zeros((10, 10, 10)), np.zeros((10, 10, 10)))


def test_node_graph_estimator_fails_clearly_when_no_iteration_scores():
    rng = np.random.default_rng(4)
    mov_nodes, ref_nodes = _make_matched_clouds(rng)
    estimator = NodeGraphEstimator(
        mov_detector=_FixedNodeDetector(mov_nodes),
        ref_detector=_FixedNodeDetector(ref_nodes),
        beads_match_settings=BeadsMatchSettings(),
        affine_transform_settings=AffineTransformSettings(transform_type="euclidean"),
        iterations=2,
        score_fn=lambda transform, mov, ref: float("nan"),
    )
    with pytest.raises(EstimationError, match="no finite score in 2 iteration"):
        estimator.estimate(np.zeros((10, 10, 10)), np.zeros((10, 10, 10)))


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


def _synthetic_bead_volume(rng, shape, n_beads=15, sigma=2.0, amplitude=500, noise_std=5.0):
    """Sharp, sparse peaks -- what BeadNodeDetector's peak detection actually needs,
    as opposed to _synthetic_blob_volume's broader blobs (fine for ants/pcc/stackreg,
    too broad for peak-based detection here)."""
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    margin = int(sigma * 4)
    centers = rng.uniform([margin] * 3, np.asarray(shape) - margin, size=(n_beads, 3))
    volume = np.zeros(shape, dtype=np.float32)
    for cz, cy, cx in centers:
        volume += amplitude * np.exp(
            -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        )
    volume += rng.normal(0, noise_std, size=shape).astype(np.float32)
    return np.clip(volume, 0, None)


def _bead_estimator():
    peaks_settings = DetectPeaksSettings(
        threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    beads_match_settings = BeadsMatchSettings(
        source_peaks_settings=peaks_settings, target_peaks_settings=peaks_settings
    )
    return NodeGraphEstimator(
        mov_detector=BeadNodeDetector(peaks_settings),
        ref_detector=BeadNodeDetector(peaks_settings),
        beads_match_settings=beads_match_settings,
        affine_transform_settings=AffineTransformSettings(transform_type="euclidean"),
    )


def test_node_graph_estimator_seed_composition_is_self_consistent():
    """Regression test for the seed composition math: pre-warping mov by the seed and
    composing the residual correction with it (`correction @ seed`) must give the exact
    same result as manually pre-warping mov, fitting the residual with no seed, and
    composing by hand -- confirms Transform.compose's "apply other first, then self"
    contract is used in the right order here.
    """
    rng = np.random.default_rng(11)
    shape = (40, 60, 60)
    ref = _synthetic_bead_volume(rng, shape)
    applied_zyx = (2, -3, 4)
    mov = ndi_shift(ref, shift=applied_zyx, order=1, mode="constant", cval=0.0)

    estimator = _bead_estimator()
    seed = Transform.from_translation([-1.5, 2.5, -3.5])  # plausible, not exact

    mov_warped = seed.apply(mov, reference=ref, order=1, backend="scipy")
    correction = estimator.estimate(mov_warped, ref)
    expected_total = correction @ seed

    actual_total = estimator.estimate(mov, ref, seed=seed)
    np.testing.assert_allclose(actual_total.matrix, expected_total.matrix, atol=1e-6)


def test_node_graph_estimator_seed_extends_matching_capture_range():
    """A large offset that cannot be matched without a seed (too few valid
    correspondences -> EstimationError, not a silent NaN matrix) succeeds exactly once a
    seed pre-aligns mov close enough for point matching's limited capture range.
    """
    rng = np.random.default_rng(11)
    shape = (40, 60, 60)
    ref = _synthetic_bead_volume(rng, shape)
    big_applied_zyx = (10, -15, 20)
    mov = ndi_shift(ref, shift=big_applied_zyx, order=1, mode="constant", cval=0.0)

    estimator = _bead_estimator()

    with pytest.raises(EstimationError, match="too few matches"):
        estimator.estimate(mov, ref)

    good_seed = Transform.from_translation([-a for a in big_applied_zyx])
    with_seed_result = estimator.estimate(mov, ref, seed=good_seed)
    np.testing.assert_allclose(
        with_seed_result.matrix[:3, 3], [-a for a in big_applied_zyx], atol=0.5
    )


def test_pcc_estimator_satisfies_protocol():
    assert isinstance(PCCEstimator(), TransformEstimator)


@pytest.mark.parametrize("function_type", ["custom", "custom_padding"])
def test_pcc_estimator_recovers_known_translation_and_warps_mov_onto_ref(function_type):
    rng = np.random.default_rng(2)
    shape = (30, 40, 50)
    ref = rng.random(shape).astype(np.float32)

    applied_zyx = (4, -6, 9)  # distinct per-axis values so an axis mixup would show
    mov = ndi_shift(ref, shift=applied_zyx, order=0, mode="constant", cval=0.0)

    transform = PCCEstimator(function_type=function_type).estimate(mov, ref)

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


def test_ants_estimator_from_settings_maps_transform_type_and_preprocessing():
    estimator = AntsEstimator.from_settings(
        AntsRegistrationSettings(sobel_filter=True, crop=True, ref_mask_radius=0.9),
        AffineTransformSettings(transform_type="affine"),
    )
    assert estimator.ants_kwargs["type_of_transform"] == "Affine"
    assert estimator.ants_kwargs["aff_iterations"] == DEFAULT_ANTS_KWARGS["aff_iterations"]
    assert (
        estimator.sobel_filter,
        estimator.crop,
        estimator.ref_mask_radius,
        estimator.clip,
    ) == (
        True,
        True,
        0.9,
        False,
    )
    assert isinstance(estimator, TransformEstimator)


def test_ants_estimator_composes_the_crop_offset_back_into_full_volume_coordinates():
    """With `crop`, ANTs sees a sub-volume; the correction it returns is in that
    sub-volume's coordinates and must be shifted back, or the result is off by the crop
    origin."""
    rng = np.random.default_rng(7)
    shape = (24, 48, 48)
    ref = _synthetic_blob_volume(rng, shape)
    applied_zyx = (2, -3, 4)
    mov = ndi_shift(ref, shift=applied_zyx, order=1, mode="constant", cval=0.0)
    # Zero out a border so the overlap crop has a non-zero origin.
    ref[:, :8, :] = 0
    mov[:, :8, :] = 0

    transform = AntsEstimator(crop=True).estimate(mov, ref)
    np.testing.assert_allclose(transform.matrix[:3, 3], [-a for a in applied_zyx], atol=0.5)


def test_correlation_score_prefers_the_correct_transform():
    rng = np.random.default_rng(8)
    shape = (24, 48, 48)
    ref = _synthetic_blob_volume(rng, shape)
    applied_zyx = (2, -3, 4)
    mov = ndi_shift(ref, shift=applied_zyx, order=1, mode="constant", cval=0.0)

    right = correlation_score(Transform.from_translation([-a for a in applied_zyx]), mov, ref)
    wrong = correlation_score(Transform.from_translation([5.0, 5.0, 5.0]), mov, ref)
    perfect = correlation_score(Transform.identity(3), ref, ref)

    assert perfect == pytest.approx(1.0)
    assert right > 0.9 > wrong
    assert correlation_score(
        Transform.identity(3), ref, ref, sobel_filter=True
    ) == pytest.approx(1.0)


def test_ants_estimator_seed_is_mechanically_used():
    """With a correct seed and near-zero optimizer iterations the correction is
    ~identity, so the result must track the seed -- proving the seed pre-warp and the
    `correction @ seed` composition are wired, and that the pass runs an invertible
    transform family (not ants.registration's default SyN, which Transform.from_ants
    cannot parse). A wildly wrong seed must still return a finite transform."""
    rng = np.random.default_rng(10)
    shape = (24, 48, 48)
    ref = _synthetic_blob_volume(rng, shape)
    applied_zyx = (2, -3, 4)
    mov = ndi_shift(ref, shift=applied_zyx, order=1, mode="constant", cval=0.0)

    estimator = AntsEstimator(
        ants_kwargs={
            "type_of_transform": "Similarity",
            "aff_iterations": (1, 1, 1),
            "aff_shrink_factors": (6, 3, 1),
            "aff_smoothing_sigmas": (2, 1, 0),
        }
    )
    true_seed = Transform.from_translation([-a for a in applied_zyx])
    result = estimator.estimate(mov, ref, seed=true_seed)
    np.testing.assert_allclose(result.matrix[:3, 3], true_seed.matrix[:3, 3], atol=1.0)

    huge_seed = Transform.from_translation([15.0, -10.0, 8.0])
    assert np.all(np.isfinite(estimator.estimate(mov, ref, seed=huge_seed).matrix))


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


def test_manual_estimator_inverts_user_assisted_registrations_inverse_output(monkeypatch):
    # user_assisted_registration returns a inverse (reference -> moving) matrix, per its own
    # explicit internal .invert() before returning -- see estimators.py's docstring.
    inverse_matrix = np.eye(4)
    inverse_matrix[:3, 3] = [1.0, 2.0, 3.0]
    captured_kwargs = {}

    def fake_user_assisted_registration(**kwargs):
        captured_kwargs.update(kwargs)
        return [inverse_matrix.tolist()]

    monkeypatch.setattr(
        "biahub.registration.methods.manual.user_assisted_registration",
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

    np.testing.assert_allclose(transform.matrix, np.linalg.inv(inverse_matrix))
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


def _synthetic_focus_volume(rng, shape, z_focus, yx_shift=(0, 0)):
    """A textured plane in focus at `z_focus`, blurrier the further a slice is from it."""
    from scipy.ndimage import gaussian_filter

    z, y, x = shape
    # Blobs, not white noise: pystackreg's pyramid needs coarse structure to lock onto.
    plane = ndi_shift(_synthetic_blob_image(rng, (y, x)), shift=yx_shift, order=1, mode="wrap")
    return np.stack(
        [gaussian_filter(plane, sigma=0.4 * abs(k - z_focus) + 1e-3) for k in range(z)]
    ).astype(np.float32)


FOCUS_PIXEL_SIZE = 6.5 / 40  # micrometres, a 40x mantis-like sampling


def test_focus_estimator_satisfies_protocol():
    assert isinstance(FocusEstimator(pixel_size=FOCUS_PIXEL_SIZE), TransformEstimator)


def test_find_focus_picks_the_sharpest_slice_and_rejects_empty_volumes():
    rng = np.random.default_rng(3)
    assert find_focus(_synthetic_focus_volume(rng, (12, 64, 64), 7), FOCUS_PIXEL_SIZE) == 7
    with pytest.raises(EstimationError, match="empty"):
        find_focus(np.zeros((12, 64, 64), dtype=np.float32), FOCUS_PIXEL_SIZE)


@pytest.mark.parametrize(
    "axes, expected_mask", [("z", (1, 0, 0)), ("xy", (0, 1, 1)), ("xyz", (1, 1, 1))]
)
def test_focus_estimator_recovers_z_and_yx_drift_on_the_selected_axes(axes, expected_mask):
    rng = np.random.default_rng(4)
    shape = (12, 64, 64)
    ref = _synthetic_focus_volume(rng, shape, z_focus=5)
    applied_z, applied_yx = 3, (2, -4)
    mov = _synthetic_focus_volume(
        np.random.default_rng(4), shape, z_focus=5 + applied_z, yx_shift=applied_yx
    )

    transform = FocusEstimator(
        pixel_size=FOCUS_PIXEL_SIZE, axes=axes, center_crop_xy=(48, 48)
    ).estimate(mov, ref)

    truth = np.array([-applied_z, -applied_yx[0], -applied_yx[1]], dtype=float)
    np.testing.assert_allclose(
        transform.translation, truth * np.array(expected_mask), atol=0.5
    )
    np.testing.assert_array_equal(transform.linear, np.eye(3))


def test_focus_estimator_from_settings_reads_axes_crop_and_optics():
    settings = FocusSettings(axes="z", center_crop_xy=[100, 200], na_det=1.2, lambda_ill=0.45)
    estimator = FocusEstimator.from_settings(settings, pixel_size=FOCUS_PIXEL_SIZE)
    assert (estimator.axes, estimator.center_crop_xy) == ("z", (100, 200))
    assert (estimator.na_det, estimator.lambda_ill) == (1.2, 0.45)


def _identity_pass_estimator(score_of, fail_from=()):
    """A NodeGraphEstimator whose pass returns its start unchanged (or fails from a start)."""
    estimator = _drifting_estimator(np.zeros(3), lambda t, m, r: score_of(t), iterations=1)

    def single_pass(mov, ref, start):
        if any(start is f for f in fail_from):
            raise EstimationError("too few matches")
        return start

    estimator._single_pass = single_pass
    return estimator


def test_a_competitor_seed_wins_only_when_it_scores_strictly_higher():
    seed, competitor = (
        Transform.from_translation([1, 0, 0]),
        Transform.from_translation([2, 0, 0]),
    )
    vol = np.zeros((4, 4, 4))
    better = _identity_pass_estimator(lambda t: t.translation[0] / 10)  # competitor 0.2 > 0.1
    assert better.estimate(vol, vol, seed=seed, competitor=competitor) is competitor
    tie = _identity_pass_estimator(lambda t: 0.5)
    assert tie.estimate(vol, vol, seed=seed, competitor=competitor) is seed


def test_a_competitor_seed_rescues_a_failed_first_pass():
    seed, competitor = (
        Transform.from_translation([1, 0, 0]),
        Transform.from_translation([2, 0, 0]),
    )
    vol = np.zeros((4, 4, 4))
    estimator = _identity_pass_estimator(lambda t: 0.3, fail_from=(seed,))
    assert estimator.estimate(vol, vol, seed=seed, competitor=competitor) is competitor
    with pytest.raises(EstimationError):
        estimator.estimate(vol, vol, seed=seed)


def test_a_competitor_seed_needs_a_score_fn():
    estimator = _drifting_estimator(np.zeros(3), None, iterations=1)
    with pytest.raises(ValueError, match="competitor"):
        estimator.estimate(
            np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), competitor=Transform.identity(3)
        )


def test_ants_estimates_are_repeatable():
    # ANTs samples voxels at random in its affine stage: without a fixed seed two runs on
    # the same data differ. With it, one thread repeats exactly (more threads add ~0.01
    # voxel of noise). A fresh process: ITK keeps its thread count once it has started.
    import subprocess
    import sys

    script = """
import numpy as np
from scipy.ndimage import shift as ndi_shift
from tests.test_registration_estimators import _synthetic_blob_volume
from biahub.core.transform import Transform
from biahub.registration.methods.ants import AntsEstimator
rng = np.random.default_rng(5)
ref = _synthetic_blob_volume(rng, (24, 48, 48))
mov = ndi_shift(ref, shift=(2, -3, 4), order=1, mode="constant", cval=0.0)
# with a seed, as the engine always calls it: the seed is applied (an ITK resample) first
seed = Transform.identity(3)
first, second = (AntsEstimator().estimate(mov, ref, seed=seed).matrix for _ in range(2))
assert np.array_equal(first, second), np.abs(first - second).max()
"""
    env = {**os.environ, "SLURM_CPUS_PER_TASK": "1"}
    env.pop("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", None)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_ants_threads_follow_the_task_cpus_unless_set(monkeypatch):
    from biahub.registration.methods.ants import _itk_threads

    monkeypatch.delenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert _itk_threads() == "8"
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert int(_itk_threads()) >= 1  # the cores this process may use
    monkeypatch.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "3")
    assert _itk_threads() == "3"  # set by the user: kept


def test_ants_reference_mask_is_centred_on_a_non_square_volume():
    # The circular reference mask must sit at the volume's centre (Y, X), whatever its
    # aspect: a swapped centre puts it off the volume and leaves nothing to register.
    from biahub.registration.methods.ants import preprocess_zyx

    vol = np.ones((4, 40, 100), dtype=np.float32)
    ref, _mov, offset = preprocess_zyx(vol, vol, crop=True, ref_mask_radius=0.5)
    centre_y = offset[1] + ref.shape[1] / 2
    centre_x = offset[2] + ref.shape[2] / 2
    assert abs(centre_y - 20) <= 1 and abs(centre_x - 50) <= 1


@pytest.mark.parametrize("padding", [False, True])
def test_classic_pcc_normalization_stays_finite_where_the_spectrum_is_zero(padding):
    # A volume with zero-magnitude frequencies (here: constant) must not turn the
    # correlation into NaNs, as the 'magnitude' normalization already guards against.
    from biahub.registration.methods.pcc import phase_cross_corr, phase_cross_corr_padding

    vol = np.full((8, 16, 16), 3.0, dtype=np.float32)
    fn = phase_cross_corr_padding if padding else phase_cross_corr
    shift, corr = fn(vol, vol, normalization="classic")
    assert np.all(np.isfinite(corr)) and np.all(np.isfinite(np.asarray(shift, dtype=float)))
