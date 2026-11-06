import numpy as np
import pytest

from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform
from biahub.registration.metrics import (
    bead_alignment_metrics,
    correlation_score,
    gradient_correlation,
    normalized_mutual_information,
    residual_score,
    residual_shift,
)
from biahub.settings import BeadsMatchSettings, DetectPeaksSettings

APPLIED_ZYX = (2.0, -3.0, 4.0)
SHAPE = (40, 60, 60)


def _beads(rng, shape=SHAPE, n_beads=20, sigma=2.0, amplitude=500.0, noise_std=5.0):
    margin = 8
    centers = rng.uniform([margin] * 3, np.asarray(shape) - margin, size=(n_beads, 3))
    grid = np.indices(shape, dtype=float)
    volume = np.zeros(shape, dtype=np.float32)
    for c in centers:
        d2 = sum((g - ci) ** 2 for g, ci in zip(grid, c, strict=True))
        volume += (amplitude * np.exp(-d2 / (2 * sigma**2))).astype(np.float32)
    return volume + rng.normal(0, noise_std, size=shape).astype(np.float32)


@pytest.fixture
def pair():
    rng = np.random.default_rng(3)
    ref = _beads(rng)
    mov = ndi_shift(ref, shift=APPLIED_ZYX, order=1, mode="constant", cval=0.0)
    truth = Transform.from_translation([-a for a in APPLIED_ZYX])
    wrong = Transform.from_translation([-a + 3.0 for a in APPLIED_ZYX])
    return mov, ref, truth, wrong


def test_bead_metrics_are_continuous_where_overlap_ties(pair):
    mov, ref, truth, wrong = pair
    peaks = DetectPeaksSettings(
        threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    settings = BeadsMatchSettings(moving_peaks_settings=peaks, reference_peaks_settings=peaks)

    good = bead_alignment_metrics(truth, mov, ref, settings)
    off = bead_alignment_metrics(wrong, mov, ref, settings)

    assert good.n_matched >= 15 and good.median_residual < 1.0
    # 3 voxels off is still inside the 6-voxel overlap radius: same bead count ...
    assert off.overlap == pytest.approx(good.overlap, abs=0.15)
    # ... but the residual sees it.
    assert off.median_residual > good.median_residual + 1.5
    assert residual_score(truth, mov, ref, settings) > residual_score(
        wrong, mov, ref, settings
    )


def test_mutual_information_and_gradient_correlation_prefer_the_correct_transform(pair):
    mov, ref, truth, wrong = pair
    assert normalized_mutual_information(Transform.identity(3), ref, ref) == pytest.approx(1.0)
    assert normalized_mutual_information(truth, mov, ref) > normalized_mutual_information(
        wrong, mov, ref
    )
    assert gradient_correlation(truth, mov, ref) > gradient_correlation(wrong, mov, ref)


def test_mutual_information_survives_a_modality_change_where_correlation_does_not(pair):
    """A 'phase-like' reference: a smooth, inverted, non-linear function of the bead
    volume. Intensities anti-correlate with the fluorescence, so Pearson correlation of
    the CORRECT transform is negative, while mutual information still ranks it first."""
    mov, ref, truth, wrong = pair
    phase_like = -np.log1p(gaussian_filter(ref, 1.0) / 50.0).astype(np.float32) + 8.0

    assert correlation_score(truth, mov, phase_like) < 0
    assert normalized_mutual_information(
        truth, mov, phase_like
    ) > normalized_mutual_information(wrong, mov, phase_like)
    assert normalized_mutual_information(
        truth, mov, phase_like
    ) > normalized_mutual_information(
        Transform.from_translation([10.0, 10.0, 10.0]), mov, phase_like
    )


def test_transform_apply_defaults_to_ants_for_3d_and_scipy_for_2d(pair):
    mov, ref, truth, _wrong = pair
    default = truth.apply(mov, reference=ref)
    ants = truth.apply(mov, reference=ref, backend="ants")
    scipy = truth.apply(mov, reference=ref, backend="scipy")
    interior = tuple(slice(6, s - 6) for s in SHAPE)
    np.testing.assert_array_equal(default, ants)
    assert (
        np.abs(ants[interior] - scipy[interior]).mean() < 0.05 * np.abs(scipy[interior]).mean()
    )

    image = np.random.default_rng(0).random((20, 30)).astype(np.float32)
    t2d = Transform.from_translation([1.0, -2.0])
    np.testing.assert_array_equal(t2d.apply(image), t2d.apply(image, backend="scipy"))


def test_residual_shift_measures_the_remaining_displacement_in_voxels(pair):
    mov, ref, truth, wrong = pair
    assert residual_shift(truth, mov, ref) < 1.0
    assert residual_shift(wrong, mov, ref) == pytest.approx(np.sqrt(3 * 3.0**2), abs=1.0)
    assert normalized_mutual_information(
        truth, mov, ref, sobel=True
    ) > normalized_mutual_information(wrong, mov, ref, sobel=True)


def test_residual_shift_in_um_scales_each_axis(pair):
    # anisotropic voxels: 3 voxels off on every axis is 0.6 µm in z but 0.3 µm in y and x
    mov, ref, _truth, wrong = pair
    voxel_um = (0.2, 0.1, 0.1)
    in_vox = residual_shift(wrong, mov, ref)
    assert residual_shift(wrong, mov, ref, ref_um=voxel_um) == pytest.approx(
        np.sqrt(0.6**2 + 0.3**2 + 0.3**2), abs=0.2
    )
    assert residual_shift(wrong, mov, ref, ref_um=None) == in_vox  # the default: voxels


def test_bead_residuals_in_um_scale_each_axis_and_pair_the_same_beads(pair):
    mov, ref, _truth, wrong = pair
    peaks = DetectPeaksSettings(
        threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    settings = BeadsMatchSettings(moving_peaks_settings=peaks, reference_peaks_settings=peaks)
    in_vox = bead_alignment_metrics(wrong, mov, ref, settings)
    in_um = bead_alignment_metrics(wrong, mov, ref, settings, ref_um=(0.2, 0.1, 0.1))
    # matching stays on the configured radius in voxels: the same beads pair up
    assert (in_um.n_matched, in_um.overlap) == (in_vox.n_matched, in_vox.overlap)
    # 3 voxels off on every axis: 0.6 µm in z, 0.3 µm in y and x
    assert in_um.median_residual == pytest.approx(np.sqrt(0.6**2 + 0.3**2 + 0.3**2), abs=0.15)
    assert in_vox.median_residual == pytest.approx(np.sqrt(3 * 3.0**2), abs=1.0)


def _drifting_series(T=12):
    """A blob volume drifting (random walk) with one-frame jumps; returns (series, where)."""
    from tests.test_registration_estimators import _synthetic_blob_volume

    rng = np.random.default_rng(7)
    first = _synthetic_blob_volume(
        np.random.default_rng(3), (16, 96, 96), n_blobs=80, sigma=2.0
    )
    where = np.cumsum(rng.normal(0, 0.3, size=(T, 3)), axis=0)
    for t, jump in {3: (0.0, 4.0, 0.0), 5: (1.0, 0.0, -3.0), 8: (0.0, -2.5, 2.5)}.items():
        if t < T:
            where[t] += jump
    frames = [ndi_shift(first + 100.0, w, order=1, mode="nearest") for w in where]
    return np.stack(frames).astype(np.float32), where


def test_leftover_motion_is_the_raw_shift_less_the_change_in_correction():
    # Checked against the truth: the stabilized sample sits at where[t] + c[t], so the
    # shift moving frame t onto t-1 is -(where[t] + c[t] - where[t-1] - c[t-1]). The
    # corrections are far off (+-5 voxels) on purpose: no warp, so nothing aliases.
    from biahub.registration.metrics import leftover_motion, raw_shift

    series, where = _drifting_series()
    um = np.array([0.5, 0.25, 0.25])
    corrections = np.random.default_rng(1).uniform(-5, 5, size=(len(series), 3))
    pairs = [(t - 1, t, raw_shift(series[t - 1], series[t])) for t in range(1, len(series))]
    result = leftover_motion(pairs, dict(enumerate(corrections)), um)
    truth = -((where[1:] + corrections[1:]) - (where[:-1] + corrections[:-1])) * um
    got = np.asarray(
        [[p["shift_z_um"], p["shift_y_um"], p["shift_x_um"]] for p in result["pairs"]]
    )
    np.testing.assert_allclose(got, truth, atol=0.15 * um.max())
    np.testing.assert_allclose(
        result["mean_um"], np.linalg.norm(truth, axis=1).mean(), atol=0.05
    )
    # unregistered: the raw shifts themselves
    raw = leftover_motion(pairs, None, um)
    np.testing.assert_allclose(
        raw["max_um"], np.linalg.norm(-(where[1:] - where[:-1]) * um, axis=1).max(), atol=0.05
    )


def test_leftover_motion_skips_pairs_without_a_raw_shift():
    from biahub.registration.metrics import leftover_motion, raw_shift

    series, _ = _drifting_series(T=4)
    empty = np.zeros_like(series[0])
    assert raw_shift(series[0], empty) is None  # nothing to correlate
    pairs = [
        (0, 1, raw_shift(series[0], series[1])),
        (1, 2, None),
        (2, 3, raw_shift(series[2], series[3])),
    ]
    result = leftover_motion(pairs, None, (1.0, 1.0, 1.0))
    assert [(p["t_from"], p["t_to"]) for p in result["pairs"]] == [(0, 1), (2, 3)]
    assert result["skipped"] == [[1, 2]]


def test_a_raw_shift_holds_at_most_18_frames():
    # each estimate job measures it on full volumes (0.4 GB on the neuromast): the job plan
    # reserves LEFTOVER_MOTION_VOLUMES frames for it
    import tracemalloc

    from biahub.registration.metrics import LEFTOVER_MOTION_VOLUMES, raw_shift

    frames = np.random.default_rng(0).random((2, 32, 128, 128)).astype(np.float32)
    tracemalloc.start()
    raw_shift(frames[0], frames[1])
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert LEFTOVER_MOTION_VOLUMES == 18
    assert peak <= LEFTOVER_MOTION_VOLUMES * frames[0].nbytes


def test_an_intensity_centroid_finds_the_sample_above_its_background():
    # the prototype's measure: binned, blurred, weighted by intensity above a percentile
    from biahub.registration.metrics import intensity_centroid

    volume = np.full((20, 40, 40), 10.0, dtype=np.float32)  # a flat background
    volume[8:12, 20:26, 10:14] = 100.0  # its centre is (9.5, 22.5, 11.5) in full-res voxels
    centroid = intensity_centroid(volume, bg_percentile=90, downsample=2, blur_sigma=1.0)
    np.testing.assert_allclose(centroid, (9.5, 22.5, 11.5), atol=0.3)
    # a projection (Z = 1) is binned in yx only, and its z is 0
    projected = volume.max(axis=0, keepdims=True)
    centroid = intensity_centroid(projected, bg_percentile=90, downsample=2, blur_sigma=1.0)
    np.testing.assert_allclose(centroid, (0.0, 22.5, 11.5), atol=0.3)
    # nothing above the background: no centroid
    flat = np.full((20, 40, 40), 10.0, dtype=np.float32)
    assert np.isnan(
        intensity_centroid(flat, bg_percentile=90, downsample=2, blur_sigma=1.0)
    ).all()


def test_a_smoothed_track_drops_single_frame_jumps_and_fills_gaps():
    from biahub.registration.metrics import smooth_track

    track = np.column_stack([np.linspace(0.0, 10.0, 21)] * 3)  # a slow drift, 0.5 / frame
    track[10] += 50.0  # a one-frame jump
    track[5] = np.nan  # a frame with no centroid
    smoothed = smooth_track(track, median=3, sigma=1.0)
    assert np.isfinite(smoothed).all()
    # the jump is gone; the median skips its frame, so t=10 lands within a frame's drift
    np.testing.assert_allclose(smoothed[10], 5.0, atol=0.5)
    np.testing.assert_allclose(smoothed[5], 2.5, atol=0.3)
    np.testing.assert_allclose(smoothed[3:18, 0], np.linspace(1.5, 8.5, 15), atol=0.5)
