import numpy as np
import pytest

from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform
from biahub.registration.ants import correlation_score
from biahub.registration.metrics import (
    bead_alignment_metrics,
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
    settings = BeadsMatchSettings(source_peaks_settings=peaks, target_peaks_settings=peaks)

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
