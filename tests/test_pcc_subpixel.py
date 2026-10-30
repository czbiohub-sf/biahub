"""biahub's phase cross-correlation: whole-pixel by default, sub-pixel when asked."""

import numpy as np
import pytest

from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndi_shift

from biahub.registration.methods.intensity.phase_cross_corr import phase_cross_corr


@pytest.mark.parametrize(
    "shape, shift", [((64, 64), (2.4, -1.7)), ((16, 48, 48), (1.3, 3.6, -2.2))]
)
def test_sub_pixel_shifts_match_the_upsampled_dft(shape, shift):
    # The evaluation measured tile and z shifts with skimage's phase correlation (sub-pixel);
    # biahub's own now refines the same way, so there is one implementation.
    from skimage.registration import phase_cross_correlation

    rng = np.random.default_rng(1)
    ref = gaussian_filter(rng.random(shape), 2).astype(np.float32)
    mov = ndi_shift(ref, shift, order=3, mode="wrap")
    ours = phase_cross_corr(ref, mov, upsample_factor=10)[0]
    theirs = phase_cross_correlation(ref, mov, upsample_factor=10, normalization=None)[0]
    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_whole_pixel_by_default_is_unchanged():
    rng = np.random.default_rng(2)
    ref = gaussian_filter(rng.random((32, 32)), 2).astype(np.float32)
    mov = np.roll(ref, (3, -4), axis=(0, 1))
    np.testing.assert_array_equal(phase_cross_corr(ref, mov)[0], [-3.0, 4.0])


def test_the_pcc_method_measures_sub_pixel_shifts_when_configured():
    # phase_cross_corr.upsample_factor: whole-pixel steps chained over hundreds of frames
    # added up to a drift; the method can now measure fractions of a pixel
    from pydantic import ValidationError
    from scipy.ndimage import fourier_shift

    from biahub.registration.methods.intensity.phase_cross_corr import PCCEstimator
    from biahub.settings import PhaseCrossCorrSettings

    rng = np.random.default_rng(3)
    ref = gaussian_filter(rng.random((16, 48, 48)), 2).astype(np.float32)
    shift = (0.0, 1.4, -2.3)
    # an exact sub-pixel shift (a spline shift with wrap leaves seams that bias any estimator)
    mov = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(ref), shift))).astype(np.float32)
    fine = PCCEstimator.from_settings(PhaseCrossCorrSettings(upsample_factor=10))
    coarse = PCCEstimator.from_settings(PhaseCrossCorrSettings())
    # forward (moving -> reference) translation undoes the applied shift
    np.testing.assert_allclose(
        fine.estimate(mov, ref).translation, [-s for s in shift], atol=0.1
    )
    assert np.all(np.mod(coarse.estimate(mov, ref).translation, 1) == 0)  # whole pixels
    with pytest.raises(ValidationError, match="custom_padding"):
        PhaseCrossCorrSettings(upsample_factor=4, function_type="custom_padding")
