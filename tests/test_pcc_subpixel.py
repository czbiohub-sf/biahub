"""biahub's phase cross-correlation: whole-pixel by default, sub-pixel when asked."""

import numpy as np
import pytest

from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndi_shift

from biahub.registration.methods.pcc import phase_cross_corr


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
