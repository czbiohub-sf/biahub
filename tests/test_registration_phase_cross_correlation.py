import numpy as np

from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform
from biahub.registration.phase_cross_correlation import get_tform_from_pcc
from biahub.registration.utils import apply_affine_transform


def _synthetic_blob_volume(rng, shape, n_blobs=12, sigma=3.0, noise_std=5.0):
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


def test_get_tform_from_pcc_correctly_aligns_source_onto_target():
    """Regression test for two bugs fixed when this function moved out of
    estimate_stabilization.py: a reference/moving swap that flipped the sign of the
    recovered shift, and a Z/X axis swap when building the translation column (#356).

    Also pins the direction: the returned matrix must be usable directly (no extra
    inversion by the caller) by this codebase's real apply step, matching every other
    registration method's convention here.
    """
    rng = np.random.default_rng(8)
    shape = (32, 40, 40)
    target_frame = _synthetic_blob_volume(rng, shape)

    applied_zyx = (2, 4, -6)  # distinct per-axis values so an axis mixup would show
    source_frame = ndi_shift(
        target_frame, shift=applied_zyx, order=1, mode="constant", cval=0.0
    )

    source_channel_tzyx = np.stack([target_frame, source_frame])
    target_channel_tzyx = np.stack([target_frame, target_frame])

    transform, shift, _corr = get_tform_from_pcc(
        t=1,
        source_channel_tzyx=source_channel_tzyx,
        target_channel_tzyx=target_channel_tzyx,
        function_type="custom",
    )

    # Matches Transform.from_translation(shift).invert() -- the same "invert before
    # returning" convention as ants.py:estimate(), user_assisted_registration, stackreg.
    np.testing.assert_allclose(transform, Transform.from_translation(shift).invert().matrix)

    margin = 6
    interior = tuple(slice(margin, s - margin) for s in shape)

    def relerr(a, b):
        return np.abs(a[interior] - b[interior]).mean() / (np.abs(b[interior]).mean() + 1e-8)

    for method in ("scipy", "ants"):
        aligned = apply_affine_transform(source_frame, transform, shape, method=method)
        assert relerr(aligned, target_frame) < 0.01
