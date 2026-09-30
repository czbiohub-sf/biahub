"""
ANTs-based intensity registration module.

Provides functions for registering volumetric imaging data using the ANTsPy
library's optimization-based registration. This complements bead-based
registration by directly optimizing image similarity metrics.

Key conventions
---------------
- Coordinates are in ZYX order for 3D data.
- "mov" / "moving" refers to the source channel being aligned.
- "ref" / "reference" refers to the fixed target channel.
- Transforms are 4x4 homogeneous matrices stored as Transform objects.
"""

from __future__ import annotations

import ants
import click
import numpy as np

from numpy.typing import ArrayLike
from skimage import filters

from biahub.core.transform import Transform
from biahub.registration.utils import find_lir
from biahub.settings import AffineTransformSettings, AntsRegistrationSettings

DEFAULT_ANTS_KWARGS = {
    "type_of_transform": "Similarity",
    "aff_shrink_factors": (6, 3, 1),
    "aff_iterations": (2100, 1200, 50),
    "aff_smoothing_sigmas": (2, 1, 0),
}


def estimate(
    ref: np.ndarray,
    mov: np.ndarray,
    verbose: bool = False,
    ants_kwargs: dict = None,
) -> tuple[Transform, Transform]:
    """
    Estimate affine transformation using ANTs registration.

    Works for both 2D (Y, X) and 3D (Z, Y, X) arrays.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (2D or 3D)
    mov : np.ndarray
        Moving image (2D or 3D)
    verbose : bool
        Print optimization progress
    ants_kwargs : dict, optional
        Additional ANTs parameters

    Returns
    -------
    fwd_transform : Transform
        Forward transformation (mov → ref)
    inv_transform : Transform
        Inverse transformation (ref → mov)
    """
    if ref.ndim not in (2, 3) or mov.ndim not in (2, 3):
        raise ValueError(
            f"Images must be 2D or 3D, got ref.ndim={ref.ndim}, mov.ndim={mov.ndim}"
        )

    if ref.ndim != mov.ndim:
        raise ValueError(f"Dimension mismatch: ref.ndim={ref.ndim}, mov.ndim={mov.ndim}")

    if ants_kwargs is None:
        ants_kwargs = dict(DEFAULT_ANTS_KWARGS)

    mov_ants = ants.from_numpy(mov)
    ref_ants = ants.from_numpy(ref)

    if verbose:
        click.echo(f"Optimizing registration parameters using ANTs with kwargs: {ants_kwargs}")

    reg = ants.registration(
        fixed=ref_ants,
        moving=mov_ants,
        **ants_kwargs,
        verbose=verbose,
    )

    fwd_transform_mat = ants.read_transform(reg["fwdtransforms"][0])
    inv_transform_mat = ants.read_transform(reg["invtransforms"][0])

    fwd_transform = Transform.from_ants(fwd_transform_mat)
    inv_transform = Transform.from_ants(inv_transform_mat)

    if fwd_transform.matrix is None or inv_transform.matrix is None:
        raise ValueError("Failed to estimate registration transform.")

    return fwd_transform, inv_transform


def preprocess_zyx(
    mov_zyx: np.ndarray,
    ref_zyx: np.ndarray,
    crop: bool = False,
    ref_mask_radius: float | None = None,
    clip: bool = False,
    sobel_filter: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare one already-aligned moving volume and its reference for intensity registration.

    `mov_zyx` must already be warped into `ref_zyx`'s frame. Returns `(ref, mov, offset)`
    where `offset` is the ZYX origin of the crop within the full volume (zeros without
    `crop`), so a correction estimated on the crop can be composed back into full-volume
    coordinates.
    """
    ref = np.asarray(ref_zyx, dtype=np.float32)
    mov = np.asarray(mov_zyx, dtype=np.float32)
    offset = np.zeros(3, dtype=np.float32)
    if crop:
        mask = (ref != 0) & (mov != 0)
        if ref_mask_radius is not None:
            ref_mask = np.zeros(ref.shape[-2:], dtype=bool)
            y, x = np.ogrid[: ref_mask.shape[-2], : ref_mask.shape[-1]]
            center = (ref_mask.shape[-2] // 2, ref_mask.shape[-1] // 2)
            radius = int(ref_mask_radius * min(center))
            ref_mask[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2] = True
            mask &= ref_mask
        z_slice, y_slice, x_slice = find_lir(mask.astype(np.uint8))
        offset = np.asarray([s.start for s in (z_slice, y_slice, x_slice)], dtype=np.float32)
        ref = ref[z_slice, y_slice, x_slice]
        mov = mov[z_slice, y_slice, x_slice]
    if clip:
        # Limits assume a phase reference; see AntsRegistrationSettings.clip.
        ref = np.clip(ref, 0, 0.5)
        mov = np.clip(mov, 110, np.quantile(mov, 0.99))
    if sobel_filter:
        ref = filters.sobel(ref)
        mov = filters.sobel(mov)
    return ref, mov, offset


_ANTS_TRANSFORM_TYPE = {
    "euclidean": "Rigid",
    "rigid": "Rigid",
    "similarity": "Similarity",
    "affine": "Affine",
}


class AntsEstimator:
    """TransformEstimator using ANTs intensity-based optimization.

    One pass: pre-warp `mov` by the seed into `ref`'s frame, prepare both volumes
    (`ants.preprocess_zyx`: optional crop to their overlap, reference mask, clip, Sobel),
    register, and compose the correction back through the crop offset --
    `shift(+offset) @ correction @ shift(-offset) @ seed`. This is the legacy
    ANTs pipeline in the engine's forward (moving -> reference)
    convention.

    `ants.estimate()`'s `fwd_transform` is, despite its name, the reference -> moving
    ("pull") direction (see `tests/test_registration_estimators.py`); it is inverted here.
    """

    def __init__(
        self,
        ants_kwargs: dict | None = None,
        crop: bool = False,
        ref_mask_radius: float | None = None,
        clip: bool = False,
        sobel_filter: bool = False,
        verbose: bool = False,
    ):
        self.ants_kwargs = dict(ants_kwargs) if ants_kwargs else dict(DEFAULT_ANTS_KWARGS)
        self.crop = crop
        self.ref_mask_radius = ref_mask_radius
        self.clip = clip
        self.sobel_filter = sobel_filter
        self.verbose = verbose

    @classmethod
    def from_settings(
        cls,
        ants_registration_settings: AntsRegistrationSettings,
        affine_transform_settings: AffineTransformSettings,
        verbose: bool = False,
    ) -> AntsEstimator:
        """Preprocessing from the ANTs settings, transform family from the affine settings."""
        ants_kwargs = dict(DEFAULT_ANTS_KWARGS)
        ants_kwargs["type_of_transform"] = _ANTS_TRANSFORM_TYPE[
            affine_transform_settings.transform_type
        ]
        return cls(
            ants_kwargs=ants_kwargs,
            crop=ants_registration_settings.crop,
            ref_mask_radius=ants_registration_settings.ref_mask_radius,
            clip=ants_registration_settings.clip,
            sobel_filter=ants_registration_settings.sobel_filter,
            verbose=verbose,
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        aligned = seed.apply(mov, reference=ref) if seed is not None else mov
        ref_prepared, mov_prepared, offset = preprocess_zyx(
            aligned,
            ref,
            crop=self.crop,
            ref_mask_radius=self.ref_mask_radius,
            clip=self.clip,
            sobel_filter=self.sobel_filter,
        )
        pull_correction, _unused = estimate(
            ref=ref_prepared,
            mov=mov_prepared,
            verbose=self.verbose,
            ants_kwargs=self.ants_kwargs,
        )
        correction = pull_correction.invert()
        if np.any(offset):
            correction = (
                Transform.from_translation(offset)
                @ correction
                @ Transform.from_translation(-offset)
            )
        return correction @ seed if seed is not None else correction
