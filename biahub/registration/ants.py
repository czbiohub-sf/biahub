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

import ants
import click
import numpy as np

from skimage import filters

from biahub.core.transform import Transform
from biahub.registration.utils import (
    find_lir,
)

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


def correlation_score(
    transform: Transform,
    mov: np.ndarray,
    ref: np.ndarray,
    sobel_filter: bool = False,
) -> float:
    """Pearson correlation between `mov` warped by `transform` and `ref`, over their overlap.

    An intensity analogue of the bead overlap score: continuous in [-1, 1], nan when
    the warped volume and the reference do not overlap. Optionally compares Sobel
    magnitudes instead, for cross-modality pairs registered that way.
    """
    ref = np.asarray(ref, dtype=np.float32)
    warped = transform.apply(np.asarray(mov, dtype=np.float32), reference=ref)
    mask = (warped != 0) & (ref != 0)
    if mask.sum() < 2:
        return float("nan")
    a, b = warped[mask], ref[mask]
    if sobel_filter:
        a, b = filters.sobel(warped)[mask], filters.sobel(ref)[mask]
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    if denominator == 0:
        return float("nan")
    return float((a * b).sum() / denominator)
