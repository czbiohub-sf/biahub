from pathlib import Path

import ants
import click
import dask.array as da
import numpy as np

from skimage import filters
from biahub.cli.utils import _check_nan_n_zeros
from biahub.core.transform import Transform
from biahub.registration.utils import (
    find_lir,
)
from biahub.settings import (
    AffineTransformSettings,
    AntsRegistrationSettings,
)

def _estimate(
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
        ants_kwargs = {
            "type_of_transform": "Similarity",
            "aff_shrink_factors": (6, 3, 1),
            "aff_iterations": (2100, 1200, 50),
            "aff_smoothing_sigmas": (2, 1, 0),
        }

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


def preprocess_czyx(
    mov_czyx: np.ndarray,
    ref_czyx: np.ndarray,
    initial_tform: Transform,
    mov_channel_index: int | list = 0,
    ref_channel_index: int = 0,
    crop: bool = False,
    ref_mask_radius: float | None = None,
    clip: bool = False,
    sobel_filter: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize the affine transform between source and target channels using ANTs library.

    Parameters
    ----------
    mov_czyx : np.ndarray
        Source channel data in CZYX format.
    ref_czyx : np.ndarray
        Target channel data in CZYX format.
    initial_tform : np.ndarray
        Approximate estimate of the affine transform matrix, often obtained through manual registration, see `estimate-registration`.
    mov_channel_index : int | list, optional
        Index or list of indices of mov channels to be used for registration, by default 0.
    ref_channel_index : int, optional
        Index of the reference channel to be used for registration, by default 0.
    crop : bool, optional
        Whether to crop the moving and reference channels to the overlapping region as determined by the LIR algorithm, by default False.
    ref_mask_radius : float | None, optional
        Radius of the circular mask which will be applied to the reference channel. By default None in which case no masking will be applied.
    clip : bool, optional
        Whether to clip the moving and reference channels to reasonable (hardcoded) values, by default False.
    sobel_filter : bool, optional
        Whether to apply Sobel filter to the moving and reference channels, by default False.
    verbose : bool, optional
        Whether to print verbose output during registration, by default False.

    Returns
    -------
    Transform | None
        Optimized affine transform matrix or None if the input data contains NaN or zeros.

    Notes
    -----
    This function applies an initial affine transform to the source channels, crops them to the overlapping region with the target channel,
    clips the values, applies Sobel filtering if specified, and then optimizes the registration parameters using ANTs library.

    This function currently assumes that target channel is phase and source channels are fluorescence.
    If multiple source channels are provided, they will be summed, after clipping, filtering, and cropping, if enabled.
    """

    mov_czyx = np.asarray(mov_czyx).astype(np.float32)
    ref_czyx = np.asarray(ref_czyx).astype(np.float32)

    if ref_mask_radius is not None and not (0 < ref_mask_radius <= 1):
        raise ValueError(
            "ref_mask_radius must be given as a fraction of image width, i.e. (0, 1]."
        )

    if _check_nan_n_zeros(mov_czyx) or _check_nan_n_zeros(ref_czyx):
        raise ValueError("Input data contains NaN or zeros.")
    t_form_ants = initial_tform.to_ants()

    ref_zyx = ref_czyx[ref_channel_index]
    if ref_zyx.ndim != 3:
        raise ValueError(f"Expected 3D reference channel, got shape {ref_zyx.shape}")
    ref_ants_pre_crop = ants.from_numpy(ref_zyx)

    if not isinstance(mov_channel_index, list):
        mov_channel_index = [mov_channel_index]

    mov_channels = []
    for idx in mov_channel_index:
        if verbose:
            click.echo(f"Applying initial transform to moving channel {idx}...")
        # Cropping, clipping, and filtering are applied after registration with initial_tform
        _mov_channel = np.asarray(mov_czyx[idx]).astype(np.float32)
        if _mov_channel.ndim != 3:
            raise ValueError(f"Expected 3D moving channel, got shape {_mov_channel.shape}")
        mov_channel = t_form_ants.apply_to_image(
            ants.from_numpy(_mov_channel), reference=ref_ants_pre_crop
        ).numpy()
        if mov_channel.ndim != 3:
            raise ValueError(
                f"apply_to_image returned non-3D array: {mov_channel.shape}.\n"
                "This is likely caused by mismatched input shape or invalid transform/reference."
            )
        mov_channels.append(mov_channel)

    _offset = np.zeros(3, dtype=np.float32)
    if crop:
        if verbose:
            click.echo(
                "Estimating crop for moving and reference channels to overlapping region..."
            )
        mask = (ref_zyx != 0) & (mov_channels[0] != 0)

        # Can be refactored with code in cropping PR #88
        if ref_mask_radius is not None:
            ref_mask = np.zeros(ref_zyx.shape[-2:], dtype=bool)

            y, x = np.ogrid[: ref_mask.shape[-2], : ref_mask.shape[-1]]
            center = (ref_mask.shape[-2] // 2, ref_mask.shape[-1] // 2)
            radius = int(ref_mask_radius * min(center))

            ref_mask[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2] = True
            mask *= ref_mask

        z_slice, y_slice, x_slice = find_lir(mask.astype(np.uint8))
        click.echo(
            f"Cropping to region z={z_slice.start}:{z_slice.stop}, "
            f"y={y_slice.start}:{y_slice.stop}, "
            f"x={x_slice.start}:{x_slice.stop}"
        )

        _offset = np.asarray(
            [_s.start for _s in (z_slice, y_slice, x_slice)], dtype=np.float32
        )
        ref_zyx = ref_zyx[z_slice, y_slice, x_slice]
        mov_channels = [_channel[z_slice, y_slice, x_slice] for _channel in mov_channels]

    # TODO: hardcoded clipping limits
    if clip:
        if verbose:
            click.echo("Clipping moving and reference channels to reasonable values...")
        ref_zyx = np.clip(ref_zyx, 0, 0.5)
        mov_channels = [
            np.clip(_channel, 110, np.quantile(_channel, 0.99)) for _channel in mov_channels
        ]

    if sobel_filter:
        if verbose:
            click.echo("Applying Sobel filter to moving and reference channels...")
        ref_zyx = filters.sobel(ref_zyx)
        mov_channels = [filters.sobel(_channel) for _channel in mov_channels]

    mov_zyx = np.sum(mov_channels, axis=0)

    return ref_zyx, mov_zyx, _offset


def estimate(
    mov: np.ndarray,
    ref: np.ndarray,
    preprocessing: bool = False,
    initial_tform: np.ndarray = None,
    mov_channel_index: int | list = 0,
    ref_channel_index: int = 0,
    crop: bool = False,
    ref_mask_radius: float | None = None,
    clip: bool = False,
    sobel_filter: bool = False,
    verbose: bool = False,
    t_idx: int = 0,
    output_dirpath: Path = None,
    debug: bool = False,
) -> Transform:
    """
    Optimize the affine transform between source and target channels using ANTs library.

    Parameters
    ----------
    mov_czyx : np.ndarray
        Moving channel data in CZYX format.
    ref_czyx : np.ndarray
        Reference channel data in CZYX format.
    initial_tform : np.ndarray
        Approximate estimate of the affine transform matrix, often obtained through manual registration, see `estimate-registration`.
    mov_channel_index : int | list, optional
        Index or list of indices of moving channels to be used for registration, by default 0.
    ref_channel_index : int, optional
        Index of the reference channel to be used for registration, by default 0.
    crop : bool, optional
        Whether to crop the moving and reference channels to the overlapping region as determined by the LIR algorithm, by default False.
    ref_mask_radius : float | None, optional
        Radius of the circular mask which will be applied to the reference channel. By default None in which case no masking will be applied.
    clip : bool, optional
        Whether to clip the moving and reference channels to reasonable (hardcoded) values, by default False.
    sobel_filter : bool, optional
        Whether to apply Sobel filter to the moving and reference channels, by default False.
    verbose : bool, optional
        Whether to print verbose output during registration, by default False.
    t_idx : int, optional
        Time index for the registration, by default 0.
    output_folder_path : str | None, optional
        Path to the folder where the output transform will be saved, by default None.

    Returns
    -------
    Transform | None
        Optimized affine transform matrix or None if the input data contains NaN or zeros.

    Notes
    -----
    This function applies an initial affine transform to the moving channels, crops them to the overlapping region with the reference channel,
    clips the values, applies Sobel filtering if specified, and then optimizes the registration parameters using ANTs library.

    This function currently assumes that reference channel is phase and moving channels are fluorescence.
    If multiple moving channels are provided, they will be summed, after clipping, filtering, and cropping, if enabled.
    """
    initial_tform = Transform(matrix=initial_tform)


    if preprocessing:
        ref, mov, preprocess_offset = preprocess_czyx(
            mov=mov,
            ref=ref,
            initial_tform=initial_tform,
            mov_channel_index=mov_channel_index,
            ref_channel_index=ref_channel_index,
            crop=crop,
            clip=clip,
            ref_mask_radius=ref_mask_radius,
            sobel_filter=sobel_filter,
            verbose=verbose,
        )

    fwd_transform, inv_transform = _estimate(
        ref=ref,
        mov=mov,
        verbose=verbose,
    )

    if preprocessing:

        final_transform = postprocess_transform(
            initial_transform=initial_tform,
            fwd_transform=fwd_transform,
            preprocess_offset=preprocess_offset,
        )
    else:
        final_transform = fwd_transform
    

    if final_transform is None:
        raise ValueError("Failed to estimate registration transform for timepoint.")

    if debug:
        click.echo(f"Initial transform: {initial_tform}")
        click.echo(f"Forward transform: {fwd_transform}")
        click.echo(f"Inverse transform: {inv_transform}")
        click.echo(f"Final transform: {final_transform}")

    if output_dirpath:
        output_dirpath.mkdir(parents=True, exist_ok=True)
        if verbose:
            click.echo(
                f"Saving registration transform for timepoint {t_idx} to {output_dirpath}"
            )

        np.save(output_dirpath / f"{t_idx}.npy", final_transform.matrix)

    return final_transform


def postprocess_transform(
    initial_transform: Transform,
    fwd_transform: Transform,
    preprocess_offset: np.ndarray,
) -> Transform:
    """
    Postprocess the transform to account for the offset.
    """

    shift_to_roi = np.eye(4)
    shift_to_roi[:3, -1] = preprocess_offset

    shift_back = np.eye(4)
    shift_back[:3, -1] = -preprocess_offset

    composed_matrix = (
        initial_transform.matrix @ shift_to_roi @ fwd_transform.matrix @ shift_back
    )

    return Transform(matrix=composed_matrix)

