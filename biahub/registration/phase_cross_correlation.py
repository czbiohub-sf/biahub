
from pathlib import Path
from typing import Literal, Optional, Tuple, cast

import click
import dask.array as da
import numpy as np

from numpy.typing import ArrayLike
from scipy.fftpack import next_fast_len
from skimage.registration import phase_cross_correlation as sk_phase_cross_correlation
from biahub.core.transform import Transform
from biahub.registration.utils import match_shape
from biahub.settings import (
    PhaseCrossCorrSettings,
)


def phase_cross_correlation(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    maximum_shift: float = 1.2,
    normalization: Optional[Literal["magnitude", "classic"]] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[int, ...]:
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2

    Computes translation shift using arg. maximum of phase cross correlation.
    Input are padded or cropped for fast FFT computation assuming a maximum translation shift.
    moving -> reference
    Parameters
    ----------
    ref_img : ArrayLike
        Reference image.
    mov_img : ArrayLike
        Moved image.
    maximum_shift : float, optional
        Maximum location shift normalized by axis size, by default 1.0

    Returns
    -------
    Tuple[int, ...]
        Shift between reference and moved image.
    """
    shape = tuple(
        cast(int, next_fast_len(int(max(s1, s2) * maximum_shift)))
        for s1, s2 in zip(ref_img.shape, mov_img.shape)
    )

    if verbose:
        click.echo(
            f"phase cross corr. fft shape of {shape} for arrays of shape {ref_img.shape} and {mov_img.shape} "
            f"with maximum shift of {maximum_shift}"
        )

    ref_img = match_shape(ref_img, shape)
    mov_img = match_shape(mov_img, shape)
    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps
    del ref_img, mov_img

    prod = Fimg1 * Fimg2.conj()

    if normalization == "magnitude":
        prod /= np.fmax(np.abs(prod), eps)
    elif normalization == "classic":
        prod /= np.abs(Fimg1) * np.abs(Fimg2)

    corr = np.fft.irfftn(prod)
    del prod, Fimg1, Fimg2

    corr = np.fft.fftshift(np.abs(corr))

    argmax = np.argmax(corr)
    peak = np.unravel_index(argmax, corr.shape)
    peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak))

    if verbose:
        click.echo(f"phase cross corr. peak at {peak}")

    return peak


def _estimate(
    mov: da.Array,
    ref: da.Array,
    function_type: Literal["skimage", "custom"] = "custom",
    normalization: Optional[Literal["magnitude", "classic"]] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Transform:
    """
    Estimate the transformation matrix from the shift.

    Parameters
    ----------
    mov : da.Array
        Moving image.
    ref : da.Array
        Reference image.
    function_type : Literal["skimage", "custom"]
        Function type to use for the phase cross correlation.
    normalization : Optional[Literal["magnitude", "classic"]]
        Normalization to use for the phase cross correlation.
    output_path : Optional[Path]
        Output path to save the plot.
    verbose : bool
        If True, print verbose output.
    """

    shift = shift_from_pcc(
        mov=mov,
        ref=ref,
        function_type=function_type,
        normalization=normalization,
        output_path=output_path,
        verbose=verbose,
    )

    if shift.ndim == 2:
        dy, dx = shift

        transform = np.eye(3)
        transform[1, 2] = dx
        transform[0, 2] = dy

    elif shift.ndim == 3:

        dz, dy, dx = shift

        transform = np.eye(4)
        transform[2, 3] = dx
        transform[1, 3] = dy
        transform[0, 3] = dz

    if verbose:
        click.echo(f"transform: {transform}")
    
    return Transform(matrix=transform), shift


def shift_from_pcc(
    mov: da.Array,
    ref: da.Array,
    function_type: Literal["skimage", "custom"] = "custom",
    normalization: Optional[Literal["magnitude", "classic"]] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[ArrayLike, Tuple[int, int, int]]:
    """
    Get the transformation matrix from phase cross correlation.

    Parameters
    ----------
    t : int
        Time index.
    source_channel_tzyx : da.Array
        Source channel data.
    target_channel_tzyx : da.Array
        Target channel data.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    ArrayLike
        Transformation matrix.
    """

    if function_type == "skimage":
        if normalization is not None:
            normalization = "phase"
        shift, _, _ = sk_phase_cross_correlation(
            reference_image=ref,
            moving_image=mov,
            normalization=normalization,
        )
    elif function_type == "custom":
        shift = phase_cross_correlation(mov, ref, normalization=normalization)
    if verbose:
        click.echo(f"shift: {shift}")
    return shift


def get_crop_idx(X, Y, Z, phase_cross_corr_settings):
    """
    Get the crop indices for the phase cross correlation.

    Parameters
    ----------
    X : int
        X dimension.
    Y : int
    Z : int
        Z dimension.
    phase_cross_corr_settings : PhaseCrossCorrSettings
        Phase cross correlation settings.

    Returns
    -------
    slice
        Crop indices.
    """
    X_slice = phase_cross_corr_settings.X_slice
    Y_slice = phase_cross_corr_settings.Y_slice
    Z_slice = phase_cross_corr_settings.Z_slice

    if phase_cross_corr_settings.center_crop_xy:
        x_idx = slice(
            X // 2 - phase_cross_corr_settings.center_crop_xy[0] // 2,
            X // 2 + phase_cross_corr_settings.center_crop_xy[0] // 2,
        )
        y_idx = slice(
            Y // 2 - phase_cross_corr_settings.center_crop_xy[1] // 2,
            Y // 2 + phase_cross_corr_settings.center_crop_xy[1] // 2,
        )
    else:
        x_idx = slice(0, X)
        y_idx = slice(0, Y)
    if X_slice == "all":
        x_idx = slice(0, X)
    else:
        x_idx = slice(X_slice[0], X_slice[1])
    if Y_slice == "all":
        y_idx = slice(0, Y)
    else:
        y_idx = slice(Y_slice[0], Y_slice[1])
    if Z_slice == "all":
        z_idx = slice(0, Z)
    else:
        z_idx = slice(Z_slice[0], Z_slice[1])

    print(f"x_idx: {x_idx}, y_idx: {y_idx}, z_idx: {z_idx}")

    return x_idx, y_idx, z_idx


def estimate(
    t: int,
    fov: str,
    mov: da.Array,
    ref: da.Array,
    output_dirpath: Path,
    preprocessing: bool = False,
    phase_cross_corr_settings: PhaseCrossCorrSettings = None,
    verbose: bool = False,
) -> list[ArrayLike]:
    """
    Estimate the xyz stabilization for a single position.

    Parameters
    ----------
    input_position_dirpath : Path
        Path to the input position directory.
    output_folder_path : Path
        Path to the output folder.
    channel_index : int
        Index of the channel to process.
    center_crop_xy : list[int]
        Size of the crop in the XY plane.
    t_reference : str
        Reference timepoint.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    list[ArrayLike]
        List of the xyz stabilization for each timepoint.
    """
    T, C, Z, Y, X = mov.shape

    # preprocessing
    if preprocessing:
        x_idx, y_idx, z_idx = get_crop_idx(X, Y, Z, phase_cross_corr_settings)
        mov = mov[z_idx, y_idx, x_idx]
        ref = ref[z_idx, y_idx, x_idx]


    output_transforms_path_fov = output_dirpath /"transforms" / fov
    output_transforms_path_fov.mkdir(parents=True, exist_ok=True)
    output_shifts_path_fov = output_dirpath / "shifts" / fov 
    output_shifts_path_fov.mkdir(parents=True, exist_ok=True)
    output_path_corr = output_dirpath / "corr_plots" / fov
    output_path_corr.mkdir(parents=True, exist_ok=True)

    transform, shift = _estimate(
            mov=mov,
            ref=ref,
            function_type=phase_cross_corr_settings.function_type,
            normalization=phase_cross_corr_settings.normalization,
            output_path=output_path_corr / f"{t}.png",
            verbose=verbose,
        )
        
    # save the transform
    np.save(output_transforms_path_fov / f"{t}.npy", transform)
    np.save(output_shifts_path_fov / f"{t}.npy", shift)



    return transform

