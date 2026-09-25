"""Phase cross-correlation registration/stabilization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import click
import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import ArrayLike
from scipy.fftpack import next_fast_len

from biahub.core.transform import Transform
from biahub.registration.utils import match_shape
from biahub.settings import PhaseCrossCorrSettings


def plot_cross_correlation(
    corr,
    title="Cross-Correlation",
    output_path=None,
    xlabel="X shift (pixels)",
    ylabel="Y shift (pixels)",
) -> None:
    """
    Plot the cross-correlation.

    Parameters
    ----------
    corr : ArrayLike
        Cross-correlation array.
    title : str
        Title for the plot.
    output_path : Path
        Path to the output directory.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.

    Returns
    -------
    None
        Saves the plot to the output directory.
    """
    # Convert to 2D if necessary
    if corr.ndim == 3:
        corr_to_plot = np.max(corr, axis=0)  # Or a center slice
    else:
        corr_to_plot = corr

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_to_plot, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation strength")

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    plt.close(fig)  # This prevents overlap in future plots


def phase_cross_corr_padding(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    maximum_shift: float = 1.2,
    normalization: Literal["magnitude", "classic"] | None = None,
    output_path: Path | None = None,
    verbose: bool = False,
) -> tuple[int, ...]:
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2.

    Computes translation shift using arg. maximum of phase cross correlation.
    Input are padded or cropped for fast FFT computation assuming a maximum translation shift.

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
        for s1, s2 in zip(ref_img.shape, mov_img.shape, strict=True)
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
    peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak, strict=True))

    if verbose:
        click.echo(f"phase cross corr. peak at {peak}")
    if output_path:
        plot_cross_correlation(corr, title="Cross-Correlation", output_path=output_path)

    return peak, corr


def phase_cross_corr(
    ref_img: ArrayLike,
    mov_img: ArrayLike,
    normalization: Literal["magnitude", "classic"] | None = None,
    output_path: Path | None = None,
    verbose: bool = False,
) -> tuple[int, ...]:
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2.

    Computes translation shift using arg. maximum of phase cross correlation.
    Input are padded or cropped for fast FFT computation assuming a maximum translation shift.

    Parameters
    ----------
    ref_img : ArrayLike
        Reference image.
    mov_img : ArrayLike
        Moved image.
    normalization : Literal["magnitude", "classic"]
        Normalization method.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    Tuple[int, ...]
        Shift between reference and moved image.
    """
    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    eps = np.finfo(Fimg1.dtype).eps

    prod = Fimg1 * Fimg2.conj()

    if normalization == "magnitude":
        norm = np.fmax(np.abs(prod), eps)
    elif normalization == "classic":
        norm = np.abs(Fimg1) * np.abs(Fimg2)
    else:
        norm = 1.0

    corr = np.fft.irfftn(prod / norm)
    corr_shifted = np.fft.fftshift(np.abs(corr))
    if output_path:
        plot_cross_correlation(
            corr_shifted, title="Cross-Correlation", output_path=output_path
        )
    maxima = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    midpoint = np.array([np.fix(axis_size / 2) for axis_size in corr.shape])

    float_dtype = prod.real.dtype
    del Fimg1, Fimg2, prod, norm

    shift = np.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= np.array(corr.shape)[shift > midpoint]

    return shift, corr_shifted


class PCCEstimator:
    """TransformEstimator using phase cross-correlation (rigid translation only).

    `phase_cross_corr(ref, mov)`'s shift is already the forward (moving -> reference)
    translation in the array's own axis order. `crop_zyx` restricts the correlation to a
    sub-volume (a translation measured on a crop holds for the whole volume).
    """

    def __init__(
        self,
        function_type: Literal["custom", "custom_padding"] = "custom",
        normalization: Literal["magnitude", "classic"] | None = None,
        maximum_shift: float = 1.2,
        crop_zyx: tuple[slice, slice, slice] | None = None,
    ):
        self.function_type = function_type
        self.normalization = normalization
        self.maximum_shift = maximum_shift
        self.crop_zyx = crop_zyx

    @classmethod
    def from_settings(
        cls, settings: PhaseCrossCorrSettings, shape_zyx: tuple[int, int, int] | None = None
    ) -> PCCEstimator:
        """Build from the settings; with `shape_zyx`, resolve the crop the settings describe."""
        return cls(
            function_type=settings.function_type,
            normalization=settings.normalization,
            maximum_shift=settings.maximum_shift,
            crop_zyx=pcc_crop(settings, shape_zyx) if shape_zyx is not None else None,
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        # PCC is correlation-based -- it finds the peak directly, no seed needed.
        mov = np.asarray(mov).astype(np.float32)
        ref = np.asarray(ref).astype(np.float32)
        if self.crop_zyx is not None:
            mov, ref = mov[self.crop_zyx], ref[self.crop_zyx]
        if self.function_type == "custom_padding":
            shift, _corr = phase_cross_corr_padding(
                ref, mov, maximum_shift=self.maximum_shift, normalization=self.normalization
            )
        else:
            shift, _corr = phase_cross_corr(ref, mov, normalization=self.normalization)
        return Transform.from_translation(shift)


def pcc_crop(
    settings: PhaseCrossCorrSettings, shape_zyx: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    """Resolve the sub-volume `PhaseCrossCorrSettings` asks to correlate on.

    Explicit `Z_slice` / `Y_slice` / `X_slice` win; otherwise `center_crop_xy` takes a
    centred window; otherwise the full extent.
    """
    Z, Y, X = shape_zyx
    y_idx, x_idx = slice(0, Y), slice(0, X)
    if settings.center_crop_xy:
        cx, cy = settings.center_crop_xy
        x_idx = slice(X // 2 - cx // 2, X // 2 + cx // 2)
        y_idx = slice(Y // 2 - cy // 2, Y // 2 + cy // 2)
    if settings.X_slice != "all":
        x_idx = slice(settings.X_slice[0], settings.X_slice[1])
    if settings.Y_slice != "all":
        y_idx = slice(settings.Y_slice[0], settings.Y_slice[1])
    z_idx = (
        slice(0, Z)
        if settings.Z_slice == "all"
        else slice(settings.Z_slice[0], settings.Z_slice[1])
    )
    return z_idx, y_idx, x_idx
