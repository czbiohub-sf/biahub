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
from biahub.registration.estimators import EstimationError
from biahub.registration.methods.vote_icp import fit_model
from biahub.registration.utils import match_shape, match_shape_origin
from biahub.settings import LocalPhaseCrossCorrSettings, PhaseCrossCorrSettings


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
    tuple[tuple[int, ...], np.ndarray]
        Shift of the moved image relative to the reference (same convention as
        `phase_cross_corr`), and the shifted correlation volume.
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

    # Resizing moves each origin; with unequal shapes by different amounts, which is not a
    # shift of the content: removed from the peak below.
    origin_offset = match_shape_origin(mov_img.shape, shape) - match_shape_origin(
        ref_img.shape, shape
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
        # guarded as 'magnitude' is: zero-magnitude frequencies would give NaNs
        prod /= np.fmax(np.abs(Fimg1) * np.abs(Fimg2), eps)

    # the spatial shape: without it an odd last axis comes back one sample short
    corr = np.fft.irfftn(prod, s=shape, axes=tuple(range(len(shape))))
    del prod, Fimg1, Fimg2

    corr = np.fft.fftshift(np.abs(corr))

    argmax = np.argmax(corr)
    peak = np.unravel_index(argmax, corr.shape)
    # Same sign convention as `phase_cross_corr`: the shift of mov relative to ref.
    peak = tuple(
        int(p - s // 2 + o) for s, p, o in zip(corr.shape, peak, origin_offset, strict=True)
    )

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
    upsample_factor: int = 1,
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
    upsample_factor : int
        1: whole-pixel shifts (the correlation's arg maximum). Above 1: refined to
        1/upsample_factor of a pixel by an upsampled DFT around that peak (Guizar-Sicairos
        et al. 2008, as skimage.registration.phase_cross_correlation).

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
        norm = np.fmax(np.abs(Fimg1) * np.abs(Fimg2), eps)
    else:
        norm = 1.0

    # the spatial shape: without it an odd last axis comes back one sample short
    corr = np.fft.irfftn(prod / norm, s=np.shape(ref_img), axes=tuple(range(np.ndim(ref_img))))
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
    if upsample_factor > 1:
        shift = _refine_shift(ref_img, mov_img, shift, normalization, upsample_factor)

    return shift, corr_shifted


def _upsampled_dft(data, region_size: int, upsample_factor: int, offsets) -> np.ndarray:
    """Return the cross-correlation upsampled by matrix-multiply DFT in a small region."""
    for n_items, offset in list(zip(data.shape, offsets, strict=True))[::-1]:
        kernel = (np.arange(region_size) - offset)[:, None] * np.fft.fftfreq(
            n_items, upsample_factor
        )
        data = np.tensordot(np.exp(-1j * 2 * np.pi * kernel), data, axes=(1, -1))
    return data


def _refine_shift(ref_img, mov_img, shift, normalization, upsample_factor: int) -> np.ndarray:
    """Return `shift` refined to 1/upsample_factor pixel around the correlation peak."""
    product = np.fft.fftn(ref_img) * np.fft.fftn(mov_img).conj()
    eps = np.finfo(product.real.dtype).eps
    if normalization == "magnitude":
        product = product / np.fmax(np.abs(product), eps)
    elif normalization == "classic":
        norm = np.abs(np.fft.fftn(ref_img)) * np.abs(np.fft.fftn(mov_img))
        product = product / np.fmax(norm, eps)
    shift = np.round(np.asarray(shift, dtype=float) * upsample_factor) / upsample_factor
    region = int(np.ceil(upsample_factor * 1.5))
    centre = np.fix(region / 2.0)
    offsets = centre - shift * upsample_factor
    correlation = _upsampled_dft(product.conj(), region, upsample_factor, offsets).conj()
    peak = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    return shift + (np.stack(peak).astype(float) - centre) / upsample_factor


class PCCEstimator:
    """TransformEstimator using phase cross-correlation (rigid translation only).

    `phase_cross_corr(ref, mov)`'s shift is already the forward (moving -> reference)
    translation in the array's own axis order. `crop_zyx` restricts the correlation to a
    sub-volume (a translation measured on a crop holds for the whole volume).
    """

    # Ignores the seed: repair from other seeds would return the same answer.
    uses_seed = False

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
            if mov.shape != ref.shape:
                raise EstimationError(
                    f"phase cross-correlation without padding needs equal grids, got moving "
                    f"{mov.shape} and reference {ref.shape}: set phase_cross_corr.function_type: "
                    "custom_padding"
                )
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
        # A crop larger than the axis takes the whole axis (a negative start would wrap);
        # the stop is start + size, so an odd size keeps its last pixel.
        cx, cy = (min(c, n) for c, n in zip(settings.center_crop_xy, (X, Y), strict=True))
        x0, y0 = X // 2 - cx // 2, Y // 2 - cy // 2
        x_idx, y_idx = slice(x0, x0 + cx), slice(y0, y0 + cy)
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


class LocalPCCEstimator:
    """TransformEstimator from local phase cross-correlation: tile shifts fitted in yx.

    The moving volume, warped by the current transform, is phase-correlated with the
    reference on a grid of tiles of their max projections (`evaluate.tile_residuals`);
    the shifts are fitted with the transform type in yx and the fit corrects the current
    transform. Max projections hold no depth, so z stays the seed's.

    Each of `rounds` measures the tiles with the current transform and corrects it, so a
    shift larger than a tile's reach in the first round is taken out by the next.
    """

    def __init__(
        self,
        settings: LocalPhaseCrossCorrSettings,
        transform_type: str,
        ref_um: tuple[float, float, float],
    ):
        self.settings = settings
        self.transform_type = transform_type
        self.ref_um = np.asarray(ref_um, float)

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        from biahub.registration.evaluate import tile_residuals  # evaluate imports this module

        s = self.settings
        current = seed if seed is not None else Transform.identity(3)
        for _ in range(s.rounds):
            tiles = tile_residuals(
                current,
                mov,
                ref,
                tuple(self.ref_um),
                compare=s.compare,
                tile_px=s.tile_px,
                grid=s.grid,
                min_ncc=s.min_ncc,
            )
            if tiles["n_tiles"] < s.min_tiles:
                raise EstimationError(
                    f"local phase_cross_corr: {tiles['n_tiles']} tiles correlated (need {s.min_tiles}): too "
                    "little shared structure, or the seed is further off than a tile reaches"
                )
            current = self._correct(current, tiles["residuals"])
        return current

    def _correct(self, current: Transform, residuals: dict) -> Transform:
        """Return `current` corrected by the yx fit of the tile shifts."""
        reference = np.asarray(residuals["position_um"])[:, 1:]
        mapped = reference - np.asarray(residuals["diff_um"])[:, 1:]  # where content landed
        yx = fit_model(mapped, reference, self.transform_type)  # 3 x 3, in µm
        correction_um = np.eye(4)
        correction_um[1:3, 1:3] = yx[:2, :2]
        correction_um[1:3, 3] = yx[:2, 2]
        to_vox = np.diag([*(1.0 / self.ref_um), 1.0])
        from_vox = np.diag([*self.ref_um, 1.0])
        correction = to_vox @ correction_um @ from_vox  # on the reference grid
        return Transform(
            correction @ np.asarray(current.matrix), transform_type=self.transform_type
        )
