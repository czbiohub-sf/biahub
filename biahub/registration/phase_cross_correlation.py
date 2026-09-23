"""Phase cross-correlation registration/stabilization."""

from pathlib import Path
from typing import Literal, cast

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.typing import ArrayLike
from scipy.fftpack import next_fast_len

from biahub.registration.utils import match_shape


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


def get_tform_from_pcc(
    t: int,
    source_channel_tzyx: ArrayLike,
    target_channel_tzyx: ArrayLike,
    function_type: Literal["custom_padding", "custom"] = "custom",
    normalization: Literal["magnitude", "classic"] | None = None,
    output_path: Path | None = None,
    verbose: bool = False,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Get the transformation matrix from phase cross correlation.

    Parameters
    ----------
    t : int
        Time index.
    source_channel_tzyx : ArrayLike
        Source (moving) channel data.
    target_channel_tzyx : ArrayLike
        Target (reference) channel data.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    ArrayLike
        Transformation matrix, in the reference -> moving direction expected by
        this codebase's apply step (`biahub.register.apply_affine_transform`,
        `biahub.stabilize`) -- see the two fixes below.

    Notes
    -----
    Fixes two bugs present in the pre-move version of this function (both confirmed
    empirically against a known synthetic shift, applied through the real
    `apply_affine_transform`):

    1. The old code passed `source`/`target` to `phase_cross_corr` swapped (reference and
       moving reversed), which flipped the sign of the recovered shift on every axis.
    2. The old code built the translation column as `[dx, dy, dz]` instead of
       `[dz, dy, dx]` -- a Z/X axis swap (tracked as issue #356 before this fix).

    A third correction, not part of either original bug report: `phase_cross_corr`'s
    shift is the forward (moving -> reference) translation, but this codebase's apply
    step (like `ants.py:estimate()`, `user_assisted_registration`, and stackreg) expects
    the reference -> moving ("pull") direction directly, with no separate inversion step
    by the caller. Without this inversion, BOTH the old buggy code and a "just the two
    bugs fixed" version are worse than doing nothing when run through the real apply
    step (confirmed: 0.88 and 0.98 relative error respectively, vs 0.85 baseline);
    only fixed + inverted gives exact alignment (0.0 error).
    """
    source = np.asarray(source_channel_tzyx[t]).astype(np.float32)
    target = np.asarray(target_channel_tzyx[t]).astype(np.float32)

    if function_type == "custom_padding":
        shift, corr = phase_cross_corr_padding(
            target, source, normalization=normalization, output_path=output_path
        )
    elif function_type == "custom":
        shift, corr = phase_cross_corr(target, source, normalization=normalization)
    if verbose:
        click.echo(f"Time {t}: shift (dz,dy,dx) = {shift[0]}, {shift[1]}, {shift[2]}")

    transform = np.eye(4)
    transform[:3, 3] = shift
    transform = np.linalg.inv(transform)
    if verbose:
        click.echo(f"transform: {transform}")

    return transform, shift, corr


def plot_pcc_drifts(
    df: pd.DataFrame,
    output_dir: Path,
    label="sample",
    title="PCC Drift Analysis",
    unit: Literal["µm", "px"] = "µm",
    voxel_size: tuple[float, float, float] = (0.174, 0.1494, 0.1494),  # (Z, Y, X) in microns
) -> None:
    """
    Plot the drifts from PCC per timepoint for a single position.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the cross-correlation data.
    output_dir : Path
        Path to the output directory.
    label : str
        Label for the plot.
    title : str
        Title for the plot.
    voxel_size : Tuple[float, float, float]
        Voxel size in microns.

    Returns
    -------
    None
        Saves the plot to the output directory.
    """
    if unit == "µm":
        # Unpack voxel sizes
        z_scale, y_scale, x_scale = voxel_size

        # Convert to microns
        df["ShiftX"] = df["ShiftX"] * x_scale
        df["ShiftY"] = df["ShiftY"] * y_scale
        df["ShiftZ"] = df["ShiftZ"] * z_scale

    # Cumulative and magnitude drift
    df["CumulativeShiftX"] = df["ShiftX"].cumsum()
    df["CumulativeShiftY"] = df["ShiftY"].cumsum()
    df["CumulativeShiftZ"] = df["ShiftZ"].cumsum()
    df["DriftMagnitude"] = np.sqrt(df["ShiftX"] ** 2 + df["ShiftY"] ** 2 + df["ShiftZ"] ** 2)
    df["CumulativeDrift"] = np.sqrt(
        df["CumulativeShiftX"] ** 2 + df["CumulativeShiftY"] ** 2 + df["CumulativeShiftZ"] ** 2
    )

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(df["TimepointID"], df["ShiftX"], label="ShiftX")
    axs[0].plot(df["TimepointID"], df["ShiftY"], label="ShiftY")
    axs[0].plot(df["TimepointID"], df["ShiftZ"], label="ShiftZ")
    axs[0].set_ylabel(f"Shift ({unit})")
    axs[0].legend()
    axs[0].set_title("Raw Drift per Axis")
    axs[0].grid(True)

    axs[1].plot(df["TimepointID"], df["CumulativeShiftX"], label="Cumulative X")
    axs[1].plot(df["TimepointID"], df["CumulativeShiftY"], label="Cumulative Y")
    axs[1].plot(df["TimepointID"], df["CumulativeShiftZ"], label="Cumulative Z")
    axs[1].set_ylabel(f"Cumulative Shift ({unit})")
    axs[1].legend()
    axs[1].set_title("Cumulative Shift")
    axs[1].grid(True)

    axs[2].plot(df["TimepointID"], df["DriftMagnitude"], label="Instantaneous", color="gray")
    axs[2].plot(
        df["TimepointID"],
        df["CumulativeDrift"],
        label="Cumulative",
        color="black",
        linestyle="--",
    )
    axs[2].set_ylabel(f"Drift Magnitude ({unit})")
    axs[2].legend()
    axs[2].set_title("Drift Magnitude")
    axs[2].grid(True)
    axs[2].set_xlabel("Timepoints")

    fig.suptitle(f"{title} - {label}", fontsize=16, y=1.02)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{label}.png", bbox_inches="tight")
    plt.close(fig)


def plot_corr_max_min_sum(
    corr_df: pd.DataFrame, output_path: Path, label="sample", title="Cross-Correlation Summary"
) -> None:
    """
    Plot the max, min, and sum of the cross-correlation from PCC per timepoint for a single position.

    Parameters
    ----------
    corr_df : pd.DataFrame
        DataFrame containing the cross-correlation data.
    output_path : Path
        Path to the output directory.
    label : str
        Label for the plot.

    Returns
    -------
    None
        Saves the plot to the output directory.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(corr_df["TimepointID"], corr_df["max"], label="Corr Max")
    axs[1].plot(corr_df["TimepointID"], corr_df["min"], label="Corr Min")
    axs[2].plot(corr_df["TimepointID"], corr_df["sum"], label="Corr Sum")
    axs[0].set_ylabel("Max")
    axs[1].set_ylabel("Min")
    axs[2].set_ylabel("Sum")
    axs[0].set_title("Corr Max")
    axs[1].set_title("Corr Min")
    axs[2].set_title("Corr Sum")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    axs[2].set_xlabel("Timepoint")
    fig.suptitle(f"{title} - {label}", y=1.02, fontsize=16)
    fig.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / f"{label}.png", bbox_inches="tight")
    plt.close(fig)
