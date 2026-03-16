import datetime
import gc
import importlib.resources as pkg_resources
import pickle
import shutil
import time
import warnings
import webbrowser

from pathlib import Path
from typing import Literal, Optional

import click
import markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from iohub.ngff import open_ome_zarr
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.signal import peak_widths

import biahub.artefacts

from biahub.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from biahub.cli.utils import yaml_to_model
from biahub.settings import CharacterizeSettings
from biahub.vendor.napari_psf_analysis import PSF, BeadExtractor, Calibrated3DImage


def _make_plots(
    output_path: Path,
    beads: list[ArrayLike],
    df_gaussian_fit: pd.DataFrame,
    df_1d_peak_width: pd.DataFrame,
    scale: tuple[float, float, float],
    axis_labels: tuple[str, str, str],
    fwhm_plot_type: Literal['1D', '3D'],
) -> tuple[Path, list[Path], tuple[Path, Path]]:
    plots_dir = output_path / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    random_bead_number = sorted(np.random.choice(len(beads), 5, replace=False))

    bead_psf_slices_path = plot_psf_slices(
        plots_dir,
        [beads[i] for i in random_bead_number],
        scale,
        axis_labels,
        random_bead_number,
    )

    if fwhm_plot_type == '1D':
        plot_data_x = [df_1d_peak_width[col].values for col in ('x_mu', 'y_mu', 'z_mu')]
        plot_data_y = [
            df_1d_peak_width[col].values for col in ('1d_x_fwhm', '1d_y_fwhm', '1d_z_fwhm')
        ]
    elif fwhm_plot_type == '3D':
        plot_data_x = [df_gaussian_fit[col].values for col in ('x_mu', 'y_mu', 'z_mu')]
        plot_data_y = [
            df_gaussian_fit[col].values for col in ('zyx_x_fwhm', 'zyx_y_fwhm', 'zyx_z_fwhm')
        ]
    else:
        raise ValueError(f'Invalid fwhm_plot_type: {fwhm_plot_type}')

    fwhm_vs_acq_axes_paths = plot_fwhm_vs_acq_axes(
        plots_dir,
        *plot_data_x,
        *plot_data_y,
        axis_labels,
    )

    psf_amp_paths = plot_psf_amp(
        plots_dir,
        df_gaussian_fit['x_mu'].values,
        df_gaussian_fit['y_mu'].values,
        df_gaussian_fit['z_mu'].values,
        df_gaussian_fit['zyx_amp'].values,
        axis_labels,
    )

    return (bead_psf_slices_path, fwhm_vs_acq_axes_paths, psf_amp_paths)


def generate_report(
    output_path: Path,
    data_dir: Path,
    dataset: str,
    beads: list[ArrayLike],
    peaks: ArrayLike,
    df_gaussian_fit: pd.DataFrame,
    df_1d_peak_width: pd.DataFrame,
    scale: tuple[float, float, float],
    axis_labels: tuple[str, str, str],
    fwhm_plot_type: str,
) -> None:
    """
    Generate a comprehensive PSF analysis report with plots, statistics, and HTML output.

    Creates an HTML report containing PSF analysis results including bead statistics,
    FWHM measurements (3D, principal components, and 1D), SNR statistics, and
    visualization plots. Saves results to CSV files and opens the HTML report in
    a web browser.

    Parameters
    ----------
    output_path : Path
        Directory path where the report and output files will be saved.
    data_dir : Path
        Path to the data directory containing the original dataset.
    dataset : str
        Name or identifier of the dataset being analyzed.
    beads : list[ArrayLike]
        List of bead patch arrays extracted from the dataset.
    peaks : ArrayLike
        Array of peak coordinates detected in the dataset.
    df_gaussian_fit : pd.DataFrame
        DataFrame containing Gaussian fit results for each bead.
    df_1d_peak_width : pd.DataFrame
        DataFrame containing 1D peak width measurements.
    scale : tuple
        Tuple representing the voxel scaling factors for each dimension (Z, Y, X).
    axis_labels : tuple
        Tuple of axis label strings for the dimensions (e.g., ('Z', 'Y', 'X')).
    fwhm_plot_type : str
        Type of FWHM plot to generate (e.g., 'acquisition_axes' or 'principal_components').

    Returns
    -------
    None
        Results are saved to disk and the HTML report is opened in a web browser.
        Output files include:
        - psf_analysis_report.html: Main HTML report
        - psf_gaussian_fit.csv: Gaussian fit results
        - psf_1d_peak_width.csv: 1D peak width measurements
        - peaks.pkl: Pickled peak coordinates
        - Various plot images (PSF slices, FWHM plots, amplitude plots)
    """
    output_path.mkdir(exist_ok=True)

    num_beads = len(beads)
    num_successful = len(df_gaussian_fit)
    num_failed = num_beads - num_successful

    # make plots
    (bead_psf_slices_path, fwhm_vs_acq_axes_paths, psf_amp_paths) = _make_plots(
        output_path,
        beads,
        df_gaussian_fit,
        df_1d_peak_width,
        scale,
        axis_labels,
        fwhm_plot_type,
    )

    # calculate statistics
    fwhm_3d_mean = [
        df_gaussian_fit[col].mean() for col in ('zyx_z_fwhm', 'zyx_y_fwhm', 'zyx_x_fwhm')
    ]
    fwhm_3d_std = [
        df_gaussian_fit[col].std() for col in ('zyx_z_fwhm', 'zyx_y_fwhm', 'zyx_x_fwhm')
    ]
    fwhm_pc_mean = [
        df_gaussian_fit[col].mean() for col in ('zyx_pc3_fwhm', 'zyx_pc2_fwhm', 'zyx_pc1_fwhm')
    ]
    fwhm_1d_mean = [
        df_1d_peak_width[col].mean() for col in ('1d_z_fwhm', '1d_y_fwhm', '1d_x_fwhm')
    ]
    fwhm_1d_std = [
        df_1d_peak_width[col].std() for col in ('1d_z_fwhm', '1d_y_fwhm', '1d_x_fwhm')
    ]
    snr_mean = df_gaussian_fit['zyx_snr'].mean()
    snr_std = df_gaussian_fit['zyx_snr'].std()

    # generate html report
    html_report = _generate_html(
        dataset,
        data_dir,
        scale,
        (num_beads, num_successful, num_failed),
        snr_mean,
        snr_std,
        fwhm_1d_mean,
        fwhm_1d_std,
        fwhm_3d_mean,
        fwhm_3d_std,
        fwhm_pc_mean,
        str(bead_psf_slices_path.relative_to(output_path).as_posix()),
        [str(_path.relative_to(output_path).as_posix()) for _path in fwhm_vs_acq_axes_paths],
        [str(_path.relative_to(output_path).as_posix()) for _path in psf_amp_paths],
        axis_labels,
        fwhm_plot_type,
    )

    # save html report and other results
    with open(output_path / 'peaks.pkl', 'wb') as file:
        pickle.dump(peaks, file)

    df_gaussian_fit.to_csv(output_path / 'psf_gaussian_fit.csv', index=False)
    df_1d_peak_width.to_csv(output_path / 'psf_1d_peak_width.csv', index=False)

    with pkg_resources.path(biahub.artefacts, 'github-markdown.css') as css_path:
        shutil.copy(css_path, output_path)
    html_file_path = output_path / ('psf_analysis_report.html')
    with open(html_file_path, 'w') as file:
        file.write(html_report)

    # display html report
    html_file_path = Path(html_file_path).absolute()
    webbrowser.open(html_file_path.as_uri())


def extract_beads(
    zyx_data: ArrayLike,
    points: ArrayLike,
    scale: tuple[float, float, float],
    patch_size: Optional[tuple[float, float, float]] = None,
) -> tuple[list[ArrayLike], list[tuple[float, float, float]]]:
    if patch_size is None:
        patch_size = (scale[0] * 15, scale[1] * 18, scale[2] * 18)

    # extract bead patches
    bead_extractor = BeadExtractor(
        image=Calibrated3DImage(data=zyx_data, spacing=scale),
        patch_size=patch_size,
    )
    beads = bead_extractor.extract_beads(points=points)
    # remove bad beads
    beads = [bead for bead in beads if bead.data.size > 0]
    beads_data = [bead.data for bead in beads]
    bead_offset = [bead.offset for bead in beads]

    return beads_data, bead_offset


def analyze_psf(
    zyx_patches: list[ArrayLike],
    peak_coordinates: list[tuple[int, int, int]],
    scale: tuple[float, float, float],
    offset: float = 0.0,
    gain: float = 1.0,
    noise: float = 1.0,
    use_robust_1d_fwhm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze point spread function (PSF) from given 3D patches.

    Parameters
    ----------
    zyx_patches : list[ArrayLike]
        List of 3D image patches to be analyzed.
    peak_coordinates : list[tuple]
        List of tuples representing the global ZYX coordinates of the peaks in the data.
    scale : tuple
        Tuple representing the scaling factors for each dimension (Z, Y, X).
    offset : float
        Offset value to be added to the patches.
    gain : float
        Gain value to be multiplied with the patches after applying offset.
    noise : float
        Noise level in the data.
    use_robust_1d_fwhm : bool
        If True, use the "robust" 1D FWHM calculation method.

    Returns
    -------
    df_gaussian_fit : pandas.DataFrame
        DataFrame containing the results of the Gaussian fit analysis.
    df_1d_peak_width : pandas.DataFrame
        DataFrame containing the 1D peak width calculations.
    """
    if use_robust_1d_fwhm:
        f_1d_peak_width = calculate_robust_peak_widths
    else:
        f_1d_peak_width = calculate_peak_widths

    results = []
    peak_coordinates = np.asarray(peak_coordinates)
    for patch, peak_coords in zip(zyx_patches, peak_coordinates):
        patch = (patch + offset) * gain
        patch = np.clip(patch, 0, None).astype(np.int32)
        bead = Calibrated3DImage(data=patch, spacing=scale, offset=peak_coords)
        psf = PSF(image=bead)
        try:
            psf.analyze()
            summary_dict = psf.get_summary_dict()
        except Exception:
            summary_dict = {}
        results.append(summary_dict)

    df_gaussian_fit = pd.DataFrame.from_records(results)
    df_gaussian_fit['z_mu'] += peak_coordinates[:, 0] * scale[0]
    df_gaussian_fit['y_mu'] += peak_coordinates[:, 1] * scale[1]
    df_gaussian_fit['x_mu'] += peak_coordinates[:, 2] * scale[2]
    df_gaussian_fit['z_amp'] /= gain  # amplitude is measured relative to the offset
    df_gaussian_fit['zyx_amp'] /= gain

    df_1d_peak_width = pd.DataFrame(
        [f_1d_peak_width(zyx_patch, scale) for zyx_patch in zyx_patches],
        columns=(f'1d_{i}_fwhm' for i in ('z', 'y', 'x')),
    )
    df_1d_peak_width = pd.concat(
        (df_gaussian_fit[['z_mu', 'y_mu', 'x_mu']], df_1d_peak_width), axis=1
    )

    # clean up dataframes
    df_gaussian_fit = df_gaussian_fit.dropna()
    df_1d_peak_width = df_1d_peak_width.dropna().loc[
        ~(df_1d_peak_width[['1d_z_fwhm', '1d_y_fwhm', '1d_x_fwhm']] == 0).any(axis=1)
    ]

    # compute peak SNR
    df_gaussian_fit['zyx_snr'] = df_gaussian_fit['zyx_amp'] / noise

    return df_gaussian_fit, df_1d_peak_width


def compute_noise_level(
    zyx_data: ArrayLike,
    peak_coordinates: list[tuple[int, int, int]],
    patch_size_pix: tuple[int, int, int],
) -> float:
    """
    Compute the noise level in the data by masking out peak regions.

    Creates a mask that excludes regions around detected peaks and calculates
    the standard deviation of the remaining background pixels as a measure
    of noise level.

    Parameters
    ----------
    zyx_data : ArrayLike
        3D array of image data with shape (Z, Y, X).
    peak_coordinates : List[tuple]
        List of tuples containing (z, y, x) coordinates of detected peaks.
    patch_size_pix : tuple
        Tuple of patch sizes in pixels for each dimension (Z, Y, X).

    Returns
    -------
    float
        Standard deviation of the background pixels (noise level).
    """
    # Mask out the peaks
    mask = np.ones_like(zyx_data, dtype=bool)
    half_patch = [size // 2 for size in patch_size_pix]

    for z, y, x in peak_coordinates:
        patch_mask = tuple(
            slice(
                max(0, coord - half_patch[i]),
                min(zyx_data.shape[i], coord + half_patch[i] + 1),
            )
            for i, coord in enumerate((z, y, x))
        )
        mask[patch_mask] = False

    return np.std(zyx_data[mask])


def calculate_robust_peak_widths(
    zyx_data: ArrayLike, zyx_scale: tuple[float, float, float]
) -> list[float]:
    """
    Calculate full width at half maximum (FWHM) using a robust method.

    Uses parabola fitting to find the peak position and linear interpolation
    to determine the FWHM along each axis. This method is more robust to noise
    than direct peak finding methods.

    Parameters
    ----------
    zyx_data : ArrayLike
        3D array of image data with shape (Z, Y, X).
    zyx_scale : tuple
        Tuple of voxel scaling factors for each dimension (Z, Y, X).

    Returns
    -------
    list[float]
        List of FWHM values in physical units for each dimension [Z, Y, X].
        Returns [0.0, 0.0, 0.0] if calculation fails for any dimension.
    """
    shape_Z, shape_Y, shape_X = zyx_data.shape

    slices = (
        (slice(None), shape_Y // 2, shape_X // 2),
        (shape_Z // 2, slice(None), shape_X // 2),
        (shape_Z // 2, shape_Y // 2, slice(None)),
    )

    fwhm = []
    for _slice, _scale in zip(slices, zyx_scale):
        try:
            y = zyx_data[_slice]
            x = np.arange(y.size)

            # fit parabola to nearest 5 points to find peak
            peak_index = np.argmax(y)
            fit_range = (slice(max(0, peak_index - 2), min(peak_index + 2, y.size)),)
            p = np.polyfit(x[fit_range], y[fit_range], 2)
            peak_index = -p[1] / (2 * p[0])
            peak_max = np.polyval(p, peak_index)
            half_max = peak_max / 2

            # Use linear interpolation on each side of the peak to find FWHM
            x = x * _scale
            indices = np.where(y >= half_max / 2)[0]
            _il = indices[indices < peak_index]
            _ir = indices[indices > peak_index]
            _fl = interp1d(y[_il], x[_il], kind="linear", fill_value="extrapolate")
            _fr = interp1d(y[_ir], x[_ir], kind="linear", fill_value="extrapolate")
            x_left = _fl(half_max)  # Left crossing
            x_right = _fr(half_max)  # Right crossing

            fwhm.append(x_right - x_left)  # Compute width
        except Exception:
            fwhm.append(0.0)

    return fwhm


def calculate_peak_widths(
    zyx_data: ArrayLike, zyx_scale: tuple[float, float, float]
) -> tuple[float, float, float]:
    """
    Calculate full width at half maximum (FWHM) using scipy's peak_widths function.

    Extracts 1D profiles along each axis through the center of the data and
    calculates FWHM using scipy.signal.peak_widths.

    Parameters
    ----------
    zyx_data : ArrayLike
        3D array of image data with shape (Z, Y, X).
    zyx_scale : tuple
        Tuple of voxel scaling factors for each dimension (Z, Y, X).

    Returns
    -------
    tuple[float, float, float]
        Tuple of FWHM values in physical units for (Z, Y, X) dimensions.
        Returns (0.0, 0.0, 0.0) if calculation fails.
    """
    scale_Z, scale_Y, scale_X = zyx_scale
    shape_Z, shape_Y, shape_X = zyx_data.shape

    try:
        z_fwhm = peak_widths(zyx_data[:, shape_Y // 2, shape_X // 2], [shape_Z // 2])[0][0]
        y_fwhm = peak_widths(zyx_data[shape_Z // 2, :, shape_X // 2], [shape_Y // 2])[0][0]
        x_fwhm = peak_widths(zyx_data[shape_Z // 2, shape_Y // 2, :], [shape_X // 2])[0][0]
    except Exception:
        z_fwhm, y_fwhm, x_fwhm = (0.0, 0.0, 0.0)

    return z_fwhm * scale_Z, y_fwhm * scale_Y, x_fwhm * scale_X


def plot_psf_slices(
    plots_dir: Path,
    beads: list[ArrayLike],
    zyx_scale: tuple[float, float, float],
    axis_labels: tuple[str, str, str],
    bead_numbers: list[int],
) -> Path:
    """
    Generate a multi-panel plot showing PSF slices for multiple beads.

    Creates a figure with 3 rows (XY, XZ, YZ slices) and N columns (one per bead),
    displaying the point spread function at the center slice of each dimension.

    Parameters
    ----------
    plots_dir : str
        Directory path where the plot will be saved.
    beads : List[ArrayLike]
        List of 3D bead patch arrays, each with shape (Z, Y, X).
    zyx_scale : tuple
        Tuple of voxel scaling factors for each dimension (Z, Y, X).
    axis_labels : tuple
        Tuple of axis label strings for the dimensions (e.g., ('Z', 'Y', 'X')).
    bead_numbers : list
        List of bead identifiers/numbers for labeling each column.

    Returns
    -------
    Path
        Path to the saved plot file (beads_psf_slices.png).
    """
    num_beads = len(beads)
    scale_Z, scale_Y, scale_X = zyx_scale
    shape_Z, shape_Y, shape_X = beads[0].shape
    cmap = 'viridis'

    bead_psf_slices_path = plots_dir / 'beads_psf_slices.png'
    fig, ax = plt.subplots(3, num_beads)
    for _ax, bead, bead_number in zip(ax[0], beads, bead_numbers):
        _ax.imshow(
            bead[shape_Z // 2, :, :],
            cmap=cmap,
            origin='lower',
            aspect=scale_Y / scale_X,
        )
        _ax.set_xlabel(axis_labels[-1])
        _ax.set_ylabel(axis_labels[-2])
        _ax.set_title(f'Bead: {bead_number}')

    for _ax, bead in zip(ax[1], beads):
        _ax.imshow(
            bead[:, shape_Y // 2, :], cmap=cmap, origin='lower', aspect=scale_Z / scale_X
        )
        _ax.set_xlabel(axis_labels[-1])
        _ax.set_ylabel(axis_labels[-3])

    for _ax, bead in zip(ax[2], beads):
        _ax.imshow(
            bead[:, :, shape_X // 2], cmap=cmap, origin='lower', aspect=scale_Z / scale_Y
        )
        _ax.set_xlabel(axis_labels[-2])
        _ax.set_ylabel(axis_labels[-3])

    for _ax in ax.flatten():
        _ax.set_xticks([])
        _ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    fig_size = fig.get_size_inches()
    fig_size_scaling = 8 / fig_size[0]  # set width to 8 inches
    fig.set_figwidth(fig_size[0] * fig_size_scaling)
    fig.set_figheight(fig_size[1] * fig_size_scaling)

    fig.savefig(bead_psf_slices_path)

    return bead_psf_slices_path


def plot_fwhm_vs_acq_axes(
    plots_dir: Path,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    fwhm_x: ArrayLike,
    fwhm_y: ArrayLike,
    fwhm_z: ArrayLike,
    axis_labels: tuple[str, str, str],
) -> list[Path]:
    """
    Plot FWHM measurements as a function of acquisition position along each axis.

    Creates separate plots for each acquisition axis showing how FWHM varies
    with position. Each plot shows FWHM for two axes on the primary y-axis
    and FWHM for the third axis on a secondary y-axis.

    Parameters
    ----------
    plots_dir : str
        Directory path where plots will be saved.
    x : ArrayLike
        X-coordinates of measurement positions.
    y : ArrayLike
        Y-coordinates of measurement positions.
    z : ArrayLike
        Z-coordinates of measurement positions.
    fwhm_x : ArrayLike
        FWHM values measured along the X dimension.
    fwhm_y : ArrayLike
        FWHM values measured along the Y dimension.
    fwhm_z : ArrayLike
        FWHM values measured along the Z dimension.
    axis_labels : tuple
        Tuple of axis label strings for the dimensions (e.g., ('Z', 'Y', 'X')).

    Returns
    -------
    list[Path]
        List of paths to the saved plot files (one per axis).
    """

    def plot_fwhm_vs_acq_axis(out_dir: str, x, fwhm_x, fwhm_y, fwhm_z, x_axis_label: str):
        fig, ax = plt.subplots(1, 1)
        artist1 = ax.plot(x, fwhm_x, 'o', x, fwhm_y, 'o')
        ax.set_ylabel('{} and {} FWHM (um)'.format(*axis_labels[1:][::-1]))
        ax.set_xlabel('{} position (um)'.format(x_axis_label))

        ax2 = ax.twinx()
        artist2 = ax2.plot(x, fwhm_z, 'o', color='green')
        ax2.set_ylabel('{} FWHM (um)'.format(axis_labels[0]), color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        plt.legend(artist1 + artist2, axis_labels[::-1])
        fig.savefig(out_dir)

    out_dirs = [plots_dir / f'fwhm_vs_{axis}.png' for axis in axis_labels]
    for our_dir, x_axis, x_axis_label in zip(out_dirs, (z, y, x), axis_labels):
        plot_fwhm_vs_acq_axis(our_dir, x_axis, fwhm_x, fwhm_y, fwhm_z, x_axis_label)

    return out_dirs


def plot_psf_amp(
    plots_dir: Path,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    amp: ArrayLike,
    axis_labels: tuple[str, str, str],
) -> tuple[Path, Path]:
    """
    Plot PSF amplitude as a function of spatial position.

    Creates two plots: one showing amplitude in the XY plane as a scatter plot
    with color-coded amplitude, and another showing amplitude vs Z position.

    Parameters
    ----------
    plots_dir : str
        Directory path where plots will be saved.
    x : ArrayLike
        X-coordinates of PSF measurements.
    y : ArrayLike
        Y-coordinates of PSF measurements.
    z : ArrayLike
        Z-coordinates of PSF measurements.
    amp : ArrayLike
        Amplitude values for each PSF measurement.
    axis_labels : tuple
        Tuple of axis label strings for the dimensions (e.g., ('Z', 'Y', 'X')).

    Returns
    -------
    tuple[Path, Path]
        Tuple of paths to the saved plot files (psf_amp_xy.png, psf_amp_z.png).
    """
    psf_amp_xy_path = plots_dir / 'psf_amp_xy.png'
    fig, ax = plt.subplots(1, 1)

    sc = ax.scatter(
        x,
        y,
        c=amp,
        vmin=np.quantile(amp, 0.01),
        vmax=np.quantile(amp, 0.99),
        cmap='summer',
    )
    ax.set_aspect('equal')
    ax.set_xlabel(f'{axis_labels[-1]} (um)')
    ax.set_ylabel(f'{axis_labels[-2]} (um)')
    plt.colorbar(sc, label='Amplitude (a.u.)')
    fig.savefig(psf_amp_xy_path)

    psf_amp_z_path = plots_dir / 'psf_amp_z.png'
    fig, ax = plt.subplots(1, 1)
    ax.scatter(z, amp)
    ax.set_xlabel(f'{axis_labels[-3]} (um)')
    ax.set_ylabel('Amplitude (a.u.)')
    fig.savefig(psf_amp_z_path)

    return psf_amp_xy_path, psf_amp_z_path


def _generate_html(
    dataset_name: str,
    data_path: str,
    dataset_scale: tuple[float, float, float],
    num_beads_total_good_bad: tuple[int, int, int],
    snr_mean: float,
    snr_std: float,
    fwhm_1d_mean: tuple[float, float, float],
    fwhm_1d_std: tuple[float, float, float],
    fwhm_3d_mean: tuple[float, float, float],
    fwhm_3d_std: tuple[float, float, float],
    fwhm_pc_mean: tuple[float, float, float],
    bead_psf_slices_path: str,
    fwhm_vs_acq_axes_paths: list[str],
    psf_amp_paths: list[str],
    axis_labels: tuple[str, str, str],
    fwhm_plot_type: str,
) -> str:

    # string indents need to be like that, otherwise this turns into a code block
    report_str = f'''
# PSF Analysis

## Overview

### Dataset

* Name: `{dataset_name}`
* Path: `{data_path}`
* Scale (z, y, x): {tuple(np.round(dataset_scale, 3))} um
* Date analyzed: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Number of beads

* Detected: {num_beads_total_good_bad[0]}
* Analyzed: {num_beads_total_good_bad[1]}
* Skipped: {num_beads_total_good_bad[2]}
* Signal-to-noise ratio: {round(snr_mean)} ± {round(snr_std)}

### FWHM

* **3D Gaussian fit**
    - {axis_labels[-1]}: {fwhm_3d_mean[-1]:.3f} ± {fwhm_3d_std[0]:.3f} um
    - {axis_labels[-2]}: {fwhm_3d_mean[-2]:.3f} ± {fwhm_3d_std[1]:.3f} um
    - {axis_labels[-3]}: {fwhm_3d_mean[-3]:.3f} ± {fwhm_3d_std[2]:.3f} um
* 1D profile
    - {axis_labels[-1]}: {fwhm_1d_mean[-1]:.3f} ± {fwhm_1d_std[0]:.3f} um
    - {axis_labels[-2]}: {fwhm_1d_mean[-2]:.3f} ± {fwhm_1d_std[1]:.3f} um
    - {axis_labels[-3]}: {fwhm_1d_mean[-3]:.3f} ± {fwhm_1d_std[2]:.3f} um
* 3D principal components
    - {'{:.3f} um, {:.3f} um, {:.3f} um'.format(*fwhm_pc_mean)}

## Representative bead PSF images
![beads psf slices]({bead_psf_slices_path})

## {fwhm_plot_type} FWHM versus {axis_labels[0]} position
![fwhm vs z]({fwhm_vs_acq_axes_paths[0]} "fwhm vs z")

## {fwhm_plot_type} FWHM versus {axis_labels[1]} position
![fwhm vs z]({fwhm_vs_acq_axes_paths[1]} "fwhm vs y")

## {fwhm_plot_type} FWHM versus {axis_labels[2]} position
![fwhm vs z]({fwhm_vs_acq_axes_paths[2]} "fwhm vs x")

## PSF amplitude versus {axis_labels[-1]}-{axis_labels[-2]} position
![psf amp xy]({psf_amp_paths[0]} "psf amp xy")

## PSF amplitude versus {axis_labels[-3]} position
![psf amp z]({psf_amp_paths[1]} "psf amp z")
'''

    css_style = '''
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="github-markdown.css">
<style>
    .markdown-body {
        box-sizing: border-box;
        min-width: 200px;
        max-width: 980px;
        margin: 0 auto;
        padding: 45px;
    }

    @media (max-width: 767px) {
        .markdown-body {
            padding: 15px;
        }
    }
</style>
'''

    head = f'''
<head>
    <title>PSF Analysis: {dataset_name}</title>
</head>
    '''

    html = markdown.markdown(report_str)
    formatted_html = f'''
{css_style}
{head}
<article class="markdown-body">
{html}
</article>
'''.strip()

    return formatted_html


def detect_peaks(
    zyx_data: np.ndarray,
    block_size: int | tuple[int, int, int] = (8, 8, 8),
    nms_distance: int = 3,
    min_distance: int = 40,
    threshold_abs: float = 200.0,
    max_num_peaks: int = 500,
    exclude_border: tuple[int, int, int] | None = None,
    blur_kernel_size: int = 3,
    device: str = "cpu",
    verbose: bool = False,
):
    """Detect peaks with local maxima.
    This is an approximate torch implementation of `skimage.feature.peak_local_max`.
    The algorithm works well with small kernel size, by default (8, 8, 8) which
    generates a large number of peak candidates, and strict peak rejection criteria
    - e.g. max_num_peaks=500, which selects top 500 brightest peaks and
    threshold_abs=200.0, which selects peaks with intensity of at least 200 counts.

    Parameters
    ----------
    zyx_data : np.ndarray
        3D image data
    block_size : int | tuple[int, int, int], optional
        block size to find approximate local maxima, by default (8, 8, 8)
    nms_distance : int, optional
        non-maximum suppression distance, by default 3
        distance is calculated assuming a Cartesian coordinate system
    min_distance : int, optional
        minimum distance between detections,
        distance needs to be smaller than block size for efficiency,
        by default 40
    threshold_abs : float, optional
        lower bound of detected peak intensity, by default 200.0
    max_num_peaks : int, optional
        max number of candidate detections to consider, by default 500
    exclude_border : tuple[int, int, int] | None, optional
        width of borders to exclude, by default None
    blur_kernel_size : int, optional
        uniform kernel size to blur the image before detection
        to avoid hot pixels, by default 3
    device : str, optional
        compute device string for torch,
        e.g. "cpu" (slow), "cuda" (single GPU) or "cuda:0" (0th GPU among multiple),
        by default "cpu"
    verbose : bool, optional
        print number of peaks detected and rejected, by default False

    Returns
    -------
    np.ndarray
        3D coordinates of detected peaks (N, 3)

    """
    zyx_shape = zyx_data.shape[-3:]
    zyx_image = torch.from_numpy(zyx_data.astype(np.float32)[None, None])

    if device != "cpu":
        zyx_image = zyx_image.to(device)

    if blur_kernel_size:
        if blur_kernel_size % 2 != 1:
            raise ValueError(f"kernel_size={blur_kernel_size} must be an odd number")
        # smooth image
        # input and output variables need to be different for proper memory clearance
        smooth_image = F.avg_pool3d(
            input=zyx_image,
            kernel_size=blur_kernel_size,
            stride=1,
            padding=blur_kernel_size // 2,
            count_include_pad=False,
        )

    # detect peaks as local maxima
    peak_value, peak_idx = (
        p.flatten().clone()
        for p in F.max_pool3d(
            smooth_image,
            kernel_size=block_size,
            stride=block_size,
            padding=(block_size[0] // 2, block_size[1] // 2, block_size[2] // 2),
            return_indices=True,
        )
    )
    num_peaks = len(peak_idx)

    # select only top max_num_peaks brightest peaks
    # peak_value (and peak_idx) are now sorted by brightness
    peak_value, sort_mask = peak_value.topk(min(max_num_peaks, peak_value.nelement()))
    peak_idx = peak_idx[sort_mask]
    num_rejected_max_num_peaks = num_peaks - len(sort_mask)

    # select only peaks above intensity threshold
    num_rejected_threshold_abs = 0
    if threshold_abs:
        abs_mask = peak_value > threshold_abs
        peak_value = peak_value[abs_mask]
        peak_idx = peak_idx[abs_mask]
        num_rejected_threshold_abs = sum(~abs_mask)

    # remove artifacts of multiple peaks detected at block boundaries
    # requires torch>=2.2
    coords = torch.stack(torch.unravel_index(peak_idx, zyx_shape), -1)
    fcoords = coords.float()
    dist = torch.cdist(fcoords, fcoords)
    dist_mask = torch.ones(len(coords), dtype=bool, device=device)

    nearby_peaks = torch.nonzero(torch.triu(dist < nms_distance, diagonal=1))
    dist_mask[nearby_peaks[:, 1]] = False  # peak in second column is dimmer
    num_rejected_nms_distance = sum(~dist_mask)

    # remove peaks withing min_distance of each other
    num_rejected_min_distance = 0
    if min_distance:
        _dist_mask = dist < min_distance
        # exclude distances from nearby peaks rejected above
        _dist_mask[nearby_peaks[:, 0], nearby_peaks[:, 1]] = False
        dist_mask &= _dist_mask.sum(1) < 2  # Ziwen magic
        num_rejected_min_distance = sum(~dist_mask) - num_rejected_nms_distance
    coords = coords[dist_mask]

    # remove peaks near the border
    num_rejected_exclude_border = 0
    match exclude_border:
        case None:
            pass
        case (int(), int(), int()):
            for dim, size in enumerate(exclude_border):
                border_mask = (size < coords[:, dim]) & (
                    coords[:, dim] < zyx_shape[dim] - size
                )
                coords = coords[border_mask]
                num_rejected_exclude_border += sum(~border_mask)
        case _:
            raise ValueError(f"invalid argument exclude_border={exclude_border}")

    num_peaks_returned = len(coords)
    if verbose:
        print(f'Number of peaks detected: {num_peaks}')
        print(f'Number of peaks rejected by max_num_peaks: {num_rejected_max_num_peaks}')
        print(f'Number of peaks rejected by threshold_abs: {num_rejected_threshold_abs}')
        print(f'Number of peaks rejected by nms_distance: {num_rejected_nms_distance}')
        print(f'Number of peaks rejected by min_distance: {num_rejected_min_distance}')
        print(f'Number of peaks rejected by exclude_border: {num_rejected_exclude_border}')
        print(f'Number of peaks returned: {num_peaks_returned}')

    del zyx_image, smooth_image
    return coords.cpu().numpy()


def _characterize_psf(
    zyx_data: np.ndarray,
    zyx_scale: tuple[float, float, float],
    settings: CharacterizeSettings,
    output_report_path: str,
    input_dataset_path: str,
    input_dataset_name: str,
) -> np.ndarray:
    settings_dict = settings.model_dump()
    patch_size = settings_dict.pop("patch_size")
    axis_labels = settings_dict.pop("axis_labels")
    offset = settings_dict.pop("offset")
    gain = settings_dict.pop("gain")
    use_robust_1d_fwhm = settings_dict.pop("use_robust_1d_fwhm")
    fwhm_plot_type = settings_dict.pop("fwhm_plot_type")

    click.echo("Detecting peaks...")
    t1 = time.time()
    peaks = detect_peaks(
        zyx_data,
        **settings_dict,
        verbose=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    t2 = time.time()
    click.echo(f'Time to detect peaks: {t2-t1}')

    t1 = time.time()
    beads, peak_coordinates = extract_beads(
        zyx_data=zyx_data,
        points=peaks,
        scale=zyx_scale,
        patch_size=patch_size,
    )

    if len(beads) == 0:
        raise RuntimeError("No beads were detected.")

    patch_size_pix = np.ceil(np.array(patch_size) / np.array(zyx_scale)).astype(int)
    noise = compute_noise_level(
        zyx_data,
        peak_coordinates,
        patch_size_pix,
    )

    click.echo("Analyzing PSFs...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_gaussian_fit, df_1d_peak_width = analyze_psf(
            zyx_patches=beads,
            peak_coordinates=peak_coordinates,
            scale=zyx_scale,
            offset=offset,
            gain=gain,
            noise=noise,
            use_robust_1d_fwhm=use_robust_1d_fwhm,
        )
    t2 = time.time()
    click.echo(f'Time to analyze PSFs: {t2-t1}')

    # Generate HTML report
    generate_report(
        output_report_path,
        input_dataset_path,
        input_dataset_name,
        beads,
        peaks,
        df_gaussian_fit,
        df_1d_peak_width,
        zyx_scale,
        axis_labels,
        fwhm_plot_type,
    )

    return peaks


@click.command("characterize-psf")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def characterize_psf_cli(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
):
    """
    Characterize the point spread function (PSF) from bead images and output an html report

    >> biahub characterize-psf -i ./beads.zarr/*/*/* -c ./characterize_params.yml -o ./
    """
    if len(input_position_dirpaths) > 1:
        warnings.warn("Only the first position will be characterized.")

    # Read settings
    settings = yaml_to_model(config_filepath, CharacterizeSettings)
    dataset_name = Path(input_position_dirpaths[0]).parts[-4]

    click.echo("Loading data...")
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        zyx_data = input_dataset["0"][0, 0]
        zyx_scale = input_dataset.scale[-3:]

    _ = _characterize_psf(
        zyx_data, zyx_scale, settings, output_dirpath, input_position_dirpaths[0], dataset_name
    )


if __name__ == "__main__":
    characterize_psf_cli()
