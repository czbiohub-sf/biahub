import datetime
import importlib.resources as pkg_resources
import pickle
import shutil
import webbrowser

from pathlib import Path
from typing import List, Literal

import markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from napari_psf_analysis.psf_analysis.extract.BeadExtractor import BeadExtractor
from napari_psf_analysis.psf_analysis.image import Calibrated3DImage
from napari_psf_analysis.psf_analysis.psf import PSF
from numpy.typing import ArrayLike
from scipy.signal import peak_widths
from scipy.interpolate import interp1d

import biahub.analysis.templates


def _make_plots(
    output_path: Path,
    beads: List[ArrayLike],
    df_gaussian_fit: pd.DataFrame,
    df_1d_peak_width: pd.DataFrame,
    scale: tuple,
    axis_labels: tuple,
    fwhm_plot_type: Literal['1D', '3D'],
):
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
    beads: List[ArrayLike],
    peaks: ArrayLike,
    df_gaussian_fit: pd.DataFrame,
    df_1d_peak_width: pd.DataFrame,
    scale: tuple,
    axis_labels: tuple,
    fwhm_plot_type: str,
):
    output_path.mkdir(exist_ok=True)

    num_beads = len(beads)
    num_successful = len(df_gaussian_fit)
    num_failed = num_beads - num_successful

    # make plots
    (bead_psf_slices_path, fwhm_vs_acq_axes_paths, psf_amp_paths) = _make_plots(
        output_path, beads, df_gaussian_fit, df_1d_peak_width, scale, axis_labels, fwhm_plot_type
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

    with pkg_resources.path(biahub.analysis.templates, 'github-markdown.css') as css_path:
        shutil.copy(css_path, output_path)
    html_file_path = output_path / ('psf_analysis_report.html')
    with open(html_file_path, 'w') as file:
        file.write(html_report)

    # display html report
    html_file_path = Path(html_file_path).absolute()
    webbrowser.open(html_file_path.as_uri())


def extract_beads(
    zyx_data: ArrayLike, points: ArrayLike, scale: tuple, patch_size: tuple = None
):
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
    zyx_patches: List[ArrayLike],
    peak_coordinates: List[tuple],
    scale: tuple,
    offset: float = 0.0,
    gain: float = 1.0,
    noise: float = 1.0,
    use_robust_1d_fwhm: bool = False,
):
    """
    Analyze point spread function (PSF) from given 3D patches.

    Parameters:
    -----------
    zyx_patches : List[ArrayLike]
        List of 3D image patches to be analyzed.
    peak_coordinates : List[tuple]
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

    Returns:
    --------
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
    zyx_data: ArrayLike, peak_coordinates: List[tuple], patch_size_pix: tuple
):
    # Mask out the peaks
    mask = np.ones_like(zyx_data, dtype=bool)
    half_patch = [size // 2 for size in patch_size_pix]

    for z, y, x in peak_coordinates:
        patch_mask = tuple(
            slice(max(0, coord - half_patch[i]), min(zyx_data.shape[i], coord + half_patch[i] + 1))
            for i, coord in enumerate((z, y, x))
        )
        mask[patch_mask] = False

    return np.std(zyx_data[mask])


def calculate_robust_peak_widths(zyx_data: ArrayLike, zyx_scale: tuple):
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
            indices = np.where(y >= half_max/2)[0]
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


def calculate_peak_widths(zyx_data: ArrayLike, zyx_scale: tuple):
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
    plots_dir: str,
    beads: List[ArrayLike],
    zyx_scale: tuple,
    axis_labels: tuple,
    bead_numbers: list,
):
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


def plot_fwhm_vs_acq_axes(plots_dir: str, x, y, z, fwhm_x, fwhm_y, fwhm_z, axis_labels: tuple):
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


def plot_psf_amp(plots_dir: str, x, y, z, amp, axis_labels: tuple):
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
    dataset_scale: tuple,
    num_beads_total_good_bad: tuple,
    snr_mean: float,
    snr_std: float,
    fwhm_1d_mean: tuple,
    fwhm_1d_std: tuple,
    fwhm_3d_mean: tuple,
    fwhm_3d_std: tuple,
    fwhm_pc_mean: tuple,
    bead_psf_slices_path: str,
    fwhm_vs_acq_axes_paths: list,
    psf_amp_paths: list,
    axis_labels: tuple,
    fwhm_plot_type: str,
):

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
