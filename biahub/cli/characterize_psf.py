import gc
import time
import warnings

from pathlib import Path
from typing import List

import click
import numpy as np
import torch

from iohub.ngff import open_ome_zarr

from biahub.analysis.AnalysisSettings import CharacterizeSettings
from biahub.analysis.analyze_psf import (
    analyze_psf,
    detect_peaks,
    extract_beads,
    generate_report,
    compute_snr,
)
from biahub.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath
from biahub.cli.utils import yaml_to_model


def _characterize_psf(
    zyx_data: np.ndarray,
    zyx_scale: tuple[float, float, float],
    settings: CharacterizeSettings,
    output_report_path: str,
    input_dataset_path: str,
    input_dataset_name: str,
):
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

    click.echo("Analyzing PSFs...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_gaussian_fit, df_1d_peak_width = analyze_psf(
            zyx_patches=beads,
            peak_coordinates=peak_coordinates,
            scale=zyx_scale,
            offset=offset,
            gain=gain,
            use_robust_1d_fwhm=use_robust_1d_fwhm,
        )
    t2 = time.time()
    click.echo(f'Time to analyze PSFs: {t2-t1}')

    patch_size_pix = np.ceil(np.array(patch_size) / np.array(zyx_scale)).astype(int)
    snr = compute_snr(
        zyx_data,
        peak_coordinates,
        patch_size_pix,
        df_gaussian_fit['z_mu'].mean(),
    )

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


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def characterize_psf(
    input_position_dirpaths: List[str],
    config_filepath: str,
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
    characterize_psf()
