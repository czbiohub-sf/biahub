import click
import numpy as np

from iohub import open_ome_zarr

from biahub.cli.parsing import (
    output_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.cli.utils import model_to_yaml
from biahub.register import find_lir
from biahub.settings import ConcatenateSettings


def estimate_crop(
    phase_data: np.ndarray, fluor_data: np.ndarray, phase_mask_radius: float = None
):
    """
    Estimate a crop region where both phase and fluorescene volumes are non-zero.

    Parameters
    ----------
    phase_data : ndarray
        TCZYX phase data array.
    fluor_data : ndarray
        TCZYX fluorescence data array.
    phase_mask_radius : float
        Radius of the circular mask which will be applied to the phase channel. If None, no masking will be applied

    """
    if phase_data.ndim != 5 or fluor_data.ndim != 5:
        raise ValueError("Both phase_data and fluor_data must be 5D arrays.")

    # Ensure data dimensions are the same
    _max_zyx_dims = np.asarray([phase_data.shape[-3:], fluor_data.shape[-3:]]).min(axis=0)

    # Concatenate the data arrays along the channel axis
    data = np.concatenate(
        [
            phase_data[..., : _max_zyx_dims[0], : _max_zyx_dims[1], : _max_zyx_dims[2]],
            fluor_data[..., : _max_zyx_dims[0], : _max_zyx_dims[1], : _max_zyx_dims[2]],
        ],
        axis=1,
    )

    # Create a mask to find time points and channels where any data is non-zero
    valid_mask = np.any((data != 0) & (~np.isnan(data)), axis=(2, 3, 4))
    valid_T, valid_C = np.where(valid_mask)

    if len(valid_T) == 0:
        raise ValueError("No valid data found.")
    valid_data = data[valid_T, valid_C]

    # Compute a mask where all voxels are non-zero along time time and channel dimensions
    combined_mask = np.all((valid_data != 0) & (~np.isnan(valid_data)), axis=0)

    # Create a circular boolean mask of radius phase_mask_radius to apply to the phase channel
    if phase_mask_radius is not None:
        phase_mask = np.zeros(phase_data.shape[-2:], dtype=bool)
        y, x = np.ogrid[: phase_data.shape[-2], : phase_data.shape[-1]]
        center = (phase_data.shape[-2] // 2, phase_data.shape[-1] // 2)
        radius = int(phase_mask_radius * min(center))
        phase_mask[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2] = True

        phase_mask_cropped = phase_mask[: _max_zyx_dims[1], : _max_zyx_dims[2]]
        combined_mask = combined_mask * phase_mask_cropped

    # Compute overlapping region
    z_slice, y_slice, x_slice = find_lir(combined_mask)

    return (
        (z_slice.start, z_slice.stop),
        (y_slice.start, y_slice.stop),
        (x_slice.start, x_slice.stop),
    )


@click.command()
@source_position_dirpaths()
@target_position_dirpaths()
@output_filepath()
@click.option(
    "--phase-mask-radius",
    type=float,
    help="(Optional) Radius of the circular mask given as fraction of image width to apply to the phase channel.",
    required=False,
)
def esitmate_crop_cli(
    source_position_dirpaths,
    target_position_dirpaths,
    output_filepath,
    phase_mask_radius,
):
    """
    Estimate a crop region where both phase and fluorescene volumes are non-zero.

    Parameters:
    ----------
    source_position_dirpaths : list
        Provide one position of the source zarr store during registration,
        for example flour.zarr/A/1/000000
    target_position_dirpaths : list
        Provide one position of the target zarr store during registration,
        for example flour.zarr/A/1/000000. If a phase mask is to be applied,
        we assume that the phase channel is the target channel.
        Switching source and target channels during crop estimation will not
        affect the output of the alignment.
    output_filepath : str
        Path to save the output config file.
    phase_mask_radius : float
        Radius of the circular mask given as fraction of image width to apply to the phase channel.
        A good value if 0.95.
    """
    # Load data
    with open_ome_zarr(source_position_dirpaths[0]) as source:
        fluor_data = source.data.dask_array()
        source_channels = source.data.channel_names

    with open_ome_zarr(target_position_dirpaths[0]) as target:
        phase_data = target.data.dask_array()
        target_channels = target.data.channel_names

    # Estimate crop region
    z_range, y_range, x_range = estimate_crop(phase_data, fluor_data, phase_mask_radius)

    # Save results
    model = ConcatenateSettings(
        concat_data_paths=source_position_dirpaths + target_position_dirpaths,
        time_indices='all',
        channel_names=[source_channels, target_channels],
        Z_slice=[z_range[0], z_range[1]],
        Y_slice=[y_range[0], y_range[1]],
        X_slice=[x_range[0], x_range[1]],
    )

    model_to_yaml(model, output_filepath)
