from pathlib import Path

import click
import dask.array as da
import numpy as np

from iohub import open_ome_zarr

from biahub.cli.parsing import config_filepath, output_filepath
from biahub.cli.utils import model_to_yaml, yaml_to_model
from biahub.register import find_lir
from biahub.settings import ConcatenateSettings


def estimate_crop(
    phase_mask: np.ndarray, fluor_mask: np.ndarray, phase_mask_radius: float = None
):
    """
    Estimate a crop region where both phase and fluorescene volumes are non-zero.

    Parameters
    ----------
    phase_data : ndarray
        TCZYX boolean mask of phase data.
    fluor_data : ndarray
        TCZYX boolean mask of fluorescence data.
    phase_mask_radius : float
        Radius of the circular mask which will be applied to the phase channel. If None, no masking will be applied

    """
    if phase_mask.ndim != 5 or fluor_mask.ndim != 5:
        raise ValueError("Both phase_data and fluor_data must be 5D arrays.")

    # Ensure data dimensions are the same
    _max_zyx_dims = np.asarray([phase_mask.shape[-3:], fluor_mask.shape[-3:]]).min(axis=0)

    # Concatenate the data arrays along the channel axis
    data = np.concatenate(
        [
            phase_mask[..., : _max_zyx_dims[0], : _max_zyx_dims[1], : _max_zyx_dims[2]],
            fluor_mask[..., : _max_zyx_dims[0], : _max_zyx_dims[1], : _max_zyx_dims[2]],
        ],
        axis=1,
    )

    # Create a mask to find time points and channels where any data is non-zero
    valid_mask = np.any(data, axis=(2, 3, 4))
    valid_T, valid_C = np.where(valid_mask)

    if len(valid_T) == 0:
        raise ValueError("No valid data found.")
    valid_data = data[valid_T, valid_C]

    # Compute a mask where all voxels are non-zero along time time and channel dimensions
    combined_mask = np.all(valid_data, axis=0)

    # Create a circular boolean mask of radius phase_mask_radius to apply to the phase channel
    if phase_mask_radius is not None:
        phase_mask = np.zeros(phase_mask.shape[-2:], dtype=bool)
        y, x = np.ogrid[: phase_mask.shape[-2], : phase_mask.shape[-1]]
        center = (phase_mask.shape[-2] // 2, phase_mask.shape[-1] // 2)
        radius = int(phase_mask_radius * min(center))
        phase_mask[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2] = True

        phase_mask_cropped = phase_mask[: _max_zyx_dims[1], : _max_zyx_dims[2]]
        combined_mask = combined_mask * phase_mask_cropped

    # Compute overlapping region
    z_slice, y_slice, x_slice = find_lir(combined_mask)

    return (
        [z_slice.start, z_slice.stop],
        [y_slice.start, y_slice.stop],
        [x_slice.start, x_slice.stop],
    )


@click.command("estimate-crop")
@config_filepath()
@output_filepath()
@click.option(
    "--phase-mask-radius",
    type=float,
    help="(Optional) Radius of the circular mask given as fraction of image width to apply to the phase channel.",
    required=False,
)
def estimate_crop_cli(
    config_filepath,
    output_filepath,
    phase_mask_radius,
):
    """
    Estimate a crop region where both phase and fluorescene volumes are non-zero.

    Parameters:
    ----------
    config_filepath : str
        Path to a yaml ConcatenateSettings file.
        This file will be replicated in the output with modified XYZ slicing parametrs.
    output_filepath : str
        Path to save the output config file.
    phase_mask_radius : float
        Radius of the circular mask given as fraction of image width to apply to the phase channel.
        A good value if 0.95.
    """
    config_filepath = Path(config_filepath)
    input_model = yaml_to_model(config_filepath, ConcatenateSettings)

    # Assume phase dataset is first and fluor dataset is second in input_model.concat_data_paths
    _target_paths = config_filepath.parent.glob(input_model.concat_data_paths[0])
    target_position_dirpaths = [p for p in _target_paths if p.is_dir()]
    _source_paths = config_filepath.parent.glob(input_model.concat_data_paths[1])
    source_position_dirpaths = [p for p in _source_paths if p.is_dir()]

    all_ranges = []
    for source_dir, target_dir in zip(source_position_dirpaths, target_position_dirpaths):
        click.echo(f"Processing {source_dir} and {target_dir}")

        # Load data
        with open_ome_zarr(target_dir) as target:
            phase_data = target.data.dask_array()[:, :1]  # Pick only first channel
            phase_mask = (phase_data != 0) & (~da.isnan(phase_data))

        with open_ome_zarr(source_dir) as source:
            fluor_data = source.data.dask_array()[:, :1]
            fluor_mask = (fluor_data != 0) & (~da.isnan(fluor_data))

        # Estimate crop region
        all_ranges.append(
            estimate_crop(phase_mask.compute(), fluor_mask.compute(), phase_mask_radius)
        )

    # Standardize ROI across all positions
    # TODO: we should be able to use a unique ROI per FOV
    all_ranges = np.array(all_ranges)
    standardized_ranges = np.concatenate(
        [
            all_ranges[..., 0].max(axis=0, keepdims=True),
            all_ranges[..., 1].min(axis=0, keepdims=True),
        ]
    )

    output_model = input_model.model_copy()
    output_model.Z_slice = standardized_ranges[:, 0].tolist()
    output_model.Y_slice = standardized_ranges[:, 1].tolist()
    output_model.X_slice = standardized_ranges[:, 2].tolist()

    model_to_yaml(output_model, output_filepath)


if __name__ == "__main__":
    estimate_crop_cli()
