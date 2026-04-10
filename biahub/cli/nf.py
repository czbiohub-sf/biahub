import logging

from pathlib import Path

import click
import numpy as np

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.cli.utils import (
    copy_position_metadata,
    estimate_resources,
    read_plate_metadata,
    yaml_to_model,
)

logger = logging.getLogger(__name__)


@click.group("nf")
def nf_cli():
    """Nextflow-oriented commands for single-unit-of-work processing."""


@nf_cli.command("list-positions")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
def list_positions(input_zarr: str):
    """List position keys in a plate zarr (one per line, for Nextflow fan-out)."""
    with open_ome_zarr(input_zarr, mode="r") as plate:
        for name, _ in plate.positions():
            click.echo(name)


# ---------------------------------------------------------------------------
# Flat-field
# ---------------------------------------------------------------------------


@nf_cli.command("init-flat-field")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def init_flat_field(input_zarr: str, output_zarr: str, config: str):
    """Create empty output zarr for flat-field correction."""
    from biahub.settings import FlatFieldCorrectionSettings

    settings = yaml_to_model(Path(config), FlatFieldCorrectionSettings)
    position_keys, channel_names, shape, scale = read_plate_metadata(input_zarr)

    if settings.channel_names:
        for ch in settings.channel_names:
            if ch not in channel_names:
                raise click.ClickException(
                    f"Channel '{ch}' not found. Available: {channel_names}"
                )

    create_empty_plate(
        store_path=Path(output_zarr),
        position_keys=position_keys,
        channel_names=channel_names,
        shape=shape,
        chunks=None,
        scale=scale,
        version="0.5",
        dtype=np.float32,
    )
    copy_position_metadata(Path(input_zarr), Path(output_zarr))
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")

    num_cpus, mem_per_cpu = estimate_resources(shape=shape, ram_multiplier=5)
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("run-flat-field")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--num-threads", "-j", default=1, type=int)
def run_flat_field(
    input_zarr: str, output_zarr: str, position: str, config: str, num_threads: int
):
    """Apply flat-field correction to a single position (all timepoints)."""
    from biahub.flat_field_correction import czyx_flat_field_correction
    from biahub.settings import FlatFieldCorrectionSettings

    input_position = Path(input_zarr) / position
    output_position = Path(output_zarr) / position
    settings = yaml_to_model(Path(config), FlatFieldCorrectionSettings)

    with open_ome_zarr(str(input_position), mode="r") as ds:
        all_channel_names = ds.channel_names

    channel_names = settings.channel_names if settings.channel_names else all_channel_names

    process_single_position(
        czyx_flat_field_correction,
        input_position,
        output_position,
        num_threads=num_threads,
        channel_names=channel_names,
        all_channel_names=all_channel_names,
    )

    click.echo(f"Flat-field done: {position}")


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------


@nf_cli.command("init-deskew")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def init_deskew(input_zarr: str, output_zarr: str, config: str):
    """Create empty output zarr for deskew."""
    from biahub.deskew import get_deskewed_data_shape
    from biahub.settings import DeskewSettings

    settings = yaml_to_model(Path(config), DeskewSettings)
    position_keys, channel_names, shape, _ = read_plate_metadata(input_zarr)
    T, C, Z, Y, X = shape

    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X),
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        settings.keep_overhang,
        settings.average_n_slices,
        settings.pixel_size_um,
    )

    create_empty_plate(
        store_path=Path(output_zarr),
        position_keys=position_keys,
        channel_names=channel_names,
        shape=(T, C) + deskewed_shape,
        scale=(1, 1) + voxel_size,
    )
    copy_position_metadata(Path(input_zarr), Path(output_zarr))
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")

    num_cpus, mem_per_cpu = estimate_resources(shape=shape, ram_multiplier=16, max_num_cpus=16)
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("run-deskew")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def run_deskew(input_zarr: str, output_zarr: str, position: str, config: str):
    """Deskew a single position (all timepoints and channels)."""
    from biahub.deskew import _czyx_deskew_data
    from biahub.settings import DeskewSettings

    input_position = Path(input_zarr) / position
    output_position = Path(output_zarr) / position
    settings = yaml_to_model(Path(config), DeskewSettings)

    process_single_position(
        _czyx_deskew_data,
        input_position,
        output_position,
        num_threads=1,
        ls_angle_deg=settings.ls_angle_deg,
        px_to_scan_ratio=settings.px_to_scan_ratio,
        keep_overhang=settings.keep_overhang,
        average_n_slices=settings.average_n_slices,
        overhang_fill=settings.overhang_fill,
        extra_metadata={"deskew": settings.model_dump()},
    )

    click.echo(f"Deskew done: {position}")


# ---------------------------------------------------------------------------
# Reconstruct
# ---------------------------------------------------------------------------


def _upsampled_zyx(settings, zyx_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return the Nyquist-upsampled ZYX shape used during TF computation."""
    import math

    from waveorder import sampling

    Z, Y, X = zyx_shape
    if settings.phase is not None:
        tf = settings.phase.transfer_function
        trans_nyq = sampling.transverse_nyquist(
            tf.wavelength_illumination,
            tf.numerical_aperture_illumination,
            tf.numerical_aperture_detection,
        )
        axial_nyq = sampling.axial_nyquist(
            tf.wavelength_illumination,
            tf.numerical_aperture_detection,
            tf.index_of_refraction_media,
        )
        yx_factor = math.ceil(tf.yx_pixel_size / trans_nyq)
        z_factor = math.ceil(tf.z_pixel_size / axial_nyq)
        Z, Y, X = Z * z_factor, Y * yx_factor, X * yx_factor
    return Z, Y, X


@nf_cli.command("init-reconstruct")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--num-threads", "-j", default=1, type=int)
def init_reconstruct(input_zarr: str, output_zarr: str, config: str, num_threads: int):
    """Create empty output zarr and estimate resources for reconstruction."""
    from waveorder.cli.apply_inverse_transfer_function import (
        get_reconstruction_output_metadata,
    )
    from waveorder.cli.settings import ReconstructionSettings
    from waveorder.cli.utils import create_empty_hcs_zarr
    from waveorder.cli.utils import estimate_resources as wo_estimate_resources

    config_path = Path(config)
    position_keys, _, shape, _ = read_plate_metadata(input_zarr)
    T, C, Z, Y, X = shape
    first_position_path = Path(input_zarr) / "/".join(position_keys[0])

    output_metadata = get_reconstruction_output_metadata(first_position_path, config_path)

    create_empty_hcs_zarr(
        store_path=Path(output_zarr),
        position_keys=position_keys,
        **output_metadata,
    )
    copy_position_metadata(Path(input_zarr), Path(output_zarr))
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")

    settings = yaml_to_model(config_path, ReconstructionSettings)
    num_cpus, mem_per_cpu = wo_estimate_resources(list(shape), settings, num_threads)
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")

    uZ, uY, uX = _upsampled_zyx(settings, (Z, Y, X))
    tf_cpus, tf_mem = wo_estimate_resources([1, 1, uZ, uY, uX], settings, 1)
    click.echo(f"TF_RESOURCES:{tf_cpus} {tf_cpus * tf_mem}")


@nf_cli.command("compute-transfer-function")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--tf-path", "-t", required=True, type=click.Path())
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def compute_transfer_function(input_zarr: str, tf_path: str, config: str):
    """Compute transfer function from the first position in the input zarr."""
    from waveorder.cli.compute_transfer_function import (
        compute_transfer_function_cli,
    )

    config_path = Path(config)
    tf_zarr = Path(tf_path)

    position_keys, _, _, _ = read_plate_metadata(input_zarr)
    if not position_keys:
        raise click.ClickException(f"Input plate '{input_zarr}' contains no positions.")
    first_position_path = Path(input_zarr) / "/".join(position_keys[0])

    click.echo(f"Computing transfer function from {first_position_path}")
    compute_transfer_function_cli(first_position_path, config_path, tf_zarr)
    click.echo(f"Transfer function saved to {tf_zarr}")


@nf_cli.command("run-apply-inv-tf")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--tf-path", "-t", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--num-threads", "-j", default=1, type=int)
def run_apply_inv_tf(
    input_zarr: str,
    output_zarr: str,
    tf_path: str,
    position: str,
    config: str,
    num_threads: int,
):
    """Apply inverse transfer function to a single position."""
    from waveorder.cli.apply_inverse_transfer_function import (
        apply_inverse_transfer_function_single_position,
        get_reconstruction_output_metadata,
    )

    input_position = Path(input_zarr) / position
    output_position = Path(output_zarr) / position
    config_path = Path(config)

    output_metadata = get_reconstruction_output_metadata(input_position, config_path)

    apply_inverse_transfer_function_single_position(
        input_position,
        Path(tf_path),
        config_path,
        output_position,
        num_threads,
        output_metadata["channel_names"],
    )
    click.echo(f"Reconstruction done: {position}")


# ---------------------------------------------------------------------------
# Virtual stain
# ---------------------------------------------------------------------------


@nf_cli.command("init-virtual-stain")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def init_virtual_stain(input_zarr: str, output_zarr: str, config: str):
    """Create empty output zarr for virtual staining predictions."""
    import yaml

    with open(config) as f:
        cfg = yaml.safe_load(f)

    target_channels = cfg["data"]["init_args"]["target_channel"]
    prediction_channels = [f"{ch}_prediction" for ch in target_channels]

    position_keys, _, shape, scale = read_plate_metadata(input_zarr)
    T, _, Z, Y, X = shape

    create_empty_plate(
        store_path=Path(output_zarr),
        position_keys=position_keys,
        channel_names=prediction_channels,
        shape=(T, len(prediction_channels), Z, Y, X),
        scale=scale,
        version="0.5",
        dtype=np.float32,
    )
    copy_position_metadata(Path(input_zarr), Path(output_zarr))
    click.echo(
        f"Created {output_zarr} ({len(position_keys)} positions, "
        f"channels={prediction_channels})"
    )

    num_cpus = 4
    mem_per_cpu = 8
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("copy-virtual-stain")
@click.option("--temp-zarr", "-t", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
def copy_virtual_stain(temp_zarr: str, output_zarr: str, position: str):
    """Copy viscy prediction from temp FOV zarr into output plate position."""
    import shutil

    temp_position = Path(temp_zarr) / position
    output_position = Path(output_zarr) / position

    with open_ome_zarr(str(temp_position), mode="r") as src:
        src_data = np.asarray(src[0][:])
        src_attrs = dict(src.zattrs)

    with open_ome_zarr(str(output_position), mode="r+") as dst:
        dst[0][:] = src_data
        dst.zattrs.update(src_attrs)

    shutil.rmtree(temp_zarr)
    click.echo(f"Virtual stain copied: {position}")
