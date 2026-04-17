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
        version="0.5",
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


# ---------------------------------------------------------------------------
# Channel rename
# ---------------------------------------------------------------------------


@nf_cli.command("rename-channels")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--prefix", default="", help="Prefix to prepend to each channel name")
@click.option("--suffix", default="", help="Suffix to append to each channel name")
def rename_channels(input_zarr: str, position: str, prefix: str, suffix: str):
    """Rename channels for a single position (metadata-only, no data copy)."""
    if not prefix and not suffix:
        raise click.ClickException("Provide at least --prefix or --suffix.")

    position_path = Path(input_zarr) / position
    with open_ome_zarr(str(position_path), mode="r+") as pos:
        for old_name in list(pos.channel_names):
            new_name = f"{prefix}{old_name}{suffix}"
            pos.rename_channel(old_name, new_name)

    click.echo(f"Renamed channels: {position}")
    click.echo("RESOURCES:1 2")


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


@nf_cli.command("init-track")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def init_track(input_zarr: str, output_zarr: str, config: str):
    """Create empty output zarr for tracking (uint32 labels, single channel)."""
    from biahub.settings import TrackingSettings
    from biahub.track import resolve_z_slice

    settings = yaml_to_model(Path(config), TrackingSettings)
    position_keys, _, shape, scale = read_plate_metadata(input_zarr)
    T, C, Z, Y, X = shape

    _, Z_out = resolve_z_slice(settings.z_range, Z)

    if settings.mode == "2D":
        output_shape = (T, 1, 1, Y, X)
    else:
        output_shape = (T, 1, Z_out, Y, X)

    create_empty_plate(
        store_path=Path(output_zarr),
        position_keys=position_keys,
        channel_names=[f"{settings.target_channel}_labels"],
        shape=output_shape,
        chunks=None,
        scale=scale,
        version="0.5",
        dtype=np.uint32,
    )
    copy_position_metadata(Path(input_zarr), Path(output_zarr))
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")

    num_cpus, mem_per_cpu = estimate_resources(
        shape=[T, C, Z_out, Y, X], ram_multiplier=16, max_num_cpus=16
    )
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("run-track")
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--blank-frames-csv", default=None, type=click.Path(exists=True))
def run_track(output_zarr: str, position: str, config: str, blank_frames_csv: str | None):
    """Run tracking for a single position."""
    from biahub.settings import TrackingSettings
    from biahub.track import resolve_z_slice, track_one_position

    settings = yaml_to_model(Path(config), TrackingSettings)
    output_dirpath = Path(output_zarr)
    position_key = tuple(position.split("/"))

    input_images_paths = [
        image.path for image in settings.input_images if image.path is not None
    ]
    if not input_images_paths:
        raise click.ClickException("No input_images_paths provided in config.")

    with open_ome_zarr(str(input_images_paths[0] / Path(*position_key)), mode="r") as ds:
        T, C, Z, Y, X = ds.data.shape
        scale = ds.scale

    z_slices, _ = resolve_z_slice(settings.z_range, Z)
    track_scale = scale[-2:] if settings.mode == "2D" else scale[-3:]

    blank_path = Path(blank_frames_csv) if blank_frames_csv else settings.blank_frames_path

    track_one_position(
        position_key=position_key,
        input_images=settings.input_images,
        output_dirpath=output_dirpath,
        tracking_config=settings.tracking_config,
        blank_frames_path=blank_path,
        z_slices=z_slices,
        scale=track_scale,
    )

    click.echo(f"Tracking done: {position}")


# ---------------------------------------------------------------------------
# Assembly (estimate-crop / init-concatenate / run-concatenate)
# ---------------------------------------------------------------------------


@nf_cli.command("estimate-crop")
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-config", "-o", required=True, type=click.Path())
@click.option(
    "--lf-mask-radius",
    type=float,
    default=0.95,
    help="Circular mask radius as fraction of image width for phase channel.",
)
def nf_estimate_crop(config: str, output_config: str, lf_mask_radius: float):
    """Estimate crop region across all positions and write updated config."""
    import glob as globmod

    from natsort import natsorted

    from biahub.cli.utils import model_to_yaml
    from biahub.estimate_crop import estimate_crop_one_position
    from biahub.settings import ConcatenateSettings

    config_path = Path(config)
    settings = yaml_to_model(config_path, ConcatenateSettings)

    if len(settings.concat_data_paths) < 2:
        raise click.ClickException(
            "estimate-crop requires at least 2 concat_data_paths (label-free + light-sheet)."
        )

    lf_positions = natsorted(
        [Path(p) for p in globmod.glob(settings.concat_data_paths[0]) if Path(p).is_dir()]
    )
    ls_positions = natsorted(
        [Path(p) for p in globmod.glob(settings.concat_data_paths[1]) if Path(p).is_dir()]
    )

    if len(lf_positions) != len(ls_positions):
        raise click.ClickException(
            f"Mismatched position counts: {len(lf_positions)} label-free vs {len(ls_positions)} light-sheet."
        )

    all_ranges = []
    for lf_dir, ls_dir in zip(lf_positions, ls_positions, strict=True):
        z_range, y_range, x_range = estimate_crop_one_position(
            lf_dir=lf_dir,
            ls_dir=ls_dir,
            lf_mask_radius=lf_mask_radius,
        )
        all_ranges.append([z_range, y_range, x_range])

    all_ranges = np.array(all_ranges)
    standardized_ranges = np.concatenate(
        [
            all_ranges[..., 0].max(axis=0, keepdims=True),
            all_ranges[..., 1].min(axis=0, keepdims=True),
        ]
    )

    output_model = settings.model_copy()
    output_model.Z_slice = standardized_ranges[:, 0].tolist()
    output_model.Y_slice = standardized_ranges[:, 1].tolist()
    output_model.X_slice = standardized_ranges[:, 2].tolist()
    model_to_yaml(output_model, Path(output_config))

    click.echo(f"Updated config written to {output_config}")
    click.echo("RESOURCES:4 32")


@nf_cli.command("init-concatenate")
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
def init_concatenate(config: str, output_zarr: str):
    """Create empty output zarr for concatenation."""
    from iohub.ngff.utils import create_empty_plate

    from biahub.cli.utils import get_output_paths
    from biahub.concatenate import (
        calculate_cropped_size,
        get_channel_combiner_metadata,
    )
    from biahub.settings import ConcatenateSettings

    settings = yaml_to_model(Path(config), ConcatenateSettings)
    slicing_params = [settings.Z_slice, settings.Y_slice, settings.X_slice]

    (
        all_data_paths,
        all_channel_names,
        _input_channel_idx,
        _output_channel_idx,
        all_slicing_params,
    ) = get_channel_combiner_metadata(
        settings.concat_data_paths, settings.channel_names, slicing_params
    )

    output_position_paths = get_output_paths(
        all_data_paths,
        Path(output_zarr),
        ensure_unique_positions=settings.ensure_unique_positions,
    )

    all_shapes = []
    all_dtypes = []
    all_voxel_sizes = []
    for path in all_data_paths:
        with open_ome_zarr(path) as ds:
            all_shapes.append(ds.data.shape)
            all_dtypes.append(ds.data.dtype)
            all_voxel_sizes.append(ds.scale[-3:])

    T = all_shapes[0][0]
    if settings.time_indices == "all":
        T = min(s[0] for s in all_shapes)
        input_time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        input_time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        input_time_indices = [settings.time_indices]

    if all(d == all_dtypes[0] for d in all_dtypes):
        dtype = all_dtypes[0]
    else:
        dtype = np.float32

    cropped_shape_zyx = calculate_cropped_size(all_slicing_params[0])
    output_voxel_size = all_voxel_sizes[0]

    output_shape = (len(input_time_indices), len(all_channel_names)) + tuple(cropped_shape_zyx)
    output_scale = (1,) * 2 + tuple(output_voxel_size)

    create_empty_plate(
        store_path=Path(output_zarr),
        position_keys=[p.parts[-3:] for p in output_position_paths],
        channel_names=all_channel_names,
        shape=output_shape,
        chunks=settings.chunks_czyx if settings.chunks_czyx else None,
        scale=output_scale,
        version=settings.output_ome_zarr_version,
        dtype=dtype,
    )

    click.echo(f"Created {output_zarr} ({len(output_position_paths)} positions)")

    C = all_shapes[0][1]
    Z, Y, X = all_shapes[0][2:]
    batch_size = settings.shards_ratio[0] if settings.shards_ratio else 1
    num_cpus, mem_per_cpu = estimate_resources(
        shape=(T // batch_size, C, Z, Y, X), ram_multiplier=4 * batch_size, max_num_cpus=16
    )
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("run-concatenate")
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--num-threads", "-j", default=1, type=int)
def run_concatenate(config: str, output_zarr: str, position: str, num_threads: int):
    """Copy and crop data from all input stores for one position into output."""
    from iohub.ngff.utils import process_single_position as iohub_process_single_position

    from biahub.cli.utils import copy_n_paste
    from biahub.concatenate import get_channel_combiner_metadata
    from biahub.settings import ConcatenateSettings

    settings = yaml_to_model(Path(config), ConcatenateSettings)
    slicing_params = [settings.Z_slice, settings.Y_slice, settings.X_slice]

    (
        all_data_paths,
        _all_channel_names,
        input_channel_idx_list,
        output_channel_idx_list,
        all_slicing_params,
    ) = get_channel_combiner_metadata(
        settings.concat_data_paths, settings.channel_names, slicing_params
    )

    all_shapes = []
    for path in all_data_paths:
        with open_ome_zarr(path) as ds:
            all_shapes.append(ds.data.shape)

    T = all_shapes[0][0]
    if settings.time_indices == "all":
        T = min(s[0] for s in all_shapes)
        input_time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        input_time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        input_time_indices = [settings.time_indices]

    output_dirpath = Path(output_zarr)

    for (
        input_path,
        input_ch_idx,
        output_ch_idx,
        zyx_slicing_params,
    ) in zip(
        all_data_paths,
        input_channel_idx_list,
        output_channel_idx_list,
        all_slicing_params,
        strict=True,
    ):
        fov_key = "/".join(Path(input_path).parts[-3:])
        if fov_key != position:
            continue

        output_position_path = output_dirpath / position

        iohub_process_single_position(
            copy_n_paste,
            input_position_path=input_path,
            output_position_path=output_position_path,
            input_channel_indices=input_ch_idx,
            output_channel_indices=output_ch_idx,
            input_time_indices=input_time_indices,
            output_time_indices=list(range(len(input_time_indices))),
            num_processes=num_threads,
            zyx_slicing_params=zyx_slicing_params,
        )

    click.echo(f"Concatenation done: {position}")


# ---------------------------------------------------------------------------
# Stabilization: combine-transforms / init-stabilize / run-stabilize
# ---------------------------------------------------------------------------


@nf_cli.command("combine-transforms")
@click.option("--config-a", "-a", required=True, type=click.Path(exists=True))
@click.option("--config-b", "-b", required=True, type=click.Path(exists=True))
@click.option("--output-config", "-o", required=True, type=click.Path())
def combine_transforms(config_a: str, config_b: str, output_config: str):
    """Compose two per-FOV transform lists: output[t] = A[t] @ B[t]."""
    from biahub.cli.utils import model_to_yaml
    from biahub.settings import StabilizationSettings

    settings_a = yaml_to_model(Path(config_a), StabilizationSettings)
    settings_b = yaml_to_model(Path(config_b), StabilizationSettings)

    transforms_a = np.array(settings_a.affine_transform_zyx_list)
    transforms_b = np.array(settings_b.affine_transform_zyx_list)

    if len(transforms_a) != len(transforms_b):
        raise click.ClickException(
            f"Transform count mismatch: {len(transforms_a)} vs {len(transforms_b)}"
        )

    composed = np.array(
        [a @ b for a, b in zip(transforms_a, transforms_b, strict=True)]
    )

    output_model = settings_a.model_copy()
    output_model.affine_transform_zyx_list = composed.tolist()
    model_to_yaml(output_model, Path(output_config))

    click.echo(f"Combined transforms written to {output_config}")
    click.echo("RESOURCES:1 4")


@nf_cli.command("init-stabilize")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def init_stabilize(input_zarr: str, output_zarr: str, config: str):
    """Create empty output zarr for stabilization."""
    from scipy.linalg import svd
    from scipy.spatial.transform import Rotation

    from biahub.settings import StabilizationSettings

    settings = yaml_to_model(Path(config), StabilizationSettings)
    position_keys, channel_names, shape, _ = read_plate_metadata(input_zarr)
    T, C, Z, Y, X = shape

    combined_mats = np.array(settings.affine_transform_zyx_list)

    R_matrix = combined_mats[0][:3, :3]
    U, _, Vt = svd(R_matrix)
    R_pure = U @ Vt
    euler_angles = Rotation.from_matrix(R_pure).as_euler("xyz", degrees=True)

    if np.isclose(euler_angles[0], 90, atol=10):
        out_Y, out_X = X, Y
    else:
        out_Y, out_X = Y, X

    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    output_shape = (len(time_indices), len(channel_names), Z, out_Y, out_X)

    create_empty_plate(
        store_path=Path(output_zarr),
        position_keys=position_keys,
        channel_names=channel_names,
        shape=output_shape,
        chunks=None,
        scale=settings.output_voxel_size,
        version="0.5",
        dtype=np.float32,
    )
    copy_position_metadata(Path(input_zarr), Path(output_zarr))
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")

    num_cpus, mem_per_cpu = estimate_resources(
        shape=output_shape, ram_multiplier=16, max_num_cpus=16
    )
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("run-stabilize")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--num-threads", "-j", default=1, type=int)
def run_stabilize(
    input_zarr: str,
    output_zarr: str,
    position: str,
    config: str,
    num_threads: int,
):
    """Apply precomputed stabilization transforms to a single position."""
    from biahub.cli.utils import process_single_position_v2
    from biahub.settings import StabilizationSettings
    from biahub.stabilize import apply_stabilization_transform

    settings = yaml_to_model(Path(config), StabilizationSettings)
    combined_mats = np.array(settings.affine_transform_zyx_list)

    input_position = Path(input_zarr) / position
    output_position = Path(output_zarr) / position

    with open_ome_zarr(str(input_position), mode="r") as ds:
        T, C, Z, Y, X = ds.data.shape
        channel_names = ds.channel_names

    with open_ome_zarr(str(output_position), mode="r") as ds_out:
        _, _, out_Z, out_Y, out_X = ds_out.data.shape

    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    for ch_idx, ch_name in enumerate(channel_names):
        process_single_position_v2(
            func=apply_stabilization_transform,
            input_data_path=input_position,
            output_path=Path(output_zarr),
            time_indices=time_indices,
            input_channel_idx=[ch_idx],
            output_channel_idx=[ch_idx],
            num_threads=num_threads,
            list_of_shifts=combined_mats,
            output_shape=(out_Z, out_Y, out_X),
        )

    click.echo(f"Stabilization done: {position}")


# ---------------------------------------------------------------------------
# Stabilization estimation: z-focus / xy / pcc / beads
# ---------------------------------------------------------------------------


@nf_cli.command("estimate-stabilization-z-focus")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
def estimate_stabilization_z_focus(
    input_zarr: str, position: str, config: str, output_dir: str
):
    """Estimate Z-focus stabilization for a single position."""
    from biahub.estimate_stabilization import estimate_z_focus_per_position
    from biahub.settings import EstimateStabilizationSettings

    settings = yaml_to_model(Path(config), EstimateStabilizationSettings)
    input_position = Path(input_zarr) / position

    with open_ome_zarr(str(input_position), mode="r") as ds:
        channel_names = ds.channel_names
        shape = ds.data.shape

    channel_index = channel_names.index(settings.stabilization_estimation_channel)

    output_path = Path(output_dir)
    focus_csv_dir = output_path / "z_focus_positions"
    transform_dir = output_path / "z_transforms"

    estimate_z_focus_per_position(
        input_position_dirpath=input_position,
        input_channel_indices=(channel_index,),
        center_crop_xy=settings.focus_finding_settings.center_crop_xy,
        output_path_focus_csv=focus_csv_dir,
        output_path_transform=transform_dir,
        verbose=settings.verbose,
    )

    click.echo(f"Z-focus estimation done: {position}")
    num_cpus, mem_per_cpu = estimate_resources(
        shape=shape, ram_multiplier=8, max_num_cpus=16
    )
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("estimate-stabilization-xy")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--focus-csv", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
def estimate_stabilization_xy(
    input_zarr: str, position: str, config: str, focus_csv: str, output_dir: str
):
    """Estimate XY stabilization for a single position (requires merged focus CSV)."""
    from biahub.estimate_stabilization import estimate_xy_stabilization_per_position
    from biahub.settings import EstimateStabilizationSettings

    settings = yaml_to_model(Path(config), EstimateStabilizationSettings)
    input_position = Path(input_zarr) / position

    with open_ome_zarr(str(input_position), mode="r") as ds:
        channel_names = ds.channel_names
        shape = ds.data.shape

    channel_index = channel_names.index(settings.stabilization_estimation_channel)

    output_path = Path(output_dir)
    transform_dir = output_path / "xy_transforms"
    transform_dir.mkdir(parents=True, exist_ok=True)

    estimate_xy_stabilization_per_position(
        input_position_dirpath=input_position,
        output_folder_path=transform_dir,
        df_z_focus_path=Path(focus_csv),
        channel_index=channel_index,
        center_crop_xy=settings.stack_reg_settings.center_crop_xy,
        t_reference=settings.stack_reg_settings.t_reference,
        verbose=settings.verbose,
    )

    click.echo(f"XY stabilization estimation done: {position}")
    num_cpus, mem_per_cpu = estimate_resources(
        shape=shape, ram_multiplier=8, max_num_cpus=16
    )
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("estimate-stabilization-pcc")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
def estimate_stabilization_pcc(
    input_zarr: str, position: str, config: str, output_dir: str
):
    """Estimate XYZ stabilization via phase cross-correlation for a single position."""
    from biahub.estimate_stabilization import (
        estimate_xyz_stabilization_pcc_per_position,
    )
    from biahub.registration.utils import save_transforms
    from biahub.settings import EstimateStabilizationSettings, StabilizationSettings

    settings = yaml_to_model(Path(config), EstimateStabilizationSettings)
    input_position = Path(input_zarr) / position

    with open_ome_zarr(str(input_position), mode="r") as ds:
        channel_names = ds.channel_names
        voxel_size = ds.scale
        shape = ds.data.shape

    channel_index = channel_names.index(settings.stabilization_estimation_channel)

    output_path = Path(output_dir)
    transforms_dir = output_path / "transforms_per_position"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    shifts_dir = output_path / "shifts_per_position"
    shifts_dir.mkdir(parents=True, exist_ok=True)

    transforms = estimate_xyz_stabilization_pcc_per_position(
        input_position_dirpath=input_position,
        output_folder_path=transforms_dir,
        output_shifts_path=shifts_dir,
        channel_index=channel_index,
        phase_cross_corr_settings=settings.phase_cross_corr_settings,
        verbose=settings.verbose,
    )

    position_filename = position.replace("/", "_")
    model = StabilizationSettings(
        stabilization_type=settings.stabilization_type,
        stabilization_method=settings.stabilization_method,
        stabilization_estimation_channel=settings.stabilization_estimation_channel,
        stabilization_channels=settings.stabilization_channels,
        affine_transform_zyx_list=[],
        time_indices="all",
        output_voxel_size=voxel_size,
    )

    if settings.eval_transform_settings:
        from biahub.registration.utils import evaluate_transforms

        T, C, Z, Y, X = shape
        transforms = evaluate_transforms(
            transforms=transforms,
            shape_zyx=(Z, Y, X),
            validation_window_size=settings.eval_transform_settings.validation_window_size,
            validation_tolerance=settings.eval_transform_settings.validation_tolerance,
            interpolation_window_size=settings.eval_transform_settings.interpolation_window_size,
            interpolation_type=settings.eval_transform_settings.interpolation_type,
            verbose=settings.verbose,
        )

    save_transforms(
        model=model,
        transforms=transforms,
        output_filepath_settings=output_path / "xyz_stabilization_settings" / f"{position_filename}.yml",
        verbose=settings.verbose,
    )

    click.echo(f"PCC stabilization estimation done: {position}")
    num_cpus, mem_per_cpu = estimate_resources(
        shape=shape, ram_multiplier=16, max_num_cpus=16
    )
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("estimate-stabilization-beads")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
def estimate_stabilization_beads(
    input_zarr: str, position: str, config: str, output_dir: str
):
    """Estimate stabilization from beads on a single reference FOV (one-shot)."""
    from biahub.registration.beads import estimate_tczyx
    from biahub.registration.utils import save_transforms
    from biahub.settings import EstimateStabilizationSettings, StabilizationSettings

    settings = yaml_to_model(Path(config), EstimateStabilizationSettings)
    input_position = Path(input_zarr) / position

    with open_ome_zarr(str(input_position), mode="r") as ds:
        channel_names = ds.channel_names
        voxel_size = ds.scale
        shape = ds.data.shape
        channel_tczyx = ds.data.dask_array()

    channel_index = channel_names.index(settings.stabilization_estimation_channel)

    output_path = Path(output_dir)

    xyz_transforms = estimate_tczyx(
        mov_tczyx=channel_tczyx,
        ref_tczyx=channel_tczyx,
        mov_channel_index=channel_index,
        ref_channel_index=channel_index,
        beads_match_settings=settings.beads_match_settings,
        affine_transform_settings=settings.affine_transform_settings,
        verbose=settings.verbose,
        output_folder_path=output_path,
        mode="stabilization",
    )

    model = StabilizationSettings(
        stabilization_type=settings.stabilization_type,
        stabilization_method=settings.stabilization_method,
        stabilization_estimation_channel=settings.stabilization_estimation_channel,
        stabilization_channels=settings.stabilization_channels,
        affine_transform_zyx_list=[],
        time_indices="all",
        output_voxel_size=voxel_size,
    )

    if settings.eval_transform_settings:
        from biahub.registration.utils import evaluate_transforms

        T, C, Z, Y, X = shape
        xyz_transforms = evaluate_transforms(
            transforms=xyz_transforms,
            shape_zyx=(Z, Y, X),
            validation_window_size=settings.eval_transform_settings.validation_window_size,
            validation_tolerance=settings.eval_transform_settings.validation_tolerance,
            interpolation_window_size=settings.eval_transform_settings.interpolation_window_size,
            interpolation_type=settings.eval_transform_settings.interpolation_type,
            verbose=settings.verbose,
        )

    save_transforms(
        model=model,
        transforms=xyz_transforms,
        output_filepath_settings=output_path / "xyz_stabilization_settings.yml",
        verbose=settings.verbose,
    )

    click.echo(f"Beads stabilization estimation done: {position}")
    num_cpus, mem_per_cpu = estimate_resources(
        shape=shape, ram_multiplier=8, max_num_cpus=16
    )
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


# ---------------------------------------------------------------------------
# PSF estimation
# ---------------------------------------------------------------------------


@nf_cli.command("estimate-psf")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
def nf_estimate_psf(input_zarr: str, position: str, config: str, output_zarr: str):
    """Estimate average PSF from bead images for a single position."""
    import torch

    from iohub.ngff.models import TransformationMeta

    from biahub.characterize_psf import detect_peaks, extract_beads
    from biahub.settings import PsfFromBeadsSettings

    settings = yaml_to_model(Path(config), PsfFromBeadsSettings)
    patch_size = (
        settings.axis0_patch_size,
        settings.axis1_patch_size,
        settings.axis2_patch_size,
    )

    position_path = Path(input_zarr) / position
    with open_ome_zarr(str(position_path), mode="r") as ds:
        zyx_data = ds["0"][0, 0]
        zyx_scale = ds.scale[-3:]

    bead_detection_settings = {
        "block_size": (64, 64, 32),
        "blur_kernel_size": 3,
        "nms_distance": 32,
        "min_distance": 50,
        "threshold_abs": 200.0,
        "max_num_peaks": 2000,
        "exclude_border": (5, 10, 5),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    peaks = detect_peaks(zyx_data, **bead_detection_settings, verbose=True)
    beads, _ = extract_beads(
        zyx_data=zyx_data,
        points=peaks,
        scale=zyx_scale,
        patch_size=tuple(a * b for a, b in zip(patch_size, zyx_scale, strict=True)),
    )

    if not beads:
        raise click.ClickException(f"No beads detected in {position}.")

    filtered_beads = [x for x in beads if x.shape == beads[0].shape]
    bzyx_data = np.stack(filtered_beads)
    normalized = bzyx_data / np.max(bzyx_data, axis=(-3, -2, -1))[:, None, None, None]
    average_psf = np.mean(normalized, axis=0)
    average_psf -= np.min(average_psf)
    average_psf /= np.max(average_psf)

    output_path = Path(output_zarr)
    with open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=["PSF"]
    ) as out:
        pos = out.create_position("0", "0", "0")
        pos.create_zeros(
            name="0",
            shape=(1, 1) + average_psf.shape,
            chunks=(1, 1) + average_psf.shape,
            dtype=np.float32,
            transform=[
                TransformationMeta(type="scale", scale=(1.0, 1.0) + tuple(zyx_scale))
            ],
        )
        pos["0"][0, 0] = average_psf

    click.echo(f"PSF estimated from {len(filtered_beads)} beads: {position}")
    click.echo("RESOURCES:4 32")


# ---------------------------------------------------------------------------
# Deconvolution
# ---------------------------------------------------------------------------


@nf_cli.command("init-deconvolve")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
@click.option("--psf-zarr", required=True, type=click.Path(exists=True))
@click.option("--tf-zarr", required=True, type=click.Path())
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def init_deconvolve(
    input_zarr: str, output_zarr: str, psf_zarr: str, tf_zarr: str, config: str
):
    """Create empty output plate, compute transfer function, and estimate resources."""
    import torch

    from iohub.ngff.models import TransformationMeta

    from biahub.settings import DeconvolveSettings

    settings = yaml_to_model(Path(config), DeconvolveSettings)
    position_keys, channel_names, shape, scale = read_plate_metadata(input_zarr)
    T, C, Z, Y, X = shape

    create_empty_plate(
        store_path=Path(output_zarr),
        position_keys=position_keys,
        channel_names=channel_names,
        shape=shape,
        scale=scale,
        version="0.5",
    )
    copy_position_metadata(Path(input_zarr), Path(output_zarr))
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")

    with open_ome_zarr(str(Path(psf_zarr) / "0/0/0"), mode="r") as psf_ds:
        psf_data = psf_ds["0"][0, 0]
        psf_scale = psf_ds.scale

    zyx_padding = np.array((Z, Y, X)) - np.array(psf_data.shape)
    pad_width = [
        (x // 2, x // 2) if x % 2 == 0 else (x // 2, x // 2 + 1) for x in zyx_padding
    ]
    padded_psf = np.pad(psf_data, pad_width=pad_width, mode="constant", constant_values=0)
    transfer_function = torch.abs(torch.fft.fftn(torch.tensor(padded_psf)))
    transfer_function /= torch.max(transfer_function)
    tf_numpy = transfer_function.numpy()

    tf_path = Path(tf_zarr)
    with open_ome_zarr(tf_path, layout="fov", mode="w-", channel_names=["PSF"]) as tf_ds:
        tf_ds.create_image(
            "0",
            tf_numpy[None, None],
            chunks=(1, 1, 256) + (Y, X),
            transform=[TransformationMeta(type="scale", scale=psf_scale)],
        )

    click.echo(f"Transfer function saved to {tf_zarr}")

    num_cpus, mem_per_cpu = estimate_resources(
        shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16
    )
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("run-deconvolve")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--tf-zarr", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--num-threads", "-j", default=1, type=int)
def run_deconvolve(
    input_zarr: str,
    output_zarr: str,
    tf_zarr: str,
    position: str,
    config: str,
    num_threads: int,
):
    """Apply deconvolution to a single position using precomputed transfer function."""
    from biahub.deconvolve import deconvolve
    from biahub.settings import DeconvolveSettings

    settings = yaml_to_model(Path(config), DeconvolveSettings)
    input_position = Path(input_zarr) / position
    output_position = Path(output_zarr) / position

    process_single_position(
        deconvolve,
        str(input_position),
        str(output_position),
        num_processes=num_threads,
        transfer_function_store_path=str(tf_zarr),
        regularization_strength=float(settings.regularization_strength),
    )

    click.echo(f"Deconvolution done: {position}")


# ---------------------------------------------------------------------------
# Flip
# ---------------------------------------------------------------------------


@nf_cli.command("flip")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--x", "flip_x", is_flag=True, help="Flip along X axis")
@click.option("--y", "flip_y", is_flag=True, help="Flip along Y axis")
def nf_flip(input_zarr: str, position: str, flip_x: bool, flip_y: bool):
    """Flip data in-place for a single position along X and/or Y."""
    if not flip_x and not flip_y:
        raise click.ClickException("Provide at least --x or --y.")

    position_path = Path(input_zarr) / position
    with open_ome_zarr(str(position_path), mode="r+") as ds:
        array = ds["0"]
        T, C, _, _, _ = array.shape
        for t in range(T):
            for c in range(C):
                data = array[t, c]
                if flip_x:
                    data = data[:, :, ::-1]
                if flip_y:
                    data = data[:, ::-1, :]
                array[t, c] = data

    click.echo(f"Flipped: {position}")
    click.echo("RESOURCES:1 2")
