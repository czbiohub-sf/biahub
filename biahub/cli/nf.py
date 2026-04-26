import logging

from pathlib import Path

import click
import numpy as np

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.cli.nf_qc import nf_qc_cli
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


nf_cli.add_command(nf_qc_cli)


@nf_cli.command("list-positions")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
def list_positions(input_zarr: str):
    """List position keys in a plate zarr (one per line, for Nextflow fan-out)."""
    with open_ome_zarr(input_zarr, mode="r") as plate:
        for name, _ in plate.positions():
            click.echo(name)


@nf_cli.command("init-resources")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--ram-multiplier", "-r", required=True, type=float)
@click.option("--max-num-cpus", default=16, type=int)
def init_resources(input_zarr: str, ram_multiplier: float, max_num_cpus: int):
    """Estimate CPU/memory resources from input zarr shape (for Nextflow fan-out)."""
    _, _, shape, _ = read_plate_metadata(input_zarr)
    num_cpus, mem_per_cpu = estimate_resources(
        shape=shape, ram_multiplier=ram_multiplier, max_num_cpus=max_num_cpus
    )
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


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
    """Create empty output zarr for deskew.

    If ``pixel_size_um`` is absent from the config, it is resolved from the
    input zarr YX scale.  The resolved config is written next to the output
    zarr as ``deskew_resolved.yml`` for downstream processes.
    """
    import yaml

    from biahub.deskew import get_deskewed_data_shape
    from biahub.settings import DeskewSettings

    config_path = Path(config)
    position_keys, channel_names, shape, scale = read_plate_metadata(input_zarr)
    T, C, Z, Y, X = shape

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "pixel_size_um" not in cfg or cfg["pixel_size_um"] is None:
        cfg["pixel_size_um"] = float(scale[-1])
        click.echo(f"Resolved pixel_size_um={cfg['pixel_size_um']} from input zarr")

    resolved_config = Path(output_zarr).parent / "deskew_resolved.yml"
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_config, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    settings = yaml_to_model(resolved_config, DeskewSettings)

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
    """Create empty output zarr and estimate resources for reconstruction.

    Reads post-deskew pixel sizes from the input zarr scale metadata and
    injects them into the reconstruction config so that the transfer function
    uses the correct voxel dimensions.  The resolved config is written next to
    the output zarr as ``reconstruct_resolved.yml``.
    """
    import yaml

    from waveorder.cli.apply_inverse_transfer_function import (
        get_reconstruction_output_metadata,
    )
    from waveorder.cli.settings import ReconstructionSettings
    from waveorder.cli.utils import create_empty_hcs_zarr
    from waveorder.cli.utils import estimate_resources as wo_estimate_resources

    config_path = Path(config)
    position_keys, _, shape, scale = read_plate_metadata(input_zarr)
    T, C, Z, Y, X = shape

    yx_pixel_size = float(scale[-1])
    z_pixel_size = float(scale[-3])

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    resolved_yx = yx_pixel_size
    resolved_z = z_pixel_size
    for section in ("phase", "birefringence", "fluorescence"):
        tf = (cfg.get(section) or {}).get("transfer_function")
        if tf is not None:
            if "yx_pixel_size" not in tf or tf["yx_pixel_size"] is None:
                tf["yx_pixel_size"] = yx_pixel_size
            else:
                resolved_yx = tf["yx_pixel_size"]
            if "z_pixel_size" not in tf or tf["z_pixel_size"] is None:
                tf["z_pixel_size"] = z_pixel_size
            else:
                resolved_z = tf["z_pixel_size"]

    resolved_config = Path(output_zarr).parent / "reconstruct_resolved.yml"
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_config, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    click.echo(f"Pixel sizes (yx={resolved_yx}, z={resolved_z}) → {resolved_config}")

    first_position_path = Path(input_zarr) / "/".join(position_keys[0])
    output_metadata = get_reconstruction_output_metadata(first_position_path, resolved_config)

    create_empty_hcs_zarr(
        store_path=Path(output_zarr),
        position_keys=position_keys,
        **output_metadata,
    )
    copy_position_metadata(Path(input_zarr), Path(output_zarr))
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")

    settings = yaml_to_model(resolved_config, ReconstructionSettings)
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


@nf_cli.command("clean-temp")
@click.argument("temp_dir", type=click.Path())
def clean_temp(temp_dir: str):
    """Remove a temp directory if it exists (idempotent pre-retry cleanup)."""
    import shutil

    path = Path(temp_dir)
    if path.exists():
        shutil.rmtree(path)
        logger.info(f"Removed stale temp directory: {path}")
    else:
        logger.info(f"No temp directory to clean: {path}")


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

    num_cpus, mem_per_cpu = estimate_resources(
        shape=(T, len(prediction_channels), Z, Y, X), ram_multiplier=16, max_num_cpus=16
    )
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
@click.option(
    "--input-images-path",
    default=None,
    type=click.Path(exists=True),
    help="Override the first non-null input_images path from config.",
)
def run_track(
    output_zarr: str,
    position: str,
    config: str,
    blank_frames_csv: str | None,
    input_images_path: str | None,
):
    """Run tracking for a single position."""
    from ultrack import MainConfig

    from biahub.cli.utils import update_model
    from biahub.settings import TrackingSettings
    from biahub.track import resolve_z_slice, track_one_position

    settings = yaml_to_model(Path(config), TrackingSettings)

    if input_images_path is not None:
        for image in settings.input_images:
            if image.path is None:
                image.path = Path(input_images_path)
                break

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

    tracking_config = update_model(MainConfig(), settings.tracking_config)

    track_one_position(
        position_key=position_key,
        input_images=settings.input_images,
        output_dirpath=output_dirpath,
        tracking_config=tracking_config,
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
@click.option(
    "--concat-data-paths",
    multiple=True,
    type=str,
    help="Override concat_data_paths from config (one per source, repeat flag).",
)
def nf_estimate_crop(
    config: str,
    output_config: str,
    lf_mask_radius: float,
    concat_data_paths: tuple[str, ...],
):
    """Estimate crop region across all positions and write updated config."""
    import glob as globmod

    from natsort import natsorted

    from biahub.cli.utils import model_to_yaml
    from biahub.estimate_crop import estimate_crop_one_position
    from biahub.settings import ConcatenateSettings

    config_path = Path(config)
    settings = yaml_to_model(config_path, ConcatenateSettings)

    if concat_data_paths:
        settings.concat_data_paths = list(concat_data_paths)

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

        process_single_position(
            copy_n_paste,
            input_position_path=input_path,
            output_position_path=output_position_path,
            input_channel_indices=input_ch_idx,
            output_channel_indices=output_ch_idx,
            input_time_indices=input_time_indices,
            output_time_indices=list(range(len(input_time_indices))),
            num_threads=num_threads,
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

    composed = np.array([a @ b for a, b in zip(transforms_a, transforms_b, strict=True)])

    output_model = settings_a.model_copy()
    output_model.affine_transform_zyx_list = composed.tolist()
    model_to_yaml(output_model, Path(output_config))

    click.echo(f"Combined transforms written to {output_config}")


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
    from biahub.settings import StabilizationSettings
    from biahub.stabilize import apply_stabilization_transform

    settings = yaml_to_model(Path(config), StabilizationSettings)
    combined_mats = np.array(settings.affine_transform_zyx_list)

    input_position = Path(input_zarr) / position
    output_position = Path(output_zarr) / position

    with open_ome_zarr(str(input_position), mode="r") as ds:
        T = ds.data.shape[0]
        channel_names = ds.channel_names

    with open_ome_zarr(str(output_position), mode="r") as ds_out:
        _, _, out_Z, out_Y, out_X = ds_out.data.shape

    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    def _stabilize_czyx(czyx_data, input_time_index, **kwargs):
        return apply_stabilization_transform(czyx_data, t_idx=input_time_index, **kwargs)

    channel_indices = [[ch] for ch in range(len(channel_names))]

    process_single_position(
        _stabilize_czyx,
        input_position,
        output_position,
        input_channel_indices=channel_indices,
        output_channel_indices=channel_indices,
        input_time_indices=time_indices,
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


@nf_cli.command("estimate-stabilization-pcc")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
def estimate_stabilization_pcc(input_zarr: str, position: str, config: str, output_dir: str):
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
        output_filepath_settings=output_path
        / "xyz_stabilization_settings"
        / f"{position_filename}.yml",
        verbose=settings.verbose,
    )

    click.echo(f"PCC stabilization estimation done: {position}")


@nf_cli.command("estimate-stabilization-beads")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
def estimate_stabilization_beads(input_zarr: str, position: str, config: str, output_dir: str):
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
    with open_ome_zarr(output_path, layout="hcs", mode="w", channel_names=["PSF"]) as out:
        pos = out.create_position("0", "0", "0")
        pos.create_zeros(
            name="0",
            shape=(1, 1) + average_psf.shape,
            chunks=(1, 1) + average_psf.shape,
            dtype=np.float32,
            transform=[TransformationMeta(type="scale", scale=(1.0, 1.0) + tuple(zyx_scale))],
        )
        pos["0"][0, 0] = average_psf

    click.echo(f"PSF estimated from {len(filtered_beads)} beads: {position}")


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
    pad_width = [(x // 2, x // 2) if x % 2 == 0 else (x // 2, x // 2 + 1) for x in zyx_padding]
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
        num_threads=num_threads,
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


# ---------------------------------------------------------------------------
# Registration estimation
# ---------------------------------------------------------------------------


@nf_cli.command("estimate-registration")
@click.option("--source-zarr", required=True, type=click.Path(exists=True))
@click.option("--target-zarr", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
@click.option("--registration-target-channel", "-rt", type=str, default=None)
@click.option("--registration-source-channel", "-rs", type=str, multiple=True)
def nf_estimate_registration(
    source_zarr: str,
    target_zarr: str,
    position: str,
    config: str,
    output_dir: str,
    registration_target_channel: str | None,
    registration_source_channel: tuple[str, ...],
):
    """Estimate affine registration between source and target for a single position."""
    from biahub.cli.utils import model_to_yaml
    from biahub.registration.utils import evaluate_transforms
    from biahub.settings import (
        EstimateRegistrationSettings,
        RegistrationSettings,
        StabilizationSettings,
    )

    settings = yaml_to_model(Path(config), EstimateRegistrationSettings)

    source_position = Path(source_zarr) / position
    target_position = Path(target_zarr) / position

    with open_ome_zarr(str(source_position), mode="r") as src:
        source_channels = src.channel_names
        source_channel_index = source_channels.index(settings.source_channel_name)
        source_data = src.data.dask_array()
        source_voxel_size = src.scale[-3:]

    with open_ome_zarr(str(target_position), mode="r") as tgt:
        target_channels = tgt.channel_names
        target_channel_index = target_channels.index(settings.target_channel_name)
        target_data = tgt.data.dask_array()
        target_voxel_size = tgt.scale[-3:]
        voxel_size = tgt.scale

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if settings.estimation_method == "beads":
        from biahub.registration.beads import estimate_tczyx

        transforms = estimate_tczyx(
            mov_tczyx=source_data,
            ref_tczyx=target_data,
            mov_channel_index=source_channel_index,
            ref_channel_index=target_channel_index,
            beads_match_settings=settings.beads_match_settings,
            affine_transform_settings=settings.affine_transform_settings,
            verbose=settings.verbose,
            output_folder_path=output_path,
            ref_voxel_size=target_voxel_size,
            mov_voxel_size=source_voxel_size,
            mode="registration",
        )
    elif settings.estimation_method == "ants":
        from biahub.registration.ants import estimate_tczyx

        transforms = estimate_tczyx(
            mov_tczyx=source_data,
            ref_tczyx=target_data,
            mov_channel_index=source_channel_index,
            ref_channel_index=target_channel_index,
            ants_registration_settings=settings.ants_registration_settings,
            affine_transform_settings=settings.affine_transform_settings,
            verbose=settings.verbose,
            output_folder_path=output_path,
        )
    else:
        raise click.ClickException(
            f"Unsupported estimation method for NF: {settings.estimation_method}. "
            "Use 'beads' or 'ants'."
        )

    reg_target_ch = registration_target_channel or settings.target_channel_name
    reg_source_chs = list(registration_source_channel) or [settings.source_channel_name]

    if settings.eval_transform_settings and len(transforms) > 1:
        transforms = evaluate_transforms(
            transforms=transforms,
            shape_zyx=source_data.shape[-3:],
            validation_window_size=settings.eval_transform_settings.validation_window_size,
            validation_tolerance=settings.eval_transform_settings.validation_tolerance,
            interpolation_window_size=settings.eval_transform_settings.interpolation_window_size,
            interpolation_type=settings.eval_transform_settings.interpolation_type,
            verbose=settings.verbose,
        )

    if len(transforms) == 1:
        model = RegistrationSettings(
            source_channel_names=reg_source_chs,
            target_channel_name=reg_target_ch,
            affine_transform_zyx=transforms[0],
        )
    else:
        model = StabilizationSettings(
            stabilization_estimation_channel=settings.target_channel_name,
            stabilization_type="affine",
            stabilization_method=settings.estimation_method,
            stabilization_channels=[
                settings.source_channel_name,
                settings.target_channel_name,
            ],
            affine_transform_zyx_list=transforms,
            time_indices="all",
            output_voxel_size=voxel_size,
        )

    model_to_yaml(model, output_path / "registration_settings.yml")

    click.echo(f"Registration estimation done: {position}")


# ---------------------------------------------------------------------------
# Registration optimization
# ---------------------------------------------------------------------------


@nf_cli.command("optimize-registration")
@click.option("--source-zarr", required=True, type=click.Path(exists=True))
@click.option("--target-zarr", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
def nf_optimize_registration(
    source_zarr: str,
    target_zarr: str,
    position: str,
    config: str,
    output: str,
):
    """Refine registration estimate using ANTs optimization for a single position."""
    from biahub.cli.utils import model_to_yaml
    from biahub.optimize_registration import _optimize_registration
    from biahub.settings import RegistrationSettings

    settings = yaml_to_model(Path(config), RegistrationSettings)

    t_idx = settings.time_indices
    if not isinstance(t_idx, int):
        t_idx = 0

    source_position = Path(source_zarr) / position
    target_position = Path(target_zarr) / position

    with open_ome_zarr(str(source_position), mode="r") as src:
        source_channel_names = src.channel_names
        source_channel_index = source_channel_names.index(settings.source_channel_names[0])
        source_data_czyx = np.asarray(src.data[t_idx])

    with open_ome_zarr(str(target_position), mode="r") as tgt:
        target_channel_names = tgt.channel_names
        target_channel_index = target_channel_names.index(settings.target_channel_name)
        target_data_czyx = np.asarray(tgt.data[t_idx])

    approx_tform = np.asarray(settings.affine_transform_zyx, dtype=np.float32)

    composed_matrix = _optimize_registration(
        source_czyx=source_data_czyx,
        target_czyx=target_data_czyx,
        initial_tform=approx_tform,
        source_channel_index=source_channel_index,
        target_channel_index=target_channel_index,
        crop=True,
        verbose=settings.verbose,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_settings = settings.model_copy()
    output_settings.affine_transform_zyx = composed_matrix.tolist()
    model_to_yaml(output_settings, output_path)

    click.echo(f"Registration optimization done: {position}")


# ---------------------------------------------------------------------------
# Registration apply: init + run
# ---------------------------------------------------------------------------


@nf_cli.command("init-register")
@click.option("--source-zarr", required=True, type=click.Path(exists=True))
@click.option("--target-zarr", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def nf_init_register(
    source_zarr: str,
    target_zarr: str,
    output_zarr: str,
    config: str,
):
    """Create empty output zarr for registration."""
    from biahub.register import find_overlapping_volume, rescale_voxel_size
    from biahub.settings import RegistrationSettings

    settings = yaml_to_model(Path(config), RegistrationSettings)
    matrix = np.array(settings.affine_transform_zyx)

    position_keys, source_channel_names, source_shape, source_scale = read_plate_metadata(
        source_zarr
    )
    T, C, Z, Y, X = source_shape
    source_shape_zyx = (Z, Y, X)

    with open_ome_zarr(str(Path(target_zarr) / "/".join(position_keys[0])), mode="r") as tgt:
        target_channel_names = tgt.channel_names
        target_shape_zyx = tgt.data.shape[-3:]

    output_voxel_size = rescale_voxel_size(matrix[:3, :3], source_scale[-3:])

    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    output_channel_names = list(target_channel_names)
    if source_zarr != target_zarr:
        output_channel_names += list(source_channel_names)

    if not settings.keep_overhang:
        Z_slice, Y_slice, X_slice = find_overlapping_volume(
            source_shape_zyx, target_shape_zyx, matrix
        )
        cropped_shape_zyx = (
            Z_slice.stop - Z_slice.start,
            Y_slice.stop - Y_slice.start,
            X_slice.stop - X_slice.start,
        )
    else:
        cropped_shape_zyx = target_shape_zyx

    output_shape = (len(time_indices), len(output_channel_names)) + tuple(cropped_shape_zyx)

    create_empty_plate(
        store_path=Path(output_zarr),
        position_keys=position_keys,
        channel_names=output_channel_names,
        shape=output_shape,
        chunks=None,
        scale=(1, 1) + tuple(output_voxel_size),
        version="0.5",
        dtype=np.float32,
    )
    copy_position_metadata(Path(source_zarr), Path(output_zarr))
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")

    num_cpus, mem_per_cpu = estimate_resources(
        shape=output_shape, ram_multiplier=5, max_num_cpus=16
    )
    click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")


@nf_cli.command("run-register")
@click.option("--source-zarr", required=True, type=click.Path(exists=True))
@click.option("--target-zarr", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--num-threads", "-j", default=1, type=int)
def nf_run_register(
    source_zarr: str,
    target_zarr: str,
    output_zarr: str,
    position: str,
    config: str,
    num_threads: int,
):
    """Apply registration transform to a single position."""
    from biahub.cli.utils import copy_n_paste_czyx
    from biahub.register import apply_affine_transform, find_overlapping_volume
    from biahub.settings import RegistrationSettings

    settings = yaml_to_model(Path(config), RegistrationSettings)
    matrix = np.array(settings.affine_transform_zyx)

    source_position = Path(source_zarr) / position
    target_position = Path(target_zarr) / position
    output_position = Path(output_zarr) / position

    with open_ome_zarr(str(source_position), mode="r") as src:
        T = src.data.shape[0]
        source_channel_names = src.channel_names
        source_shape_zyx = src.data.shape[-3:]

    with open_ome_zarr(str(target_position), mode="r") as tgt:
        target_channel_names = tgt.channel_names
        target_shape_zyx = tgt.data.shape[-3:]

    with open_ome_zarr(str(output_position), mode="r") as out:
        output_channel_names = out.channel_names

    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    if not settings.keep_overhang:
        Z_slice, Y_slice, X_slice = find_overlapping_volume(
            source_shape_zyx, target_shape_zyx, matrix
        )
        crop_slicing = [Z_slice, Y_slice, X_slice]
    else:
        crop_slicing = None

    affine_kwargs = {
        "matrix": matrix,
        "output_shape_zyx": target_shape_zyx,
        "crop_output_slicing": crop_slicing,
        "interpolation": settings.interpolation,
    }

    copy_kwargs = {
        "czyx_slicing_params": crop_slicing
        if crop_slicing
        else [
            slice(0, target_shape_zyx[0]),
            slice(0, target_shape_zyx[1]),
            slice(0, target_shape_zyx[2]),
        ],
    }

    source_input_ch = []
    source_output_ch = []
    for channel_name in source_channel_names:
        if channel_name not in settings.source_channel_names:
            continue
        source_input_ch.append([source_channel_names.index(channel_name)])
        source_output_ch.append([output_channel_names.index(channel_name)])

    if source_input_ch:
        process_single_position(
            apply_affine_transform,
            source_position,
            output_position,
            input_channel_indices=source_input_ch,
            output_channel_indices=source_output_ch,
            input_time_indices=time_indices,
            num_threads=num_threads,
            **affine_kwargs,
        )

    target_input_ch = []
    target_output_ch = []
    for channel_name in target_channel_names:
        if channel_name in settings.source_channel_names:
            continue
        target_input_ch.append([target_channel_names.index(channel_name)])
        target_output_ch.append([output_channel_names.index(channel_name)])

    if target_input_ch:
        process_single_position(
            copy_n_paste_czyx,
            target_position,
            output_position,
            input_channel_indices=target_input_ch,
            output_channel_indices=target_output_ch,
            input_time_indices=time_indices,
            num_threads=num_threads,
            **copy_kwargs,
        )

    click.echo(f"Registration done: {position}")


# ---------------------------------------------------------------------------
# Stitch estimation
# ---------------------------------------------------------------------------


@nf_cli.command("estimate-stitch")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
@click.option("--fliplr", is_flag=True)
@click.option("--flipud", is_flag=True)
@click.option("--flipxy", is_flag=True)
@click.option("--pcc-channel-name", default=None, type=str)
@click.option("--pcc-z-index", default=0, type=int)
def nf_estimate_stitch(
    input_zarr: str,
    output: str,
    fliplr: bool,
    flipud: bool,
    flipxy: bool,
    pcc_channel_name: str | None,
    pcc_z_index: int,
):
    """Estimate stitching translations from stage positions (one-shot)."""
    from collections import defaultdict

    from biahub.cli.utils import model_to_yaml
    from biahub.estimate_stitch import extract_stage_position
    from biahub.settings import StitchSettings

    plate_path = Path(input_zarr)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_ome_zarr(str(plate_path), mode="r") as plate:
        position_keys = [name for name, _ in plate.positions()]

    translation_dict = {}
    for pos_key in position_keys:
        with open_ome_zarr(str(plate_path / pos_key), mode="r") as pos_ds:
            position_name = pos_ds.zattrs["omero"]["name"]

        with open_ome_zarr(str(plate_path), mode="r") as plate_ds:
            zyx_position = extract_stage_position(plate_ds, position_name)

        translation_dict[pos_key] = zyx_position

    grouped_wells = defaultdict(dict)
    for key, value in translation_dict.items():
        well_name = "/".join(key.split("/")[:2])
        grouped_wells[well_name][key] = value

    final_translation_dict = {}
    for well_name, well_positions in grouped_wells.items():
        zyx_array = np.array(list(well_positions.values()))
        zyx_array -= np.min(zyx_array, axis=0)

        with open_ome_zarr(str(plate_path / position_keys[0]), mode="r") as ref_pos:
            scale = ref_pos.scale[2:]
        zyx_array /= scale

        if pcc_channel_name is not None:
            from stitch.stitch.tile import optimal_positions, pairwise_shifts

            tile_lut = {t.split("/")[-1]: i for i, t in enumerate(well_positions)}
            initial_guess = {
                well_name: {
                    "i": zyx_array[:, 1],
                    "j": zyx_array[:, 2],
                }
            }
            with open_ome_zarr(str(plate_path), mode="r") as p:
                channel_index = p.get_channel_index(pcc_channel_name)

            edge_list, confidence_dict = pairwise_shifts(
                well_positions,
                plate_path,
                well_name,
                flipud=flipud,
                fliplr=fliplr,
                rot90=False,
                overlap=300,
                channel_index=channel_index,
                z_index=pcc_z_index,
            )

            first_pos = list(well_positions.keys())[0]
            with open_ome_zarr(str(plate_path / first_pos), mode="r") as fp:
                tile_size = fp.data.shape[-2:]

            opt_shift_dict = optimal_positions(
                edge_list,
                tile_lut,
                well_name,
                tile_size=tile_size,
                initial_guess=initial_guess,
            )
            zyx_array[:, 1] = [a[0] for a in opt_shift_dict.values()]
            zyx_array[:, 2] = [a[1] for a in opt_shift_dict.values()]

        if fliplr:
            zyx_array[:, 2] *= -1
        if flipud:
            zyx_array[:, 1] *= -1
        if flipxy:
            zyx_array[:, [1, 2]] = zyx_array[:, [2, 1]]

        zyx_array -= np.minimum(zyx_array.min(axis=0), 0)

        for i, fov_name in enumerate(well_positions.keys()):
            final_translation_dict[fov_name] = list(np.round(zyx_array[i], 2))

    settings = StitchSettings(channels=None, total_translation=final_translation_dict)
    model_to_yaml(settings, output_path)

    click.echo(f"Stitch estimation done ({len(position_keys)} positions)")
    click.echo("RESOURCES:4 16")


# ---------------------------------------------------------------------------
# Stitch apply (per-well)
# ---------------------------------------------------------------------------


@nf_cli.command("stitch")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
@click.option("--well", required=True, type=str)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--blending-exponent", "-b", type=float, default=1.0)
def nf_stitch(
    input_zarr: str,
    output_zarr: str,
    well: str,
    config: str,
    blending_exponent: float,
):
    """Stitch FOVs for a single well with distance-weighted blending."""
    from iohub.ngff import TransformationMeta

    from biahub.settings import StitchSettings
    from biahub.stitch import (
        get_output_shape,
        list_of_nd_slices_from_array_shape,
        write_output_chunk,
    )

    settings = yaml_to_model(Path(config), StitchSettings)
    plate_path = Path(input_zarr)
    output_path = Path(output_zarr)

    input_plate = open_ome_zarr(str(plate_path), mode="r")
    input_channels = input_plate.channel_names
    if settings.channels is None:
        settings.channels = input_channels
    channel_idx = np.asarray([input_channels.index(ch) for ch in settings.channels])

    fov_shifts = {
        k: v for k, v in settings.total_translation.items() if k.startswith(well + "/")
    }
    if not fov_shifts:
        raise click.ClickException(f"No FOVs found for well {well}")

    first_fov_name = list(fov_shifts.keys())[0]
    input_fov_shape = input_plate[first_fov_name].data.shape
    output_shape_zyx = get_output_shape(fov_shifts, input_fov_shape)
    output_chunk_size = (
        1,
        1,
        output_shape_zyx[0],
        *input_plate[first_fov_name].data.chunks[-2:],
    )
    output_scale = input_plate[first_fov_name].scale

    output_shape = (
        input_fov_shape[0],
        len(channel_idx),
    ) + output_shape_zyx

    output_plate = open_ome_zarr(
        str(output_path), layout="hcs", mode="a", channel_names=settings.channels
    )
    output_position = output_plate.create_position(
        first_fov_name.split("/")[0],
        first_fov_name.split("/")[1],
        "0",
    )
    _ = output_position.create_zeros(
        "0",
        shape=output_shape,
        chunks=(1, 1, 10, output_chunk_size[-2], output_chunk_size[-1]),
        dtype=np.float16,
        transform=[TransformationMeta(type="scale", scale=output_scale)],
    )

    chunk_list = list_of_nd_slices_from_array_shape(output_shape_zyx, output_chunk_size[2:])

    for chunk in chunk_list:
        write_output_chunk(
            chunk,
            fov_shifts,
            channel_idx,
            input_plate,
            input_fov_shape,
            output_position,
            False,
            blending_exponent,
        )

    click.echo(f"Stitching done: well {well}")
