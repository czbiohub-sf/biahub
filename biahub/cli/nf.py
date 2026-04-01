import logging
from pathlib import Path

import click
import numpy as np

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate

from biahub.cli.utils import _check_nan_n_zeros, yaml_to_model

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

    with open_ome_zarr(input_zarr, mode="r") as plate:
        position_keys = []
        channel_names = scale = None
        T = C = Z = Y = X = 0
        for name, pos in plate.positions():
            position_keys.append(tuple(name.split("/")))
            if channel_names is None:
                channel_names = pos.channel_names
                T, C, Z, Y, X = pos.data.shape
                scale = pos.scale

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
        shape=(T, C, Z, Y, X),
        chunks=None,
        scale=scale,
        version="0.5",
        dtype=np.float32,
    )
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")


@nf_cli.command("run-flat-field")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def run_flat_field(input_zarr: str, output_zarr: str, position: str, config: str):
    """Apply flat-field correction to a single position (all timepoints)."""
    from biahub.flat_field_correction import flat_field_single_timepoint
    from biahub.settings import FlatFieldCorrectionSettings

    input_position = Path(input_zarr) / position
    output_zarr_path = Path(output_zarr)
    settings = yaml_to_model(Path(config), FlatFieldCorrectionSettings)

    with open_ome_zarr(str(input_position), mode="r") as ds:
        all_channel_names = ds.channel_names
        T = ds.data.shape[0]

    channel_names = settings.channel_names if settings.channel_names else all_channel_names

    for t_idx in range(T):
        flat_field_single_timepoint(input_position, output_zarr_path, channel_names, t_idx)

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

    with open_ome_zarr(input_zarr, mode="r") as plate:
        position_keys = []
        channel_names = None
        T = C = Z = Y = X = 0
        for name, pos in plate.positions():
            position_keys.append(tuple(name.split("/")))
            if channel_names is None:
                channel_names = pos.channel_names
                T, C, Z, Y, X = pos.data.shape

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
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")


@nf_cli.command("run-deskew")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def run_deskew(input_zarr: str, output_zarr: str, position: str, config: str):
    """Deskew a single position (all timepoints and channels)."""
    from biahub.deskew import deskew
    from biahub.settings import DeskewSettings

    input_position = Path(input_zarr) / position
    output_position = Path(output_zarr) / position
    settings = yaml_to_model(Path(config), DeskewSettings)

    deskew_kwargs = {
        "ls_angle_deg": settings.ls_angle_deg,
        "px_to_scan_ratio": settings.px_to_scan_ratio,
        "keep_overhang": settings.keep_overhang,
        "average_n_slices": settings.average_n_slices,
    }

    with open_ome_zarr(str(input_position), mode="r") as input_ds:
        T, C, _, _, _ = input_ds.data.shape

        for t_idx in range(T):
            for c_idx in range(C):
                click.echo(f"Deskewing {position} t={t_idx} c={c_idx}")
                zyx_data = np.asarray(input_ds.data[t_idx, c_idx])

                if _check_nan_n_zeros(zyx_data):
                    click.echo(f"  Skipping (empty)")
                    continue

                deskewed = deskew(zyx_data, **deskew_kwargs)

                with open_ome_zarr(str(output_position), mode="r+") as output_ds:
                    output_ds[0][t_idx, c_idx] = deskewed

    with open_ome_zarr(str(output_position), mode="r+") as output_ds:
        output_ds.zattrs["extra_metadata"] = {"deskew": settings.model_dump()}

    click.echo(f"Deskew done: {position}")


# ---------------------------------------------------------------------------
# Reconstruct
# ---------------------------------------------------------------------------


@nf_cli.command("init-reconstruct")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path())
@click.option("--tf-path", "-t", required=True, type=click.Path())
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def init_reconstruct(input_zarr: str, output_zarr: str, tf_path: str, config: str):
    """Compute transfer function and create empty output zarr for reconstruction."""
    from waveorder.cli.apply_inverse_transfer_function import (
        get_reconstruction_output_metadata,
    )
    from waveorder.cli.compute_transfer_function import (
        compute_transfer_function_cli as compute_transfer_function,
    )
    from waveorder.cli.utils import create_empty_hcs_zarr

    config_path = Path(config)
    tf_zarr = Path(tf_path)

    with open_ome_zarr(input_zarr, mode="r") as plate:
        position_keys = []
        first_position_path = None
        for name, pos in plate.positions():
            position_keys.append(tuple(name.split("/")))
            if first_position_path is None:
                first_position_path = Path(input_zarr) / name

    click.echo(f"Computing transfer function from {first_position_path}")
    compute_transfer_function(first_position_path, config_path, tf_zarr)
    click.echo(f"Transfer function saved to {tf_zarr}")

    output_metadata = get_reconstruction_output_metadata(first_position_path, config_path)

    create_empty_hcs_zarr(
        store_path=Path(output_zarr),
        position_keys=position_keys,
        **output_metadata,
    )
    click.echo(f"Created {output_zarr} ({len(position_keys)} positions)")


@nf_cli.command("run-apply-inv-tf")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-zarr", "-o", required=True, type=click.Path(exists=True))
@click.option("--tf-path", "-t", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--num-processes", "-j", default=1, type=int)
def run_apply_inv_tf(
    input_zarr: str,
    output_zarr: str,
    tf_path: str,
    position: str,
    config: str,
    num_processes: int,
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
        num_processes,
        output_metadata["channel_names"],
    )
    click.echo(f"Reconstruction done: {position}")
