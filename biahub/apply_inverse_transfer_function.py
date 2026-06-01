import math

from pathlib import Path

import click
import submitit
import yaml

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from waveorder import sampling
from waveorder.cli.apply_inverse_transfer_function import (
    apply_inverse_transfer_function_single_position,
    get_reconstruction_output_metadata,
)
from waveorder.cli.settings import ReconstructionSettings
from waveorder.cli.utils import estimate_resources as wo_estimate_resources

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    copy_position_metadata,
    get_submitit_cluster,
    yaml_to_model,
)


def _resolve_config(
    config_filepath: Path,
    reference_position_path: Path,
    output_dirpath: Path,
) -> Path:
    """Resolve pixel sizes from zarr metadata into the reconstruction config.

    Reads voxel sizes from the input zarr scale metadata and injects them
    into the config's transfer_function sections when absent.  Writes the
    resolved config next to the output zarr as ``reconstruct_resolved.yml``.

    Returns the path to the resolved config.
    """
    with open(config_filepath) as f:
        cfg = yaml.safe_load(f)

    with open_ome_zarr(str(reference_position_path), mode="r") as ds:
        yx_pixel_size = float(ds.scale[-1])
        z_pixel_size = float(ds.scale[-3])

    for section in ("phase", "birefringence", "fluorescence"):
        tf = (cfg.get(section) or {}).get("transfer_function")
        if tf is not None:
            if "yx_pixel_size" not in tf or tf["yx_pixel_size"] is None:
                tf["yx_pixel_size"] = yx_pixel_size
            if "z_pixel_size" not in tf or tf["z_pixel_size"] is None:
                tf["z_pixel_size"] = z_pixel_size

    resolved_config = output_dirpath.parent / "reconstruct_resolved.yml"
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_config, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Pixel sizes (yx={yx_pixel_size}, z={z_pixel_size}) -> {resolved_config}")
    return resolved_config


def _upsampled_zyx(
    settings: ReconstructionSettings,
    zyx_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Return the Nyquist-upsampled ZYX shape used during TF computation."""
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


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    config_filepath: Path,
) -> tuple[tuple[int, int, int, int, int], list[str], Path]:
    """Create the empty reconstruction output plate.

    Resolves pixel sizes from zarr metadata, creates the output store, and
    copies per-position metadata from the input plate.

    Returns (input_shape, channel_names, resolved_config_path).
    """
    resolved_config = _resolve_config(
        config_filepath, input_position_dirpaths[0], output_dirpath
    )

    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as ds:
        input_shape = ds.data.shape

    output_metadata = get_reconstruction_output_metadata(
        input_position_dirpaths[0], resolved_config
    )
    output_metadata.pop("plate_metadata", None)
    channel_names = output_metadata["channel_names"]

    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[Path(p).parts[-3:] for p in input_position_dirpaths],
        **output_metadata,
    )

    input_plate = Path(input_position_dirpaths[0]).parents[2]
    copy_position_metadata(input_plate, output_dirpath)

    click.echo(f"Created {output_dirpath} ({len(input_position_dirpaths)} positions)")
    return input_shape, channel_names, resolved_config


def apply_inverse_transfer_function(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    num_processes: int = 1,
    sbatch_filepath: Path | None = None,
    local: bool = False,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
) -> None:
    """Apply an inverse transfer function to a dataset.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to input positions.
    transfer_function_dirpath : Path
        Path to transfer function zarr (ignored in ``--init`` mode).
    config_filepath : Path
        Path to YAML reconstruction config.
    output_dirpath : Path
        Path to output zarr.
    num_processes : int
        Number of parallel processes per position.
    sbatch_filepath : Path, optional
        SBATCH file with slurm parameter overrides.
    local : bool
        Legacy flag — use ``cluster='local'`` instead.
    cluster : str
        Execution cluster: 'slurm', 'local', or 'debug'.
    monitor : bool
        Monitor submitted SLURM jobs.
    init_only : bool
        Only initialize the output store and exit.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    if init_only:
        input_shape, _, resolved_config = _init_output_plate(
            input_position_dirpaths, output_dirpath, config_filepath
        )
        T, C, Z, Y, X = input_shape
        settings = yaml_to_model(resolved_config, ReconstructionSettings)

        num_cpus, mem_per_cpu = wo_estimate_resources(list(input_shape), settings, 1)
        click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")

        uZ, uY, uX = _upsampled_zyx(settings, (Z, Y, X))
        tf_cpus, tf_mem = wo_estimate_resources([1, 1, uZ, uY, uX], settings, 1)
        click.echo(f"TF_RESOURCES:{tf_cpus} {tf_cpus * tf_mem}")
        return

    output_metadata = get_reconstruction_output_metadata(
        input_position_dirpaths[0], config_filepath
    )
    channel_names = output_metadata["channel_names"]

    resolved_cluster = get_submitit_cluster(local=local, cluster=cluster)

    # --cluster debug: apply inverse TF per-position in-process.
    # Nextflow already handles per-position fan-out and resource scheduling,
    # so the CLI runs work in-process via submitit's DebugExecutor rather than
    # submitting its own SLURM jobs.  See also:
    # examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md
    if resolved_cluster == "debug":
        for pos_path in input_position_dirpaths:
            output_position = output_dirpath / Path(*pos_path.parts[-3:])
            apply_inverse_transfer_function_single_position(
                pos_path,
                transfer_function_dirpath,
                config_filepath,
                output_position,
                num_processes,
                channel_names,
            )
            click.echo(f"Apply-inv-tf done: {'/'.join(pos_path.parts[-3:])}")
        return

    # SLURM / local: create output plate and fan out via submitit
    output_metadata.pop("plate_metadata", None)
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        **output_metadata,
    )
    input_plate = Path(input_position_dirpaths[0]).parents[2]
    copy_position_metadata(input_plate, output_dirpath)

    settings = yaml_to_model(config_filepath, ReconstructionSettings)
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as ds:
        input_shape = ds.data.shape
    num_cpus, gb_ram_per_cpu = wo_estimate_resources(
        list(input_shape), settings, num_processes
    )

    slurm_args = {
        "slurm_job_name": "apply-inverse-transfer-function",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_time": 360,
        "slurm_partition": "cpu",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for pos_path in input_position_dirpaths:
            jobs.append(
                executor.submit(
                    apply_inverse_transfer_function_single_position,
                    pos_path,
                    transfer_function_dirpath,
                    config_filepath,
                    output_dirpath / Path(*pos_path.parts[-3:]),
                    num_processes,
                    channel_names,
                )
            )

    job_ids = [job.job_id for job in jobs]
    slurm_out_path.mkdir(exist_ok=True)
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("apply-inv-tf")
@input_position_dirpaths()
@click.option(
    "--transfer-function-dirpath",
    "-t",
    default=None,
    type=click.Path(),
    help="Path to transfer function zarr (not required for --init).",
)
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
def apply_inverse_transfer_function_cli(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: str | None,
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: Path | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
):
    r"""Apply an inverse transfer function to a dataset using a configuration file.

    Applies a transfer function to all positions in the list `input-position-dirpaths`,
    so all positions must have the same TCZYX shape.

    \b
    Full SLURM fan-out:
    >>> biahub apply-inv-tf -i ./input.zarr/*/*/* -t ./tf.zarr -c ./config.yml -o ./output.zarr

    \b
    Initialize the output plate only (Nextflow init step):
    >>> biahub apply-inv-tf --init -i ./input.zarr/*/*/* -c ./config.yml -o ./output.zarr

    \b
    In-process run of a single position (Nextflow per-position step):
    >>> biahub apply-inv-tf --cluster debug -i ./input.zarr/A/1/0 -t ./tf.zarr -c ./config.yml -o ./output.zarr
    """
    if not init_only and transfer_function_dirpath is None:
        raise click.UsageError(
            "--transfer-function-dirpath / -t is required unless using --init."
        )

    apply_inverse_transfer_function(
        input_position_dirpaths=input_position_dirpaths,
        transfer_function_dirpath=Path(transfer_function_dirpath)
        if transfer_function_dirpath
        else None,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
    )


if __name__ == "__main__":
    apply_inverse_transfer_function_cli()
