from pathlib import Path

import click
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
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
    echo_resources,
    estimate_time_minutes,
    get_submitit_cluster,
    yaml_to_model,
)


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    config_filepath: Path,
    settings: ReconstructionSettings,
) -> tuple[tuple[int, int, int, int, int], list[str]]:
    """Create the empty reconstruction output plate.

    ``settings`` is the validated ReconstructionSettings loaded by the caller.
    Output shape/scale/version/channel names are derived by waveorder from the
    config, and per-position metadata is copied from the input plate.

    create_empty_plate is idempotent: re-running with the same positions is a
    no-op, and new positions get appended. Safe to call from both the init
    orchestrator and from per-position runs.

    Returns the input (T, C, Z, Y, X) shape and output channel names.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as ds:
        input_shape = ds.data.shape

    # Pixel sizes come from the config (the source of truth). waveorder's
    # get_reconstruction_output_metadata already emits a PixelSizeMismatchWarning
    # when the config pixel sizes disagree (>5%) with the input zarr scale, so we
    # do not warn again here.
    output_metadata = get_reconstruction_output_metadata(
        input_position_dirpaths[0], config_filepath
    )
    output_metadata.pop("plate_metadata", None)
    channel_names = output_metadata["channel_names"]

    input_plate = Path(input_position_dirpaths[0]).parents[2]
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[Path(p).parts[-3:] for p in input_position_dirpaths],
        **output_metadata,
        metadata_sources=input_plate,
    )

    return input_shape, channel_names


def apply_inverse_transfer_function(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: Path | None = None,
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
    sbatch_filepath : Path, optional
        SBATCH file with slurm parameter overrides.
    cluster : str
        Execution cluster: 'slurm', 'local', or 'debug'.
    monitor : bool
        Monitor submitted SLURM jobs.
    init_only : bool
        Only initialize the output store and exit.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    settings = yaml_to_model(config_filepath, ReconstructionSettings)
    input_shape, channel_names = _init_output_plate(
        input_position_dirpaths, output_dirpath, config_filepath, settings
    )

    max_num_cpus = 16
    num_cpus, mem_per_cpu = wo_estimate_resources(list(input_shape), settings, max_num_cpus)
    mem_gb = num_cpus * mem_per_cpu
    # Wall-time scales with total voxels reconstructed. Phase reconstruction only
    # processes the input channels named in the config, so scale T*Z*Y*X by that
    # count rather than the full C. Calibrated from a completed run: worst-case
    # ~2.4 h for a single-channel 2.8e10-voxel volume -> ~3e6 voxels/s (this step
    # is largely single-threaded per position). safety_factor covers the rest.
    T, C, Z, Y, X = input_shape
    n_reconstructed = len(settings.input_channel_names)
    time_minutes = estimate_time_minutes(
        T * n_reconstructed * Z * Y * X, voxels_per_second=3.0e6
    )
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        click.echo(
            f"Created {output_dirpath} ({len(input_position_dirpaths)} positions, "
            f"{len(settings.output_channel_names)} output channels)"
        )
        return

    resolved_cluster = get_submitit_cluster(cluster=cluster)

    # Fan out one job per position via submitit. With --cluster debug, submitit's
    # DebugExecutor runs the work in-process (the slurm_* parameters are ignored):
    # Nextflow already handles per-position fan-out and resource scheduling, so the
    # CLI must NOT submit its own SLURM jobs. See also:
    # examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md
    slurm_args = {
        "slurm_job_name": "apply-inverse-transfer-function",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_time": time_minutes,
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
                    num_cpus,
                    channel_names,
                )
            )

    job_ids = [job.job_id for job in jobs]
    slurm_out_path.mkdir(exist_ok=True)
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a DebugJob
    # but execution only happens when .wait()/.done()/.result() is called. Run each
    # one in the foreground and stream progress; monitor's async polling UI is
    # pointless against synchronous in-process jobs.
    if resolved_cluster == "debug":
        for job, path in zip(jobs, input_position_dirpaths, strict=True):
            job.wait()
            click.echo(f"Apply-inv-tf complete: {path}")
        return

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
