import os

from pathlib import Path

import click
import submitit
import torch

from iohub import open_ome_zarr
from waveorder.cli.apply_inverse_transfer_function import (
    apply_inverse_transfer_function_single_position,
    get_reconstruction_output_metadata,
)
from waveorder.cli.parsing import transfer_function_dirpath
from waveorder.cli.settings import ReconstructionSettings
from waveorder.cli.utils import create_empty_hcs_zarr, estimate_resources

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    monitor,
    num_processes,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import yaml_to_model


def apply_inverse_transfer_function(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    num_processes: int,
    sbatch_filepath: Path,
    local: bool = False,
    monitor: bool = True,
) -> None:

    output_metadata = get_reconstruction_output_metadata(
        input_position_dirpaths[0], config_filepath
    )

    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        **output_metadata,
    )
    # Initialize torch num of threads and interoeration operations
    if num_processes > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # Estimate resources
    with open_ome_zarr(input_position_dirpaths[0]) as input_dataset:
        T, C, Z, Y, X = input_dataset["0"].shape

    settings = yaml_to_model(config_filepath, ReconstructionSettings)

    num_cpus, gb_ram_per_cpu = estimate_resources([T, C, Z, Y, X], settings, num_processes)
    num_jobs = len(input_position_dirpaths)

    # Prepare and submit jobs
    click.echo(
        f"Preparing {num_jobs} job{'s, each with' if num_jobs > 1 else ' with'} "
        f"{num_cpus} CPU{'s' if num_cpus > 1 else ''} and "
        f"{gb_ram_per_cpu} GB of memory per CPU."
    )

    name_without_ext = os.path.splitext(Path(output_dirpath).name)[0]
    slurm_out_path = output_dirpath.parent / "slurm_output"

    executor_folder = os.path.join(
        Path(output_dirpath).parent.absolute(), name_without_ext + "_logs"
    )
    executor = submitit.AutoExecutor(folder=Path(executor_folder))

    slurm_args = {
        "slurm_job_name": "apply-inverse-transfer-function",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_time": 60,
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo('Submitting SLURM jobs...')
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_dirpath in input_position_dirpaths:
            job = executor.submit(
                apply_inverse_transfer_function_single_position,
                input_position_dirpath,
                transfer_function_dirpath,
                config_filepath,
                output_dirpath / Path(*input_position_dirpath.parts[-3:]),
                num_processes,
                output_metadata["channel_names"],
            )
            jobs.append(job)

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("apply-inv-tf")
@input_position_dirpaths()
@transfer_function_dirpath()
@config_filepath()
@output_dirpath()
@num_processes()
@sbatch_filepath()
@local()
@monitor()
def apply_inverse_transfer_function_cli(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    num_processes: int,
    sbatch_filepath: Path,
    local: bool = False,
    monitor: bool = True,
):
    """
    Apply an inverse transfer function to a dataset using a configuration file.

    Applies a transfer function to all positions in the list `input-position-dirpaths`,
    so all positions must have the same TCZYX shape.

    Appends channels to ./output.zarr, so multiple reconstructions can fill a single store.

    See https://github.com/mehta-lab/waveorder/tree/main/docs/examples for example configuration files.

    >> biahub apply-inv-tf -i ./input.zarr/*/*/* -t ./transfer-function.zarr -c /examples/birefringence.yml -o ./output.zarr
    """

    apply_inverse_transfer_function(
        input_position_dirpaths=input_position_dirpaths,
        transfer_function_dirpath=transfer_function_dirpath,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        num_processes=num_processes,
        sbatch_filepath=sbatch_filepath,
        local=local,
        monitor=monitor,
    )


if __name__ == "__main__":
    apply_inverse_transfer_function_cli()
