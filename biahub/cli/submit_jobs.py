import click
import submitit
from pathlib import Path
from typing import Callable, Any, Optional

from biahub.cli.utils import estimate_resources
from biahub.cli.parsing import sbatch_to_submitit

def submit_jobs_with_submitit(
    job_name: str,
    function: Callable,
    args_list: list[tuple[Any, ...]],
    output_dirpath: str,
    shape: tuple[int, int, int, int, int],
    ram_multiplier: int = 16,
    sbatch_filepath: Optional[str] = None,
    local: bool = False,
) -> list[submitit.Job]:
    """
    Submit a batch of SLURM jobs using Submitit.

    Parameters:
        job_name: SLURM job name.
        function: The Python function to execute.
        args_list: List of tuples representing positional args for each job.
        output_dirpath: Base output path where SLURM logs will be saved.
        shape: Input data shape (T, C, Z, Y, X) for estimating resources.
        ram_multiplier: Multiplier to convert shape to RAM needs.
        sbatch_filepath: Optional path to sbatch YAML file.
        monitor: Whether to monitor jobs after submission.
        local: Run locally instead of on SLURM.

    Returns:
        List of submitted jobs.
    """

    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    num_cpus, gb_ram_per_cpu = estimate_resources(shape=shape, ram_multiplier=ram_multiplier)

    slurm_args = {
        "slurm_job_name": job_name,
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 60,
        "slurm_partition": "preempted",
    }

    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    cluster = "local" if local else "slurm"

    click.echo(f"Preparing jobs for Submitit (cluster={cluster}):\n{slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(name=job_name, **slurm_args)

    try:
        jobs = []

        with executor.batch():
            for args in args_list:
                jobs.append(executor.submit(function, *args))

        job_ids = [job.job_id for job in jobs]
        log_path = slurm_out_path / "submitit_jobs_ids.log"

        with log_path.open("w") as log_file:
            log_file.write("\n".join(job_ids))

        click.echo(f"jobs submitted: {job_ids}")

        return jobs

    except Exception as e:
        click.echo(f"Submitit job submission failed: {e}", err=True)
        raise
