from collections.abc import Callable, Iterable
from pathlib import Path

import click
import submitit

from tqdm import tqdm

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import sbatch_to_submitit
from biahub.cli.utils import estimate_resources, get_submitit_cluster

# A single unit of work: positional args (tuple) and keyword args (dict) for the
# queued callable. ``JobRunner``/``SlurmExecutor.submit_batch`` iterate over a list
# of these and call ``func(*args, **kwargs)`` once per entry.
JobArgs = tuple[tuple, dict]


def resolve_slurm_args(
    slurm_args: dict,
    shape: tuple[int, ...],
    *,
    ram_multiplier: float = 1.0,
    max_num_cpus: int = 64,
) -> dict:
    """
    Return a copy of ``slurm_args`` with cpus/mem sized to the data volume.

    Every image-processing CLI shares this pattern: a static base of SLURM params
    (job name, partition, time, array parallelism, ...) plus two fields that should
    scale with the data -- ``slurm_cpus_per_task`` and ``slurm_mem_per_cpu``. This
    estimates those two from ``shape`` and overrides them, leaving the rest of the
    base untouched (so extra keys like ``slurm_use_srun`` survive).

    Use the returned ``slurm_args["slurm_cpus_per_task"]`` as the kernel's
    ``num_workers`` to keep a single source of truth for the CPU count.

    Parameters
    ----------
    slurm_args : dict
        Base SLURM parameters. Comes from ``settings.slurm_settings.model_dump()`` or
        a literal dict. Not mutated.
    shape : tuple[int, ...]
        Data shape passed to :func:`estimate_resources` (typically ``(T, C, Z, Y, X)``).
    ram_multiplier : float, optional
        Per-CLI memory scaling for ``estimate_resources``. Default 1.0.
    max_num_cpus : int, optional
        Per-CLI cap for ``estimate_resources``. Default 64.
    """
    num_cpus, gb_ram = estimate_resources(
        shape=shape, ram_multiplier=ram_multiplier, max_num_cpus=max_num_cpus
    )
    resolved = dict(slurm_args)
    resolved["slurm_cpus_per_task"] = num_cpus
    resolved["slurm_mem_per_cpu"] = f"{gb_ram}G"
    return resolved


def wait_for_jobs_to_finish(jobs: list[submitit.Job]) -> None:
    """
    Wait for SLURM jobs to finish using a progress bar with tqdm.

    Parameters
    ----------
    jobs : list
        A list of submitit Job objects that represent the SLURM jobs to wait for.

    Returns
    -------
    None
    """
    for job in tqdm(
        submitit.helpers.as_completed(jobs), total=len(jobs), desc="Waiting for jobs to finish"
    ):
        try:
            pass  # as_completed polls every 10 seconds by default, so we don't need to do anything here
        except Exception as e:
            print(f"Job {job.job_id} failed with exception: {e}")


class SlurmExecutor:
    """
    Submit batches of SLURM jobs via a shared ``submitit.AutoExecutor`` wrapper.

    Captures the job-submission boilerplate shared by every biahub CLI: cluster
    selection, parameter setup, optional ``sbatch`` overrides, batched submission
    inside a clean environment, job-id logging, and monitoring.

    The generic :meth:`submit_batch` (``func`` + a list of ``(args, kwargs)``) is the
    core abstraction and serves every shape of work: one-job-per-position image
    processing (deskew, register, ...) *and* estimator fan-outs over an arbitrary
    index such as timepoints (``estimate_registration``'s ``estimate_tczyx``).

    Most callers should use :class:`JobRunner` (backend selection plus a single
    ``add``/``execute`` flow). Modules with multiple batches (e.g. ``register``,
    ``stabilize``) or dependency chains (e.g. ``virtual_stain``) construct a
    ``SlurmExecutor`` directly, call :meth:`submit_batch` more than once, and
    finalize with :meth:`monitor` (live progress, image modules) or :meth:`wait`
    (block silently, estimators).

    Parameters
    ----------
    output_dirpath : Union[str, Path]
        The run's output store. submitit logs and the job-id file are written to a
        ``slurm_output`` directory beside it (``output_dirpath.parent / "slurm_output"``).
    slurm_args : dict
        Parameters passed to ``executor.update_parameters`` (e.g. ``slurm_job_name``,
        ``slurm_cpus_per_task``, ``slurm_mem_per_cpu``, ``slurm_partition``).
    local : bool, optional
        Run locally instead of submitting to SLURM. Used to resolve ``cluster`` when
        ``cluster`` is not given. Default False.
    cluster : str, optional
        A pre-resolved submitit cluster backend (e.g. ``"slurm"``/``"local"``). If
        given it takes precedence over ``local``; estimators that already called
        ``get_submitit_cluster`` upstream pass it through here.
    sbatch_filepath : Union[str, Path], optional
        Optional sbatch file whose parsed contents override ``slurm_args``.
    """

    def __init__(
        self,
        output_dirpath: str | Path,
        slurm_args: dict,
        *,
        local: bool = False,
        cluster: str | None = None,
        sbatch_filepath: str | Path | None = None,
    ) -> None:
        # SLURM logs + job-id file live in a `slurm_output` dir beside the output store.
        self.slurm_out_path = Path(output_dirpath).parent / "slurm_output"

        # Override defaults if an sbatch file is provided. Copy so we don't mutate
        # the caller's dict.
        slurm_args = dict(slurm_args)
        if sbatch_filepath:
            slurm_args.update(sbatch_to_submitit(sbatch_filepath))
        self.slurm_args = slurm_args

        # Accept a pre-resolved cluster (estimators pass one in) or derive it from
        # the ``local`` flag (image-processing modules).
        if cluster is None:
            cluster = get_submitit_cluster(local)
        self._executor = submitit.AutoExecutor(folder=self.slurm_out_path, cluster=cluster)
        self._executor.update_parameters(**slurm_args)

        # Accumulates jobs across all submit_batch calls so the log file and
        # monitor reflect every job this executor submitted.
        self.jobs: list[submitit.Job] = []

    def submit_batch(self, func: Callable, job_args: Iterable[JobArgs]) -> list[submitit.Job]:
        """
        Submit one batch of jobs, all calling ``func`` with per-job args.

        Parameters
        ----------
        func : Callable
            The callable to submit (e.g. ``process_single_position``). The same
            callable is used for every job in the batch.
        job_args : Iterable[JobArgs]
            One ``(args, kwargs)`` tuple per job; each job runs
            ``func(*args, **kwargs)``.

        Returns
        -------
        list[submitit.Job]
            The jobs submitted in this batch (also appended to ``self.jobs``).
        """
        job_args = list(job_args)
        click.echo(f"Submitting {len(job_args)} jobs: {self.slurm_args}")

        batch_jobs = []
        with submitit.helpers.clean_env(), self._executor.batch():
            for args, kwargs in job_args:
                batch_jobs.append(self._executor.submit(func, *args, **kwargs))

        self.jobs.extend(batch_jobs)
        self._write_job_ids()
        return batch_jobs

    def _write_job_ids(self) -> None:
        """Write all submitted job IDs to ``submitit_jobs_ids.log``."""
        self.slurm_out_path.mkdir(parents=True, exist_ok=True)
        log_path = self.slurm_out_path / "submitit_jobs_ids.log"
        with log_path.open("w") as log_file:
            log_file.write("\n".join(job.job_id for job in self.jobs))

    def monitor(self, position_dirpaths: list | None = None) -> None:
        """Monitor all jobs submitted by this executor with live progress."""
        monitor_jobs(self.jobs, position_dirpaths or [])

    def wait(self) -> None:
        """Block until all submitted jobs finish, without live monitoring."""
        wait_for_jobs_to_finish(self.jobs)
