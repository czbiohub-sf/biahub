from collections.abc import Callable, Iterable
from pathlib import Path

import click
import submitit

from tqdm import tqdm

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import sbatch_to_submitit
from biahub.cli.utils import get_submitit_cluster

# A single unit of work: positional args (tuple) and keyword args (dict) for the
# submitted callable. ``submit_jobs``/``SlurmExecutor.submit_batch`` iterate over
# a list of these and call ``func(*args, **kwargs)`` once per entry.
JobArgs = tuple[tuple, dict]


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

    For a single batch + finalize in one call, use the :func:`submit_jobs` wrapper.
    Modules with multiple batches (e.g. ``register``, ``stabilize``) or dependency
    chains (e.g. ``virtual_stain``) call :meth:`submit_batch` more than once and
    finalize with :meth:`monitor` (live progress, image modules) or :meth:`wait`
    (block silently, estimators).

    Parameters
    ----------
    slurm_out_path : Union[str, Path]
        Directory where submitit writes its logs and the job-id file.
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
        slurm_out_path: str | Path,
        slurm_args: dict,
        *,
        local: bool = False,
        cluster: str | None = None,
        sbatch_filepath: str | Path | None = None,
    ) -> None:
        self.slurm_out_path = Path(slurm_out_path)

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


def submit_jobs(
    func: Callable,
    job_args: Iterable[JobArgs],
    *,
    slurm_out_path: str | Path,
    slurm_args: dict,
    finalize: str = "monitor",
    monitor_paths: list | None = None,
    local: bool = False,
    cluster: str | None = None,
    sbatch_filepath: str | Path | None = None,
) -> SlurmExecutor:
    """
    Submit one batch of SLURM jobs and finalize, in a single call.

    Function-agnostic convenience wrapper around :class:`SlurmExecutor`: it sets up
    the executor, submits ``func`` once per ``(args, kwargs)`` entry in ``job_args``,
    and finalizes. ``func`` and the fan-out axis are entirely up to the caller — the
    same call serves one-job-per-FOV image processing
    (``submit_jobs(process_single_position, [...])``) and per-timepoint estimators
    (``submit_jobs(estimate_tzyx, [...], finalize="wait")``) alike.

    Modules that submit multiple batches (e.g. ``register``, ``stabilize``) or
    dependency chains (e.g. ``virtual_stain``) should use :class:`SlurmExecutor`
    directly, calling :meth:`SlurmExecutor.submit_batch` more than once.

    Parameters
    ----------
    func : Callable
        The callable to submit for every job. The same callable is used for the
        whole batch; per-job variation lives in ``job_args``.
    job_args : Iterable[JobArgs]
        One ``(args, kwargs)`` tuple per job; each job runs ``func(*args, **kwargs)``.
        Build this however the work fans out — over FOV, timepoints, channels, etc.
    slurm_out_path : Union[str, Path]
        Directory for submitit logs and the job-id file.
    slurm_args : dict
        Parameters for ``executor.update_parameters``.
    finalize : {"monitor", "wait", "none"}, optional
        How to finalize after submission: ``"monitor"`` for live progress (image
        modules), ``"wait"`` to block silently until done (estimators), or
        ``"none"`` to return immediately. Default ``"monitor"``.
    monitor_paths : list, optional
        Paths passed to ``monitor_jobs`` for progress labels when
        ``finalize="monitor"`` (typically the input position dirpaths).
    local : bool, optional
        Run locally instead of submitting to SLURM. Default False.
    cluster : str, optional
        Pre-resolved submitit cluster backend; takes precedence over ``local``.
    sbatch_filepath : Union[str, Path], optional
        Optional sbatch file whose parsed contents override ``slurm_args``.

    Returns
    -------
    SlurmExecutor
        The executor used, with ``.jobs`` populated, so callers can inspect or
        further wait on the submitted jobs.
    """
    if finalize not in ("monitor", "wait", "none"):
        raise ValueError(f"finalize must be 'monitor', 'wait', or 'none', got {finalize!r}")

    executor = SlurmExecutor(
        slurm_out_path,
        slurm_args,
        local=local,
        cluster=cluster,
        sbatch_filepath=sbatch_filepath,
    )
    executor.submit_batch(func, job_args)

    if finalize == "monitor":
        executor.monitor(monitor_paths)
    elif finalize == "wait":
        executor.wait()
    return executor
