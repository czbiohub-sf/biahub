import os

from collections.abc import Callable, Iterable
from pathlib import Path

import click

from biahub.cli.slurm import SlurmExecutor

# The execution backends ``JobRunner`` dispatches over. The kernel (``fn``) and its
# per-job call signature are identical across all three -- only the wrapping differs
# (direct call vs. submitit local subprocess vs. submitit SLURM job).
EXECUTORS = ("sequential", "local", "slurm")


class JobRunner:
    """
    Run one ``fn`` across a batch of jobs on a chosen backend.

    A job is ``fn(*item, **fn_args)``: ``fn`` and ``fn_args`` are constant across the
    whole batch, and ``items`` is the fan-out axis -- one job per element. Each element
    supplies that job's *varying* positional args (a tuple is unpacked; a scalar is
    passed as a single argument). Bind any constant *leading* positional args into
    ``fn`` with :func:`functools.partial`::

        # deskew: kernel + kwargs constant, (input, output) paths vary
        JobRunner(
            partial(process_single_position, deskew_kernel),
            zip(inputs, outputs, strict=True),
            kernel_kwargs,
            executor="slurm",
            output_dirpath=output_dirpath,
            slurm_args=slurm_args,
        ).execute(monitor_paths=inputs)

        # estimate: everything constant except the timepoint index
        JobRunner(
            estimate_tzyx, range(T), shared_kwargs, executor="slurm", output_dirpath=...
        ).execute(finalize="wait")

    ``fn`` and its call signature are *identical* across all backends; only the
    wrapping differs. No backend may substitute a different kernel or call shape, which
    makes the "two parallel implementations drift apart" bug structurally impossible.

    Backends (``executor``):

    - ``"sequential"`` -- in-process, one job at a time, fail-fast. No SLURM, no
      submitit, no monitoring. For debugging and for orchestrators that already own
      the fan-out (e.g. a Nextflow worker processing a single position).
    - ``"local"`` -- parallel subprocesses on this machine via submitit.
    - ``"slurm"`` -- parallel jobs on a SLURM cluster via submitit.

    Parameters
    ----------
    fn : Callable
        The callable run for every job. Constant across the batch.
    items : Iterable
        The fan-out axis: one job per element. Each element is that job's varying
        positional args -- a tuple is unpacked into ``fn``, anything else is passed
        as a single positional argument.
    fn_args : dict, optional
        Constant keyword arguments passed to ``fn`` on every job.
    executor : {"sequential", "local", "slurm"}, optional
        Execution backend. Default ``"slurm"``.
    output_dirpath : Union[str, Path], optional
        The run's output store (submitit backends only). submitit logs and the job-id
        file are written to a ``slurm_output`` directory beside it.
    slurm_args : dict, optional
        Parameters for ``executor.update_parameters`` (submitit backends only).
        Ignored (with a log line) for ``"sequential"``.
    sbatch_filepath : Union[str, Path], optional
        Optional sbatch file whose parsed contents override ``slurm_args``.
    """

    def __init__(
        self,
        fn: Callable,
        items: Iterable,
        fn_args: dict | None = None,
        *,
        executor: str = "slurm",
        output_dirpath: str | Path | None = None,
        slurm_args: dict | None = None,
        sbatch_filepath: str | Path | None = None,
    ) -> None:
        if executor not in EXECUTORS:
            raise ValueError(f"executor must be one of {EXECUTORS}, got {executor!r}")
        self.fn = fn
        self.fn_args = dict(fn_args or {})
        self.executor = executor
        self.output_dirpath = output_dirpath
        self.slurm_args = dict(slurm_args or {})
        self.sbatch_filepath = sbatch_filepath
        # Per-job positional args: unpack tuples, wrap scalars.
        self.job_args = [item if isinstance(item, tuple) else (item,) for item in items]

    def _execute_sequential(self, labels: list | None = None) -> list:
        """Run every job in-process, serially, fail-fast.

        The first job that raises propagates immediately, so a wrapping CLI (or a
        Nextflow worker invoking ``--executor sequential``) exits non-zero with a
        real traceback instead of swallowing the error.
        """
        results = []
        for i, args in enumerate(self.job_args):
            label = labels[i] if labels and i < len(labels) else i
            click.echo(f"[{i + 1}/{len(self.job_args)}] running job ({label})")
            results.append(self.fn(*args, **self.fn_args))
        return results

    def execute(
        self,
        *,
        finalize: str = "monitor",
        monitor_paths: list | None = None,
    ) -> list | SlurmExecutor:
        """
        Run all jobs on the configured backend.

        In CI (``CI=true``) there is no SLURM and subprocesses are flaky, so any
        submitit backend is transparently downgraded to ``"sequential"`` -- mirroring
        the old ``get_submitit_cluster`` "debug" behavior, but without submitit's
        lazy-execution indirection.

        Parameters
        ----------
        finalize : {"monitor", "wait", "none"}, optional
            How to finalize submitit backends: ``"monitor"`` for live progress,
            ``"wait"`` to block silently, ``"none"`` to submit and return. Ignored
            for ``"sequential"`` (which always runs to completion in-process).
            Default ``"monitor"``.
        monitor_paths : list, optional
            Per-job labels: progress labels for ``monitor_jobs`` (submitit) or for
            the sequential progress lines. Typically the input position dirpaths.

        Returns
        -------
        list | SlurmExecutor
            For ``"sequential"``, the list of per-job return values. For submitit
            backends, the :class:`SlurmExecutor` (with ``.jobs`` populated).
        """
        if finalize not in ("monitor", "wait", "none"):
            raise ValueError(
                f"finalize must be 'monitor', 'wait', or 'none', got {finalize!r}"
            )

        executor = self.executor
        # No SLURM in CI, and local subprocesses are flaky there: run in-process.
        if executor != "sequential" and os.environ.get("CI") == "true":
            click.echo(f"CI detected: running '{executor}' jobs sequentially in-process.")
            executor = "sequential"

        if executor == "sequential":
            if self.slurm_args:
                click.echo(
                    f"executor='sequential': ignoring SLURM parameters {sorted(self.slurm_args)}."
                )
            return self._execute_sequential(labels=monitor_paths)

        # submitit backends: "local" (this machine) or "slurm" (cluster).
        slurm_executor = SlurmExecutor(
            self.output_dirpath,
            self.slurm_args,
            cluster=executor,
            sbatch_filepath=self.sbatch_filepath,
        )
        slurm_executor.submit_batch(self.fn, [(args, self.fn_args) for args in self.job_args])

        if finalize == "monitor":
            if monitor_paths is not None:
                slurm_executor.monitor(monitor_paths)
            else:
                slurm_executor.wait()
        elif finalize == "wait":
            slurm_executor.wait()
        # "none": submit and return without blocking.
        return slurm_executor
