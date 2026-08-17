"""Helpers for running cellpose on the cluster.

Two problems this module exists to solve, both observed in one overnight run:

* cellpose resolves its own device in ``cellpose.core.assign_device``, which probes
  CUDA inside a bare ``except: pass`` and quietly returns the CPU whenever the probe
  fails. The CPU is ~130x slower, and nothing says which device was used.
* A cellpose 4 checkpoint is ~1.2 GB and lives in the user's home directory, which
  is an NFS export on Bruno. A per-position fan-out reads that one file from dozens
  of nodes at once.

``cellpose_device`` addresses the first, ``warm_cellpose_weights`` (once, on the
head node) and ``stage_cellpose_weights`` (per worker) the second.
"""

import getpass
import logging
import os
import shutil
import sys
import time

from pathlib import Path

import click
import torch

logger = logging.getLogger(__name__)

# Attempts for each node-local cellpose weight copy. ESTALE on the NFS home export
# is transient and is the very thing staging exists to avoid, so retry the copy
# before giving up and reading the weights over NFS. See stage_cellpose_weights.
_STAGE_ATTEMPTS = 3


def cellpose_device(gpu: bool) -> torch.device:
    """Resolve the torch device for cellpose, refusing a silent CPU fallback.

    Left to itself, cellpose picks its device in ``cellpose.core.assign_device``,
    which probes CUDA inside a bare ``except: pass`` and quietly returns the CPU
    whenever the probe fails — a transient CUDA init error on a busy node included.
    Nothing in the logs distinguishes the two devices, and on an A6000 against a
    1664x1193 frame the CPU is 130x slower (2.0 s vs 264 s per frame), so a
    three-minute task silently runs for hours. Probing here surfaces the real CUDA
    error and fails the task in seconds instead, which lets the caller (or the
    workflow) retry on a fresh allocation.

    Parameters
    ----------
    gpu : bool
        Whether GPU segmentation was requested.

    Returns
    -------
    torch.device
        ``cuda:0`` — the GPU SLURM allocated, via CUDA_VISIBLE_DEVICES — when
        ``gpu`` is set, ``cpu`` otherwise.

    Raises
    ------
    RuntimeError
        If ``gpu`` is set but CUDA cannot be used on this host.
    """
    if not gpu:
        return torch.device("cpu")

    device = torch.device("cuda:0")
    try:
        torch.zeros(1).to(device)
    except Exception as exc:
        raise RuntimeError(
            "GPU segmentation was requested but CUDA is unusable on this host "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}): {exc}. "
            "Refusing to fall back to the CPU, which is ~130x slower for cellpose. "
            "Ask for the CPU explicitly to segment on it deliberately "
            "(tracking: cellpose_config.gpu: false)."
        ) from exc
    return device


def _cellpose_models_dir() -> Path:
    """Return the directory cellpose loads its weights from."""
    env_dir = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
    return Path(env_dir) if env_dir else Path.home() / ".cellpose" / "models"


def _copy_weight(source: Path, dest: Path) -> None:
    """Copy ``source`` to ``dest``, skipping a copy that is already there.

    The bytes land on a private temporary name and are renamed into place, so tasks
    staging concurrently on the same node never read a half-written file and the
    loser of the race just overwrites with identical content. Weights are matched
    on size alone: they are ~1.2 GB and immutable once downloaded, so hashing them
    on every task would cost more than the read this function avoids.
    """
    if dest.is_file() and dest.stat().st_size == source.stat().st_size:
        return

    partial = dest.with_name(f".{dest.name}.{os.getpid()}.partial")
    for attempt in range(1, _STAGE_ATTEMPTS + 1):
        try:
            shutil.copyfile(source, partial)
            break
        except OSError as exc:
            partial.unlink(missing_ok=True)
            if attempt == _STAGE_ATTEMPTS:
                raise
            logger.warning("Retrying copy of %s to node-local scratch: %r", source, exc)
            time.sleep(attempt)
    os.replace(partial, dest)


def stage_cellpose_weights() -> Path | None:
    """Cache the cellpose weights on node-local scratch and point cellpose at them.

    A cellpose 4 checkpoint is ~1.2 GB and lives under ``~/.cellpose/models``, which
    is an NFS export on Bruno. Every task ``torch.load``s it at startup, so a
    plate-wide fan-out reads that one file from dozens of nodes at once; five
    positions of a 291-position run died that way with ``OSError: [Errno 116] Stale
    file handle`` inside ``torch.load``. Copying the weights under ``$TMPDIR`` first
    reduces the burst to one sequential read per node, and every later task landing
    on the same node skips NFS entirely (``$TMPDIR`` is node-local scratch that
    outlives the job, so the cache is warm for the rest of the run).

    Call this BEFORE importing cellpose: ``cellpose.models`` reads
    CELLPOSE_LOCAL_MODELS_PATH into a module constant at import time, so a
    redirection afterwards has no effect. Weights cellpose has never downloaded
    cannot be staged either — see ``warm_cellpose_weights``.

    Returns
    -------
    Path or None
        The staging directory, or ``None`` if staging was skipped — nothing to copy,
        cellpose already imported, or the copy failed. Cellpose then reads its usual
        shared location, exactly as it did before.
    """
    source = _cellpose_models_dir()
    dest = Path(os.environ.get("TMPDIR", "/tmp")) / f"cellpose-models-{getpass.getuser()}"
    if dest == source:
        # Already staged in this process, e.g. a second position on one worker.
        return dest

    if "cellpose.models" in sys.modules:
        logger.warning(
            "cellpose was imported before staging, so its weights directory is "
            "already fixed at %s; reading them over the network instead.",
            source,
        )
        return None

    if not source.is_dir():
        return None

    try:
        weights = sorted(path for path in source.iterdir() if path.is_file())
        if not weights:
            return None
        dest.mkdir(parents=True, exist_ok=True)
        for weight in weights:
            _copy_weight(weight, dest / weight.name)
    except OSError as exc:
        logger.warning("Not staging cellpose weights from %s: %r", source, exc)
        return None

    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.fspath(dest)
    click.echo(f"Staged cellpose weights from {source} in {dest}")
    return dest


def warm_cellpose_weights() -> Path | None:
    """Ensure the shared weights cache is populated, downloading it if it is not.

    Meant for a step's ``--init``, which runs once on the head node before the
    per-position fan-out. Building the model is what makes cellpose resolve and, if
    needed, download the checkpoint it defaults to, so this leaves the shared cache
    in the state ``stage_cellpose_weights`` needs: populated, and populated once
    rather than by every node in the fan-out — or, on a cluster whose compute nodes
    have no route to the internet, populated at all. Loading the weights here also
    surfaces a corrupt download once, at init, instead of in every worker.

    Do NOT call this from a worker. It imports ``cellpose.models``, which freezes the
    directory cellpose loads weights from, leaving ``stage_cellpose_weights`` nothing
    to redirect.

    Returns
    -------
    Path or None
        The checkpoint cellpose resolved, or ``None`` if warming failed — cellpose
        missing, no network, no disk. Workers then download it themselves, as they
        did before.
    """
    models_dir = _cellpose_models_dir()
    try:
        from cellpose import models as cp_models

        # gpu=False: --init runs on the head node, which has no GPU. Only the weights
        # matter here, and they are the same either way.
        weights = Path(cp_models.CellposeModel(gpu=False).pretrained_model)
    except Exception as exc:
        logger.warning("Could not warm the cellpose weights cache in %s: %r", models_dir, exc)
        return None

    click.echo(f"Cellpose weights ready: {weights}")
    return weights
