"""Transfer-function construction and node-local cache ownership."""

import hashlib
import os
import time

from pathlib import Path

from biahub.tile_stitch import _nvtx


def _host_arrays(settings, modality: str) -> dict:
    """Return host transfer arrays through a process-shared disk cache."""
    import numpy as np

    from waveorder.api.tile_stitch import prepare_transfer_function

    keys = (
        ("optical_transfer_function",)
        if modality == "fluorescence"
        else ("real_potential_transfer_function", "imaginary_potential_transfer_function")
    )

    def build() -> dict:
        transfer = prepare_transfer_function(settings, device=None)
        return {key: np.asarray(transfer[key].values) for key in keys}

    if os.environ.get("TILESTITCH_TF_DISK_CACHE", "1") != "1":
        return build()

    digest = hashlib.sha1(settings.model_dump_json().encode()).hexdigest()[:16]
    root = Path(
        os.environ.get("TILESTITCH_TF_CACHE")
        or os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    )
    cache = root / "tilestitch_tf"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{digest}.npz"
    lock = cache / f"{digest}.lock"
    deadline = time.monotonic() + 1800

    while True:
        if archive.exists():
            try:
                with np.load(archive) as data:
                    return {key: data[key] for key in keys}
            except Exception:
                time.sleep(2)
                continue
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            if time.monotonic() > deadline:
                return build()
            time.sleep(2)
            continue
        try:
            arrays = build()
            temporary = cache / f"{digest}.tmp{os.getpid()}.npz"
            np.savez(temporary, **arrays)
            os.replace(temporary, archive)
            return arrays
        finally:
            lock.unlink(missing_ok=True)


def get_transfer_tensors(settings, device: str) -> tuple[dict, object]:
    """Return transfer tensors and modality settings for one CUDA device."""
    import torch

    from waveorder.api.tile_stitch import (
        prepare_transfer_function_tensors,
        select_recon_modality,
    )

    modality, modality_settings = select_recon_modality(settings.recon)
    tensors = None
    if os.environ.get("TILESTITCH_GPU_OPTICS", "1") == "1":
        try:
            with _nvtx.stage("tf_build_device", "magenta"):
                tensors = prepare_transfer_function_tensors(settings, device=device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()

    if tensors is None:
        with _nvtx.stage("tf_build_host", "magenta"):
            arrays = _host_arrays(settings, modality)
            tensors = {
                key: torch.as_tensor(value, device=device) for key, value in arrays.items()
            }
    return tensors, modality_settings
