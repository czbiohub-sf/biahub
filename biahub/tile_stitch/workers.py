"""Stage A / Stage B worker functions.

Verbatim port of legacy v6 (CPU batched) and v7 (GPU) workers. Same
names so the driver's dispatch is trivially the legacy pattern.

CPU path:
  * ``reconstruct_batch_memory`` (Stage A): K input tiles per call, one
    waveorder ``apply_inverse_transfer_function`` invocation, returns
    dict[tile_id, ndarray].
  * ``stitch_output_tile_v6`` (Stage B): consumes batches dict → flatten
    contributors → in-place blend → zarr write.

GPU path:
  * ``reconstruct_tile_memory_gpu`` (Stage A, single tile): direct call
    to ``phase_thick_3d.apply_inverse_transfer_function`` (skips xarray
    wrapper); returns cupy.ndarray via DLPack.
  * ``stitch_output_tile_v7`` (Stage B): cupy → torch DLPack, GPU
    accumulator, single CPU bounce at zarr write.
"""

import ctypes
import gc
import json
import os
import resource
import socket
import time

from typing import Any

import numpy as np
import xarray as xr
import zarr

from biahub.tile_stitch.plan import RunPlan, load_plan

# --- helpers ---------------------------------------------------------------


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _node_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    sj = os.environ.get("SLURM_JOB_ID")
    if sj:
        info["slurm_job"] = sj
    return info


def _release_memory_to_os() -> None:
    """Force glibc to return free pages to OS — keeps unmanaged memory bounded."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _write_meta(meta_dir: str, meta: dict[str, Any]) -> None:
    if not meta_dir:
        return
    os.makedirs(meta_dir, exist_ok=True)
    fname = f"{socket.gethostname()}_{os.getpid()}.jsonl"
    with open(os.path.join(meta_dir, fname), "a") as f:
        f.write(json.dumps({**meta, "_schema_version": "1"}, default=str) + "\n")


def _resolve_recon_module(plan: RunPlan):
    """Dispatch on settings.recon.kind."""
    kind = plan.settings.recon.kind
    if kind == "phase":
        from waveorder.api import phase

        return phase
    if kind == "fluorescence":
        from waveorder.api import fluorescence

        return fluorescence
    if kind == "birefringence":
        from waveorder.api import birefringence

        return birefringence
    raise ValueError(f"unknown recon kind: {kind!r}")


# --- per-worker TF cache --------------------------------------------------

_TF_CACHE: dict[tuple, Any] = {}


def _get_tf_settings(plan: RunPlan) -> tuple[Any, Any]:
    """Build (or return cached) transfer function for this plan's settings.

    Cached per-process so K=4 batches run after the first all skip the
    ~5 s TF build.
    """
    key = (plan.settings.model_dump_json(),)
    if key in _TF_CACHE:
        return _TF_CACHE[key]
    from waveorder.api.tile_stitch import prepare_transfer_function

    tf = prepare_transfer_function(plan.settings, recon_dim=3, device=None)
    _TF_CACHE[key] = (tf, plan.settings.recon)
    return _TF_CACHE[key]


# --- CPU Stage A: batched recon (port of v6.reconstruct_batch_memory) -----


def reconstruct_batch_memory(
    batch_id: int,
    plan_path: str,
    *,
    tf_bundle: Any | None = None,
    device: str = "cpu",
    meta_dir: str = "",
) -> dict[int, np.ndarray]:
    """K input tiles → one waveorder call → dict[tile_id, ndarray]."""
    from iohub.ngff import open_ome_zarr

    t_started = time.time()
    plan = load_plan(plan_path)
    batch_tile_ids = (plan.input_batches or [])[batch_id]
    if not batch_tile_ids:
        return {}

    t0 = time.monotonic()
    if tf_bundle is not None:
        tf, recon_settings = tf_bundle
    else:
        tf, recon_settings = _get_tf_settings(plan)
    t_tf = time.monotonic() - t0

    src = open_ome_zarr(plan.input_path, layout="fov", mode="r")
    z_arr = src["0"]

    blocks: list[xr.DataArray] = []
    t0 = time.monotonic()
    for tid in batch_tile_ids:
        tile = next(t for t in plan.input_tiles if t.tile_id == tid)
        sl = (
            slice(plan.timepoint, plan.timepoint + 1),
            slice(plan.channel_idx, plan.channel_idx + 1),
        ) + tuple(tile.slices[d] for d in plan.tile_dims)
        block_np = np.asarray(z_arr[sl])
        block_xa = xr.DataArray(block_np, dims=("t", "c", *plan.tile_dims)).isel(t=0)
        blocks.append(block_xa)
    t_read = time.monotonic() - t0

    recon_module = _resolve_recon_module(plan)
    t0 = time.monotonic()
    recons = recon_module.apply_inverse_transfer_function(
        blocks,
        tf,
        recon_dim=3 if "z" in plan.tile_dims else 2,
        settings=recon_settings,
        device=device,
    )
    t_recon = time.monotonic() - t0

    out: dict[int, np.ndarray] = {}
    n = len(batch_tile_ids)
    per_tile_wall = (t_tf + t_read + t_recon) / n
    t_finished = time.time()
    node = _node_info()
    for tid, recon in zip(batch_tile_ids, recons, strict=True):
        # 5D layout (T, C, Z, Y, X) so stitch_output_tile_v6 sees the same
        # rank as the output zarr (legacy convention; n_lead=2 throughout).
        recon_np = np.asarray(recon.expand_dims("t", axis=0).values, dtype=np.float32)
        out[tid] = recon_np
        _write_meta(
            meta_dir,
            {
                "stage": "a",
                "tile_id": tid,
                "shape": list(recon_np.shape),
                "batch_id": batch_id,
                "batch_size": n,
                "t_tf_s": t_tf / n,
                "t_read_s": t_read / n,
                "t_recon_s": t_recon / n,
                "wall_s": per_tile_wall,
                "batch_wall_s": t_tf + t_read + t_recon,
                "t_started": t_started,
                "t_finished": t_finished,
                "peak_rss_mb": _peak_rss_mb(),
                **node,
            },
        )

    _release_memory_to_os()
    return out


# --- CPU Stage B: in-place blend (port of v6.stitch_output_tile_v6) -------


def stitch_output_tile_v6(
    out_tile_id: int,
    plan_path: str,
    batches: dict[int, dict[int, np.ndarray]],
) -> dict[str, Any]:
    """Flatten batches into per-tile contributors → in-place blend → write."""
    t_started = time.time()
    t_wall_start = time.monotonic()
    plan = load_plan(plan_path)
    out_tile = plan.output_tiles[out_tile_id]
    needed = set(plan.output_to_inputs[out_tile_id])

    contributors: dict[int, np.ndarray] = {}
    for batch_dict in batches.values():
        for tid, arr in batch_dict.items():
            if tid in needed:
                contributors[tid] = arr

    if not contributors:
        return {
            "stage": "b",
            "out_tile_id": out_tile_id,
            "n_inputs": 0,
            "wall_s": 0.0,
            "t_started": t_started,
            "t_finished": time.time(),
            **_node_info(),
        }

    blend_kernel = plan.settings.blend.build()
    tiles_by_id = {t.tile_id: t for t in plan.input_tiles}
    out_arr = zarr.open_group(plan.output_path, mode="a")["0"]
    n_lead = len(plan.leading_shape)

    out_spatial = [(out_tile.slices[d].start, out_tile.slices[d].stop) for d in plan.tile_dims]
    out_shape = plan.leading_shape + tuple(hi - lo for lo, hi in out_spatial)

    sample_dtype = next(iter(contributors.values())).dtype
    accum_v = np.zeros(out_shape, dtype=sample_dtype)
    accum_w = np.zeros(out_shape, dtype=sample_dtype)

    t_blend_total = 0.0
    for tid, tile_full in contributors.items():
        in_tile = tiles_by_id[tid]

        in_local: list[slice] = []
        out_local: list[slice] = []
        for i_d, d in enumerate(plan.tile_dims):
            in_lo = in_tile.slices[d].start
            in_hi = in_tile.slices[d].stop
            ot_lo, ot_hi = out_spatial[i_d]
            isect_lo = max(in_lo, ot_lo)
            isect_hi = min(in_hi, ot_hi)
            in_local.append(slice(isect_lo - in_lo, isect_hi - in_lo))
            out_local.append(slice(isect_lo - ot_lo, isect_hi - ot_lo))

        in_lead = (slice(None),) * n_lead
        in_full_idx = in_lead + tuple(in_local)
        out_full_idx = in_lead + tuple(out_local)

        t = time.monotonic()
        kernel_full = blend_kernel.weight_kernel(in_tile.shape).astype(
            sample_dtype, copy=False
        )
        kernel_view = kernel_full[tuple(in_local)]
        v_view = tile_full[in_full_idx]
        if v_view.dtype != sample_dtype:
            v_view = v_view.astype(sample_dtype, copy=False)
        accum_v[out_full_idx] += v_view * kernel_view
        accum_w[out_full_idx] += kernel_view
        t_blend_total += time.monotonic() - t

    t = time.monotonic()
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(accum_w > 0, accum_v / accum_w, blend_kernel.fill_value).astype(
            np.float32, copy=False
        )
    t_blend_total += time.monotonic() - t

    t = time.monotonic()
    write_region = tuple(slice(0, plan.leading_shape[i]) for i in range(n_lead)) + tuple(
        slice(lo, hi) for lo, hi in out_spatial
    )
    out_arr[write_region] = result
    t_write = time.monotonic() - t

    summary = {
        "stage": "b",
        "out_tile_id": out_tile_id,
        "n_inputs": len(contributors),
        "n_batches": len(batches),
        "t_blend_s": t_blend_total,
        "t_write_s": t_write,
        "wall_s": time.monotonic() - t_wall_start,
        "t_started": t_started,
        "t_finished": time.time(),
        "peak_rss_mb": _peak_rss_mb(),
        **_node_info(),
    }
    _release_memory_to_os()
    return summary


# --- GPU Stage A (port of v7.reconstruct_tile_memory_gpu) -----------------


_TF_CUDA_CACHE: dict[tuple, Any] = {}


def _get_tf_cuda(plan: RunPlan) -> tuple[dict[str, Any], Any]:
    """Build (or fetch cached) CUDA-resident TF tensors for phase recon."""
    import torch

    key = (plan.settings.model_dump_json(),)
    if key in _TF_CUDA_CACHE:
        return _TF_CUDA_CACHE[key]
    from waveorder.api.tile_stitch import prepare_transfer_function

    tf = prepare_transfer_function(plan.settings, recon_dim=3, device=None)
    cuda_tf = {
        "real_potential_transfer_function": torch.as_tensor(
            tf["real_potential_transfer_function"].values, device="cuda"
        ),
        "imaginary_potential_transfer_function": torch.as_tensor(
            tf["imaginary_potential_transfer_function"].values, device="cuda"
        ),
    }
    _TF_CUDA_CACHE[key] = (cuda_tf, plan.settings.recon)
    return _TF_CUDA_CACHE[key]


def reconstruct_tile_memory_gpu(
    tile_id: int,
    plan_path: str,
    *,
    meta_dir: str = "",
):
    """v7 Stage A — phase recon on GPU, returns cupy.ndarray (DLPack)."""
    import cupy
    import torch

    from iohub.ngff import open_ome_zarr
    from waveorder.models import phase_thick_3d

    t_started = time.time()
    plan = load_plan(plan_path)
    tile = next(t for t in plan.input_tiles if t.tile_id == tile_id)

    t0 = time.monotonic()
    cuda_tf, recon_settings = _get_tf_cuda(plan)
    t_tf = time.monotonic() - t0

    t0 = time.monotonic()
    src = open_ome_zarr(plan.input_path, layout="fov", mode="r")
    z_arr = src["0"]
    sl = (
        slice(plan.timepoint, plan.timepoint + 1),
        slice(plan.channel_idx, plan.channel_idx + 1),
    ) + tuple(tile.slices[d] for d in plan.tile_dims)
    block_np = np.asarray(z_arr[sl])
    zyx = torch.as_tensor(block_np[0, 0], dtype=torch.float32, device="cuda")
    t_read = time.monotonic() - t0

    t0 = time.monotonic()
    recon = phase_thick_3d.apply_inverse_transfer_function(
        zyx,
        cuda_tf["real_potential_transfer_function"],
        cuda_tf["imaginary_potential_transfer_function"],
        z_padding=recon_settings.transfer_function.z_padding,
        **recon_settings.apply_inverse.model_dump(),
    )
    recon = recon.unsqueeze(0).to(torch.float32).contiguous()
    t_recon = time.monotonic() - t0

    out_cupy = cupy.from_dlpack(recon.detach())

    _write_meta(
        meta_dir,
        {
            "stage": "a-gpu",
            "tile_id": tile_id,
            "shape": list(out_cupy.shape),
            "t_tf_s": t_tf,
            "t_read_s": t_read,
            "t_recon_s": t_recon,
            "wall_s": t_tf + t_read + t_recon,
            "t_started": t_started,
            "t_finished": time.time(),
            **_node_info(),
        },
    )
    return out_cupy


# --- GPU Stage B (port of v7.stitch_output_tile_v7) -----------------------


def stitch_output_tile_v7(
    out_tile_id: int,
    plan_path: str,
    contributors: dict[int, Any],  # cupy ndarrays
) -> dict[str, Any]:
    """v7 Stage B — GPU accumulator, single CPU bounce at zarr write."""
    import torch

    t_started = time.time()
    t_wall_start = time.monotonic()
    plan = load_plan(plan_path)
    out_tile = plan.output_tiles[out_tile_id]

    if not contributors:
        return {
            "stage": "b-gpu",
            "out_tile_id": out_tile_id,
            "n_inputs": 0,
            "wall_s": 0.0,
            "t_started": t_started,
            "t_finished": time.time(),
            **_node_info(),
        }

    torch_contribs: dict[int, Any] = {
        tid: torch.from_dlpack(arr.toDlpack()) for tid, arr in contributors.items()
    }
    sample = next(iter(torch_contribs.values()))
    blend_kernel = plan.settings.blend.build()
    tiles_by_id = {t.tile_id: t for t in plan.input_tiles}

    out_spatial = [(out_tile.slices[d].start, out_tile.slices[d].stop) for d in plan.tile_dims]
    out_shape = plan.leading_shape[1:] + tuple(hi - lo for lo, hi in out_spatial)
    accum_v = torch.zeros(out_shape, dtype=sample.dtype, device="cuda")
    accum_w = torch.zeros(out_shape, dtype=sample.dtype, device="cuda")

    t_blend_total = 0.0
    for tid, tile_full in torch_contribs.items():
        in_tile = tiles_by_id[tid]
        in_local: list[slice] = []
        out_local: list[slice] = []
        for d in plan.tile_dims:
            in_lo, in_hi = in_tile.slices[d].start, in_tile.slices[d].stop
            ot_lo, ot_hi = out_tile.slices[d].start, out_tile.slices[d].stop
            isect_lo = max(in_lo, ot_lo)
            isect_hi = min(in_hi, ot_hi)
            if isect_hi <= isect_lo:
                continue
            in_local.append(slice(isect_lo - in_lo, isect_hi - in_lo))
            out_local.append(slice(isect_lo - ot_lo, isect_hi - ot_lo))
        if len(in_local) != len(plan.tile_dims):
            continue

        # tile_full shape: (C, Z, Y, X). out_shape uses leading_shape[1:] = (C,).
        in_full_idx = (slice(None),) * (len(plan.leading_shape) - 1) + tuple(in_local)
        out_full_idx = (slice(None),) * (len(plan.leading_shape) - 1) + tuple(out_local)

        t = time.monotonic()
        kernel_np = blend_kernel.weight_kernel(in_tile.shape).astype(np.float32, copy=False)
        kernel_full = torch.as_tensor(kernel_np, dtype=sample.dtype, device="cuda")
        kernel_view = kernel_full[tuple(in_local)]
        v_view = tile_full[in_full_idx]
        accum_v[out_full_idx] += v_view * kernel_view
        accum_w[out_full_idx] += kernel_view
        t_blend_total += time.monotonic() - t

    t = time.monotonic()
    fill = torch.tensor(blend_kernel.fill_value, dtype=sample.dtype, device="cuda")
    result_gpu = torch.where(accum_w > 0, accum_v / accum_w, fill).to(torch.float32)
    result_np = result_gpu.cpu().numpy()
    t_finalize = time.monotonic() - t

    t = time.monotonic()
    out_arr = zarr.open_group(plan.output_path, mode="a")["0"]
    write_region = tuple(slice(0, n) for n in plan.leading_shape) + tuple(
        slice(lo, hi) for lo, hi in out_spatial
    )
    # result_np is (C, Z, Y, X); zarr is (T, C, Z, Y, X). Unsqueeze T.
    out_arr[write_region] = result_np[None]
    t_write = time.monotonic() - t

    return {
        "stage": "b-gpu",
        "out_tile_id": out_tile_id,
        "n_inputs": len(contributors),
        "t_blend_s": t_blend_total,
        "t_finalize_s": t_finalize,
        "t_write_s": t_write,
        "wall_s": time.monotonic() - t_wall_start,
        "t_started": t_started,
        "t_finished": time.time(),
        **_node_info(),
    }
