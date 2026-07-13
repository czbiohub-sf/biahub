"""Tile-stitch compute helpers: pure functions + small caches.

No Monarch imports, so this layer unit-tests without a mesh; the actor is a thin
transport shell over it.

Leading-axis convention: geometry strips the T axis (``leading_shape[1:]``) and
the write region carries T via an explicit ``slice(t_off, t_off + 1)``.
"""

import concurrent.futures
import logging
import os
import threading
import time

from collections.abc import Callable
from functools import cache
from typing import Any

from biahub.tile_stitch import _nvtx

# ONE module-level TF cache, shared across all actors in a process. The key
# includes the gpu_idx (via the device string) so multi-actor-in-one-process
# (or process reuse across runs) never returns cuda:0 tensors to a cuda:1
# caller (tile_worker.py:451,462-467).
_TF_CUDA_CACHE: dict[tuple, Any] = {}


@cache
def _open_input_array(input_path: str):
    """Level-0 array of the input FOV, opened once per process and cached.

    Re-opening per tile read does a fresh NFS metadata handshake each time —
    slow over the shared store, and it widens the window for a transient stall to
    wedge the (timeout-less) prefetch reader. zarr reads are thread-safe, so one
    cached handle is reused by the background reader and the synchronous fallback.
    """
    from iohub.ngff import open_ome_zarr

    return open_ome_zarr(input_path, layout="fov", mode="r")["0"]


# A single tile read that stalls on the filesystem would otherwise wedge the whole
# pipeline (nothing downstream has a deadline). Each read runs on a pool thread with
# a timeout; a stalled attempt is abandoned and retried on a fresh thread, and a
# persistent stall raises (fails loudly) instead of hanging forever.
_READ_TIMEOUT_S = float(os.environ.get("TILE_READ_TIMEOUT_S", "120"))
_READ_RETRIES = int(os.environ.get("TILE_READ_RETRIES", "2"))
_GET_TIMEOUT_S = float(os.environ.get("TILE_GET_TIMEOUT_S", "600"))

logger = logging.getLogger("tile_stitch._core")

_read_pool: concurrent.futures.ThreadPoolExecutor | None = None
_read_pool_lock = threading.Lock()


def _get_read_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _read_pool
    if _read_pool is None:
        with _read_pool_lock:
            if _read_pool is None:
                _read_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="zarr-read"
                )
    return _read_pool


def _read_with_retry(fn, *, timeout: float = _READ_TIMEOUT_S, retries: int = _READ_RETRIES):
    """Run a blocking read with a deadline, retrying a stalled attempt.

    Runs on a fresh thread. A read that raises propagates immediately (a real
    error, not a stall); a read that never returns is abandoned after ``timeout``
    and retried; a persistent stall raises ``TimeoutError`` after ``retries`` so
    the caller fails loudly rather than wedging the pipeline.
    """
    last_exc: BaseException | None = None
    for _ in range(retries + 1):
        fut = _get_read_pool().submit(fn)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            last_exc = exc  # abandon the stuck read; retry on a fresh thread
    raise TimeoutError(
        f"tile read stalled: no progress in {retries + 1} attempts of {timeout:.0f}s"
    ) from last_exc


class PrefetchReader:
    """Background zarr reader: read tile N+1 while the GPU works on tile N.

    Stage A is IO-bound (zarr read ≫ FFT on our datasets), and the read and
    the FFT run on different threads — the recon thread spends most of its
    time inside ``cuda.synchronize`` (GIL released) and zarr decompression
    releases the GIL too. So a single background reader pulling the actor's
    *next* assigned tile from the shared filesystem overlaps with the
    current tile's compute, hiding the FFT behind the read.

    The reader walks ``order`` (the actor's assigned tile sequence) and
    stays at most ``depth`` tiles ahead of the consumer. ``get(tile_id)``
    blocks until that tile's bytes are ready. The design is deadlock-safe
    under out-of-order consumption: the reader's look-ahead gate tracks the
    consumer's index (never the buffers it has produced), so a consumer that
    jumps ahead simply pulls the reader forward rather than wedging it; a
    tile the reader genuinely failed to read returns ``None`` so the caller
    falls back to a synchronous read.
    """

    def __init__(self, load_fn: Callable[[int], Any], order: list[int], depth: int):
        self._load_fn = load_fn
        self._order = order
        self._index = {tid: i for i, tid in enumerate(order)}
        self._depth = max(1, depth)
        self._buffers: dict[int, Any] = {}
        self._consume_idx = 0  # index of the next tile the consumer wants
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="prefetch-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        for i, tid in enumerate(self._order):
            with self._cv:
                # Stay within ``depth`` of the consumer's position so memory
                # is bounded and we don't read tiles that won't be used soon.
                while not self._stop and i >= self._consume_idx + self._depth:
                    self._cv.wait()
                if self._stop:
                    return
            try:
                with _nvtx.stage("zarr_read", "blue"):
                    buf = self._load_fn(tid)
                if buf is not None:
                    _nvtx.counter("bytes_read", unit="bytes").sample(int(getattr(buf, "nbytes", 0)))
            except Exception as exc:
                # Consumer falls back to a synchronous read; log it so a non-stall
                # failure (corrupt chunk / shape mismatch) is not invisible until it
                # re-fails synchronously later.
                logger.warning("prefetch read failed for tile %s: %s", tid, exc)
                buf = None
            with self._cv:
                self._buffers[tid] = buf
                self._cv.notify_all()

    def get(self, tile_id: int) -> Any:
        """Return the pre-read array for ``tile_id`` (waits if not yet read).

        Returns ``None`` if the tile is not in this reader's assignment or
        the background read failed — the caller reads it synchronously.
        """
        i = self._index.get(tile_id)
        if i is None:
            return None
        with self._cv:
            # Advance the consume point and wake the reader to extend its
            # look-ahead window; drop any buffers strictly behind us.
            if i + 1 > self._consume_idx:
                self._consume_idx = i + 1
                self._cv.notify_all()
            for stale in [k for k in self._buffers if self._index[k] < i]:
                self._buffers.pop(stale, None)
            deadline = time.monotonic() + _GET_TIMEOUT_S
            while tile_id not in self._buffers and not self._stop:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Reader wedged past the deadline. A silent multi-minute wait is
                    # invisible; log it AND mark the reader dead so subsequent get()s
                    # short-circuit to the synchronous read immediately instead of
                    # each re-waiting the full deadline (serialising the actor behind
                    # repeated long waits).
                    logger.warning(
                        "prefetch reader wedged: tile %s not ready after %.0fs; "
                        "disabling prefetch, falling back to synchronous reads",
                        tile_id,
                        _GET_TIMEOUT_S,
                    )
                    self._stop = True
                    self._cv.notify_all()
                    return None
                self._cv.wait(timeout=remaining)
            return self._buffers.pop(tile_id, None)

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=5)
        with self._cv:
            self._buffers.clear()


def build_stitch_geom(plan) -> dict[int, dict]:
    """Pre-compute per-output-tile intersection slices and shapes.

    Tile geometry (input tile slices, output tile slices, leading
    dims) is constant for the actor's lifetime: TP-to-TP swaps only
    change pixel values, not the grid. Pre-building this per
    ``out_tile_id`` eliminates O(N) Python work in the hot blend loop.

    Uses ``leading_shape[1:]`` — the T axis is stripped here and carried
    instead by an explicit ``slice(t_off, t_off + 1)`` at write time.
    """
    tiles_by_id = {t.tile_id: t for t in plan.input_tiles}
    geom: dict[int, dict] = {}
    leading = plan.leading_shape[1:]
    n_lead = len(leading)
    in_lead = (slice(None),) * n_lead
    for out_tile in plan.output_tiles:
        oid = out_tile.tile_id
        out_spatial = [
            (out_tile.slices[d].start, out_tile.slices[d].stop) for d in plan.tile_dims
        ]
        out_shape = leading + tuple(hi - lo for lo, hi in out_spatial)
        contrib_geom: dict[int, dict] = {}
        for tid in plan.output_to_inputs.get(oid, ()):
            in_tile = tiles_by_id[tid]
            in_local: list[slice] = []
            out_local: list[slice] = []
            skip = False
            for d_idx, d in enumerate(plan.tile_dims):
                in_lo, in_hi = (
                    in_tile.slices[d].start,
                    in_tile.slices[d].stop,
                )
                ot_lo, ot_hi = out_spatial[d_idx]
                isect_lo = max(in_lo, ot_lo)
                isect_hi = min(in_hi, ot_hi)
                if isect_hi <= isect_lo:
                    skip = True
                    break
                in_local.append(slice(isect_lo - in_lo, isect_hi - in_lo))
                out_local.append(slice(isect_lo - ot_lo, isect_hi - ot_lo))
            if skip or len(in_local) != len(plan.tile_dims):
                continue
            contrib_geom[tid] = {
                "tile_shape": tuple(in_tile.shape),
                "in_local": tuple(in_local),
                "in_full_idx": in_lead + tuple(in_local),
                "out_full_idx": in_lead + tuple(out_local),
            }
        geom[oid] = {
            "out_spatial": out_spatial,
            "out_shape": out_shape,
            "n_lead": n_lead,
            "contributors": contrib_geom,
        }
    return geom


def _gpu_shared_optics(device):
    """Context manager: build waveorder's phase shared optics on ``device`` (GPU).

    Replaces ``phase_thick_3d._compute_shared_optics`` (which builds the large
    ``(up_Z, up_Y, up_X)`` propagation-kernel + Green's-function arrays on CPU,
    then the caller moves them to device) with a version that builds them
    directly on ``device``. Keeps the transfer-function build off host RAM and on
    the GPU (waveorder PR #562 approach), turning a ~300 s / ~340 GB-host build
    into a sub-second GPU build for tiles whose upsampled optics fit VRAM.
    """
    import contextlib

    import torch

    from waveorder import optics, util
    from waveorder.models import phase_thick_3d as _p3

    @contextlib.contextmanager
    def _cm():
        orig = _p3._compute_shared_optics

        def patched(zyx_shape, yx_pixel_size, z_pixel_size, wavelength_illumination,
                    z_padding, index_of_refraction_media, numerical_aperture_detection,
                    invert_phase_contrast=False, pupil_steepness=1e4):
            fyy, fxx = util.generate_frequencies(zyx_shape[1:], yx_pixel_size)
            fyy, fxx = fyy.to(device), fxx.to(device)
            radial = torch.sqrt(fyy**2 + fxx**2)
            z_total = zyx_shape[0] + 2 * z_padding
            z_pos = torch.fft.ifftshift(
                (torch.arange(z_total, device=device) - z_total // 2) * z_pixel_size
            )
            if invert_phase_contrast:
                z_pos = torch.flip(z_pos, dims=(0,))
            if torch.is_tensor(numerical_aperture_detection):
                numerical_aperture_detection = numerical_aperture_detection.to(device)
            det = optics.generate_pupil(
                radial, numerical_aperture_detection, wavelength_illumination, steepness=pupil_steepness
            )
            wl_media = wavelength_illumination / index_of_refraction_media
            prop = optics.generate_propagation_kernel(radial, det, wl_media, z_pos)
            greens = optics.generate_greens_function_z(radial, det, wl_media, z_pos, axially_even=False)
            return fyy, fxx, det, prop, greens

        _p3._compute_shared_optics = patched
        try:
            yield
        finally:
            _p3._compute_shared_optics = orig

    return _cm()


def _build_phase_tf_gpu(settings, phase_settings, device):
    """Build the 3D phase transfer function directly on ``device`` (GPU tensors).

    Mirrors ``waveorder.api.phase.compute_transfer_function`` (recon_dim=3) but
    passes GPU tilt tensors so ``calculate_transfer_function`` runs on the GPU
    (it infers ``device = zen.device``) with the shared optics also built on-GPU.
    Returns GPU tensors directly (no CPU round-trip). Propagates CUDA OOM so the
    caller can fall back to the CPU build.
    """
    import torch

    from waveorder.models import phase_thick_3d

    s = phase_settings.transfer_function.resolve_floats()
    ts = settings.tile.tile_size
    zyx_shape = (int(ts["z"]), int(ts["y"]), int(ts["x"]))
    zen = torch.as_tensor(float(s.tilt_angle_zenith), dtype=torch.float32, device=device)
    azi = torch.as_tensor(float(s.tilt_angle_azimuth), dtype=torch.float32, device=device)
    with _gpu_shared_optics(device):
        real_tf, imag_tf = phase_thick_3d.calculate_transfer_function(
            zyx_shape=zyx_shape,
            yx_pixel_size=s.yx_pixel_size,
            z_pixel_size=s.z_pixel_size,
            wavelength_illumination=s.wavelength_illumination,
            z_padding=s.z_padding,
            index_of_refraction_media=s.index_of_refraction_media,
            numerical_aperture_illumination=s.numerical_aperture_illumination,
            numerical_aperture_detection=s.numerical_aperture_detection,
            invert_phase_contrast=s.invert_phase_contrast,
            tilt_angle_zenith=zen,
            tilt_angle_azimuth=azi,
        )
    return {
        "real_potential_transfer_function": real_tf.contiguous(),
        "imaginary_potential_transfer_function": imag_tf.contiguous(),
    }


def _cpu_tf_arrays(settings, modality):
    """Build the CPU transfer function once per node (disk cache + lock).

    Actors are separate processes, so without sharing each rebuilds the identical
    TF and multiplies the host-RAM peak (the fullZ-YX512 OOM cause). This builds
    it once per node and shares the downsampled result via a node-local ``.npz``;
    later actors block on an ``O_EXCL`` lock then load. Returns numpy arrays.
    """
    import hashlib
    import os
    import pathlib
    import time

    import numpy as np

    from waveorder.api.tile_stitch import prepare_transfer_function

    keys = (
        ["optical_transfer_function"]
        if modality == "fluorescence"
        else ["real_potential_transfer_function", "imaginary_potential_transfer_function"]
    )

    def _build():
        tf = prepare_transfer_function(settings, device=None)
        return {k: np.asarray(tf[k].values) for k in keys}

    if os.environ.get("TILESTITCH_TF_DISK_CACHE", "1") != "1":
        return _build()

    h = hashlib.sha1(settings.model_dump_json().encode()).hexdigest()[:16]
    root = (
        os.environ.get("TILESTITCH_TF_CACHE")
        or os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    )
    cache = pathlib.Path(root) / "tilestitch_tf"
    cache.mkdir(parents=True, exist_ok=True)
    npz, lock = cache / f"{h}.npz", cache / f"{h}.lock"
    deadline = time.monotonic() + 1800
    while True:
        if npz.exists():
            try:
                data = np.load(npz)
                return {k: data[k] for k in keys}
            except Exception:  # partially written; retry
                time.sleep(2)
                continue
        try:
            os.close(os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            if time.monotonic() > deadline:  # builder likely died -> local build
                return _build()
            time.sleep(2)
            continue
        try:
            arrs = _build()
            tmp = cache / f"{h}.tmp{os.getpid()}.npz"  # .npz suffix: np.savez won't re-append
            np.savez(tmp, **arrs)
            os.replace(tmp, npz)
            return arrs
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


def get_tf_cuda(settings, device: str) -> tuple[dict[str, Any], Any]:
    """Build (or fetch cached) CUDA transfer-function tensors.

    Per-process cache keyed on ``(settings, device)`` — the device is load-bearing
    for multi-actor-in-one-process correctness (``torch.cuda.set_device`` is
    thread-local). For 3D phase the TF is built directly on the GPU (fast, no
    host-RAM balloon); on CUDA OOM it falls back to a CPU build shared once per
    node. Fluorescence / 2D use the CPU path.
    """
    import os

    import torch

    key = (settings.model_dump_json(), device)
    if key in _TF_CUDA_CACHE:
        return _TF_CUDA_CACHE[key]

    from waveorder.api.tile_stitch import select_recon_modality

    modality, modality_settings = select_recon_modality(settings.recon)
    recon_dim = settings.recon.reconstruction_dimension

    cuda_tf = None
    if modality == "phase" and recon_dim == 3 and os.environ.get("TILESTITCH_GPU_OPTICS", "1") == "1":
        try:
            with _nvtx.stage("tf_build_gpu", "magenta"):
                cuda_tf = _build_phase_tf_gpu(settings, modality_settings, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            cuda_tf = None  # fall back to the CPU build

    if cuda_tf is None:
        with _nvtx.stage("tf_build_cpu", "magenta"):
            arrs = _cpu_tf_arrays(settings, modality)
            cuda_tf = {k: torch.as_tensor(v, device=device) for k, v in arrs.items()}

    _TF_CUDA_CACHE[key] = (cuda_tf, modality_settings)
    return _TF_CUDA_CACHE[key]


def make_eager_recon(cuda_tf, recon_settings) -> Callable:
    """Return the bare eager ``zyx -> recon`` closure for the active modality.

    Binds the TF tensors + ``apply_inverse`` kwargs into a closure so a
    downstream ``torch.compile`` graph only sees a single dynamic input.
    ``torch.compile`` wrapping (and the compiled-callable cache) stays in the
    actor — this module only provides the pure eager closure both the eager
    and compiled paths build on.

    Modality is inferred from ``cuda_tf`` (built by :func:`get_tf_cuda`):
    ``optical_transfer_function`` -> fluorescence (thick 3D OTF deconvolution),
    otherwise the phase real+imaginary potential TF. The
    ``apply_inverse.model_dump()`` kwargs (reconstruction_algorithm,
    regularization_strength, TV_*) line up with both model signatures.
    """
    z_padding = recon_settings.transfer_function.z_padding
    apply_kwargs = recon_settings.apply_inverse.model_dump()

    if "optical_transfer_function" in cuda_tf:
        from waveorder.models import isotropic_fluorescent_thick_3d as fluor

        otf = cuda_tf["optical_transfer_function"]

        def _eager(zyx):
            return fluor.apply_inverse_transfer_function(
                zyx,
                otf,
                z_padding=z_padding,
                **apply_kwargs,
            )

        return _eager

    from waveorder.models import phase_thick_3d

    tf_real = cuda_tf["real_potential_transfer_function"]
    tf_imag = cuda_tf["imaginary_potential_transfer_function"]

    def _eager(zyx):
        return phase_thick_3d.apply_inverse_transfer_function(
            zyx,
            tf_real,
            tf_imag,
            z_padding=z_padding,
            **apply_kwargs,
        )

    return _eager


def read_tile_block(plan, tile):
    """Read one input tile as a ``(Z, Y, X)`` array.

    The leaf zarr read used by the actor's per-TP ``PrefetchReader`` loader
    closure. Returns the raw source dtype (typically uint16) as a numpy array;
    the recon path casts to float32 during the H2D copy.
    """
    import numpy as np

    z_arr = _open_input_array(plan.input_path)  # opened once, reused per tile
    sl = (
        slice(plan.timepoint, plan.timepoint + 1),
        slice(plan.channel_idx, plan.channel_idx + 1),
    ) + tuple(tile.slices[d] for d in plan.tile_dims)
    return _read_with_retry(lambda: np.asarray(z_arr[sl]).squeeze(axis=(0, 1)))  # drop T, C


def load_tile_zyx(plan, tile, *, volume=None, device: str):
    """Load one input tile to a ``(Z, Y, X)`` float32 tensor on ``device``.

    Streams from zarr when ``volume`` is ``None`` (default) or slices the
    resident ``volume`` tensor. The ``.clone()`` on the resident path
    decouples the slice so ``torch.compile`` reduce-overhead CUDA-graph
    capture sees a stable input buffer (a bare view aliases other tiles).
    """
    import torch

    if volume is None:
        import numpy as np

        z_arr = _open_input_array(plan.input_path)  # opened once, reused per tile
        sl_full = (
            slice(plan.timepoint, plan.timepoint + 1),
            slice(plan.channel_idx, plan.channel_idx + 1),
        ) + tuple(tile.slices[d] for d in plan.tile_dims)
        block_np = _read_with_retry(lambda: np.asarray(z_arr[sl_full]).squeeze(axis=(0, 1)))
        return torch.as_tensor(block_np, dtype=torch.float32, device=device)
    sl = tuple(tile.slices[d] for d in plan.tile_dims)
    return volume[sl].to(torch.float32).contiguous().clone()


def get_blend_kernel(blend, tile_shape: tuple, dtype, cache: dict):
    """Return the cached blend weight kernel for ``tile_shape`` + ``dtype``.

    ``cache`` is keyed on ``(shape, dtype.str)`` so identical inputs across
    stitches and TPs reuse the same numpy buffer instead of rebuilding it
    (each kernel is ~512 MB for a 512³ tile).
    """
    key = (tile_shape, str(dtype))
    cached = cache.get(key)
    if cached is not None:
        return cached
    kernel = blend.weight_kernel(tile_shape).astype(dtype, copy=False)
    cache[key] = kernel
    return kernel


def blend_contributors(geom_entry, contribs_np, blend, kernel_cache):
    """Weighted-mean blend of contributor tiles into one output array.

    Accumulates ``value * weight`` and ``weight`` over every contributor,
    then divides (filling empty pixels with ``blend.fill_value``). Returns
    the blended ``float32`` array; the zarr write stays in the actor.
    """
    import numpy as np

    out_shape = geom_entry["out_shape"]
    contrib_geom = geom_entry["contributors"]

    # Accumulate in float32 regardless of the contributor dtype: a float16
    # recon store (monarch.recon_dtype=float16) halves the RDMA payload, but
    # summing value*weight in float16 would overflow/round badly. The float32
    # kernel upcasts each f16 ``v_view`` for free, so only the stored tile
    # loses bits — the blend math is full precision.
    accum_dtype = np.float32
    accum_v = np.zeros(out_shape, dtype=accum_dtype)
    accum_w = np.zeros(out_shape, dtype=accum_dtype)

    for tid, tile_full in contribs_np.items():
        cinfo = contrib_geom.get(tid)
        if cinfo is None:
            continue
        kernel_full = get_blend_kernel(blend, cinfo["tile_shape"], accum_dtype, kernel_cache)
        kernel_view = kernel_full[cinfo["in_local"]]
        v_view = tile_full[cinfo["in_full_idx"]]
        accum_v[cinfo["out_full_idx"]] += v_view * kernel_view
        accum_w[cinfo["out_full_idx"]] += kernel_view

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(accum_w > 0, accum_v / accum_w, blend.fill_value).astype(
            np.float32, copy=False
        )
    return result
