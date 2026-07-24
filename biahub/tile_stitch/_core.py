"""Tile-stitch compute helpers: pure functions + small caches.

No Monarch imports, so this layer unit-tests without a mesh; the actor is a thin
transport shell over it.

Leading-axis convention: geometry strips the T axis (``leading_shape[1:]``) and
the write region carries T via an explicit ``slice(t_off, t_off + 1)``.
"""

from collections.abc import Callable


def build_stitch_geom(plan) -> dict[int, dict]:
    """Precompute contributor intersections for each output tile.

    Parameters
    ----------
    plan : RunPlan
        Execution plan containing input and output tile geometry.

    Returns
    -------
    dict[int, dict]
        Geometry indexed by output tile ID, including output shapes and
        contributor-local slices.

    Notes
    -----
    Timepoint is excluded from geometry and supplied by the write region.
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


def make_eager_recon(cuda_tf, recon_settings) -> Callable:
    """Build the eager reconstruction closure for the active modality.

    Parameters
    ----------
    cuda_tf : dict[str, torch.Tensor]
        Transfer-function tensors for the selected modality.
    recon_settings : object
        Resolved modality-specific reconstruction settings.

    Returns
    -------
    Callable
        Function mapping one source tensor to its reconstruction.

    Notes
    -----
    Transfer functions and inverse settings are captured so compiled callers
    receive only the source tensor.
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


def get_blend_kernel(blend, tile_shape: tuple, dtype, cache: dict):
    """Return a cached blend weight kernel.

    Parameters
    ----------
    blend : object
        Blend configuration providing ``weight_kernel``.
    tile_shape : tuple[int, ...]
        Spatial shape for which to build weights.
    dtype : numpy.dtype
        Requested kernel dtype.
    cache : dict
        Mutable kernel cache keyed by shape and dtype.

    Returns
    -------
    numpy.ndarray
        Blend weights for ``tile_shape``.
    """
    key = (tile_shape, str(dtype))
    cached = cache.get(key)
    if cached is not None:
        return cached
    kernel = blend.weight_kernel(tile_shape).astype(dtype, copy=False)
    cache[key] = kernel
    return kernel


def _get_reduced_blend_kernel(blend, tile_shape, active_axes, dtype, cache):
    """Return a separable kernel with common per-voxel factors removed."""
    import numpy as np

    key = ("reduced", tile_shape, active_axes, str(dtype))
    cached = cache.get(key)
    if cached is not None:
        return cached

    shape = tuple(
        size if active else 1
        for size, active in zip(tile_shape, active_axes, strict=True)
    )
    kernel = np.ones(shape, dtype=dtype)
    for axis, (tile_size, active) in enumerate(
        zip(tile_shape, active_axes, strict=True)
    ):
        if not active:
            continue
        axis_shape = [1] * len(tile_shape)
        axis_shape[axis] = tile_size
        kernel *= get_blend_kernel(
            blend, (tile_size,), dtype, cache
        ).reshape(axis_shape)
    cache[key] = kernel
    return kernel


def _separable_weight_sum(geom_entry, blend, dtype, kernel_cache):
    """Build a reduced Gaussian or uniform denominator from 1-D axis sums."""
    import math

    import numpy as np

    if blend.name != "uniform_mean" and not blend.name.startswith("gaussian_mean"):
        return None, None

    n_lead = geom_entry["n_lead"]
    spatial_shape = geom_entry["out_shape"][n_lead:]
    axis_parts: list[dict[tuple, tuple]] = [dict() for _ in spatial_shape]
    combinations = set()

    def _bounds(value: slice, size: int) -> tuple[int, int, int]:
        return (
            0 if value.start is None else value.start,
            size if value.stop is None else value.stop,
            1 if value.step is None else value.step,
        )

    for cinfo in geom_entry["contributors"].values():
        combination = []
        for axis, (tile_size, in_slice, out_slice) in enumerate(
            zip(
                cinfo["tile_shape"],
                cinfo["in_local"],
                cinfo["out_full_idx"][n_lead:],
                strict=True,
            )
        ):
            key = (
                tile_size,
                _bounds(in_slice, tile_size),
                _bounds(out_slice, spatial_shape[axis]),
            )
            axis_parts[axis][key] = (tile_size, in_slice, out_slice)
            combination.append(key)
        combinations.add(tuple(combination))

    n_contributors = len(geom_entry["contributors"])
    if (
        len(combinations) != n_contributors
        or math.prod(len(parts) for parts in axis_parts) != n_contributors
    ):
        return None, None

    active_axes = tuple(len(parts) > 1 for parts in axis_parts)
    axis_sums = []
    for axis, (parts, active) in enumerate(
        zip(axis_parts, active_axes, strict=True)
    ):
        if not active:
            axis_sums.append(np.ones(1, dtype=dtype))
            continue
        total = np.zeros(spatial_shape[axis], dtype=dtype)
        for tile_size, in_slice, out_slice in parts.values():
            kernel_1d = get_blend_kernel(
                blend, (tile_size,), dtype, kernel_cache
            )
            total[out_slice] += kernel_1d[in_slice]
        axis_sums.append(total)

    denominator = axis_sums[0]
    for axis_sum in axis_sums[1:]:
        denominator = np.multiply.outer(denominator, axis_sum)
    return denominator, active_axes


def blend_contributors(geom_entry, contribs_np, blend, kernel_cache):
    """Blend contributor tiles into one weighted-mean output.

    Parameters
    ----------
    geom_entry : dict
        Precomputed output geometry from :func:`build_stitch_geom`.
    contribs_np : Mapping[int, numpy.ndarray]
        Contributor arrays keyed by input tile ID.
    blend : object
        Blend configuration defining weights and the fill value.
    kernel_cache : dict
        Mutable cache of blend weight kernels.

    Returns
    -------
    numpy.ndarray
        Blended float32 output array.
    """
    import numpy as np
    import torch

    out_shape = geom_entry["out_shape"]
    contrib_geom = geom_entry["contributors"]

    # Accumulate in float32 even when recon storage uses float16.
    accum_dtype = np.float32
    accum_v = np.zeros(out_shape, dtype=accum_dtype)
    accum_w, active_axes = _separable_weight_sum(
        geom_entry, blend, accum_dtype, kernel_cache
    )
    accumulate_weights = accum_w is None
    if accumulate_weights:
        accum_w = np.zeros(out_shape, dtype=accum_dtype)

    for tid, tile_full in contribs_np.items():
        cinfo = contrib_geom.get(tid)
        if cinfo is None:
            continue
        if active_axes is None:
            kernel_full = get_blend_kernel(
                blend, cinfo["tile_shape"], accum_dtype, kernel_cache
            )
            kernel_view = kernel_full[cinfo["in_local"]]
        else:
            kernel_full = _get_reduced_blend_kernel(
                blend,
                cinfo["tile_shape"],
                active_axes,
                accum_dtype,
                kernel_cache,
            )
            kernel_view = kernel_full[
                tuple(
                    in_slice if active else slice(None)
                    for in_slice, active in zip(
                        cinfo["in_local"], active_axes, strict=True
                    )
                )
            ]
        v_view = tile_full[cinfo["in_full_idx"]]
        accum_view = accum_v[cinfo["out_full_idx"]]
        accum_tensor = torch.from_numpy(accum_view)
        torch.addcmul(
            accum_tensor,
            torch.from_numpy(v_view),
            torch.from_numpy(kernel_view),
            out=accum_tensor,
        )
        if accumulate_weights:
            accum_w[cinfo["out_full_idx"]] += kernel_view

    nonzero = accum_w > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(accum_v, accum_w, out=accum_v, where=nonzero)
    if blend.fill_value != 0:
        np.copyto(accum_v, blend.fill_value, where=~nonzero)
    return accum_v
