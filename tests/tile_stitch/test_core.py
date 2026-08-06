"""CPU-only parity guard for ``biahub.tile_stitch._core``.

No CUDA, no Monarch — exercises the pure geometry + blend logic that was
lifted out of the Monarch ``tile_worker`` actor. This pins:

  * ``build_stitch_geom`` intersection slices for a known 2-tile overlap,
  * ``blend_contributors`` weighted-mean output against a hand-computed
    reference, and
  * the ``timepoint > 0`` write-region convention — the T axis is carried by
    an explicit ``slice(t_off, t_off + 1)`` while the spatial geometry uses
    ``leading_shape[1:]`` (the exact bug the v6-vs-v7 leading-axis divergence
    would have introduced).
"""

from __future__ import annotations

import numpy as np

from waveorder.tile_stitch.blend import Blend
from waveorder.tile_stitch.partition import InputTile, OutputTile

from biahub.tile_stitch import _core


def _ramp_kernel(shape: tuple[int, ...]) -> np.ndarray:
    """Non-uniform weight kernel: column index + 1 along the last axis.

    Constant across leading + earlier spatial axes, increasing along X. This
    makes the overlap-region weighted mean differ from a plain arithmetic
    mean, so the test actually exercises the weighting (not just averaging).
    """
    x = shape[-1]
    row = np.arange(1, x + 1, dtype=np.float64)
    return np.broadcast_to(row, shape).copy()


def _ramp_blend() -> Blend:
    """A minimal ``Blend`` carrying only the fields ``blend_contributors`` uses.

    ``blend_contributors`` touches just ``weight_kernel`` and ``fill_value``;
    the accumulate/combine/finalize callables are required by the dataclass
    but never invoked here, so they are stubbed.
    """

    def _unused(*_a, **_k):  # pragma: no cover - never called by _core
        raise AssertionError("blend accumulate path not exercised by _core")

    return Blend(
        name="ramp",
        weight_kernel=_ramp_kernel,
        init=_unused,
        combine=_unused,
        finalize=_unused,
        fill_value=float("nan"),
    )


class _Plan:
    """Minimal ``RunPlan``-shaped object for the geometry builder.

    ``build_stitch_geom`` only reads ``input_tiles``, ``output_tiles``,
    ``output_to_inputs``, ``tile_dims`` and ``leading_shape`` — so a tiny
    namespace stand-in keeps the test free of the (currently broken) recon
    settings schema.
    """

    def __init__(
        self, *, input_tiles, output_tiles, output_to_inputs, tile_dims, leading_shape
    ):
        self.input_tiles = input_tiles
        self.output_tiles = output_tiles
        self.output_to_inputs = output_to_inputs
        self.tile_dims = tile_dims
        self.leading_shape = leading_shape


def _two_tile_plan(leading_shape: tuple[int, ...]):
    """One 8×8 output tile fed by two tiles overlapping in x[3:5]."""
    tile_dims = ("y", "x")
    in0 = InputTile(tile_id=0, slices={"y": slice(0, 8), "x": slice(0, 5)})
    in1 = InputTile(tile_id=1, slices={"y": slice(0, 8), "x": slice(3, 8)})
    out0 = OutputTile(tile_id=0, slices={"y": slice(0, 8), "x": slice(0, 8)})
    plan = _Plan(
        input_tiles=[in0, in1],
        output_tiles=[out0],
        output_to_inputs={0: [0, 1]},
        tile_dims=tile_dims,
        leading_shape=leading_shape,
    )
    return plan


def test_build_stitch_geom_intersection_slices():
    plan = _two_tile_plan(leading_shape=(1, 1))
    geom = _core.build_stitch_geom(plan)

    entry = geom[0]
    # leading_shape[1:] == (1,) → one leading axis (C); spatial is 8×8.
    assert entry["n_lead"] == 1
    assert entry["out_shape"] == (1, 8, 8)
    assert entry["out_spatial"] == [(0, 8), (0, 8)]

    contribs = entry["contributors"]
    assert set(contribs) == {0, 1}

    # Tile 0 spans x[0:5] → its whole extent lands in the output unchanged.
    c0 = contribs[0]
    assert c0["tile_shape"] == (8, 5)
    assert c0["in_local"] == (slice(0, 8), slice(0, 5))
    # Leading axis is a full slice; spatial slices follow.
    assert c0["in_full_idx"] == (slice(None), slice(0, 8), slice(0, 5))
    assert c0["out_full_idx"] == (slice(None), slice(0, 8), slice(0, 5))

    # Tile 1 spans x[3:8]; in its own frame x maps to local [0:5], output [3:8].
    c1 = contribs[1]
    assert c1["tile_shape"] == (8, 5)
    assert c1["in_local"] == (slice(0, 8), slice(0, 5))
    assert c1["in_full_idx"] == (slice(None), slice(0, 8), slice(0, 5))
    assert c1["out_full_idx"] == (slice(None), slice(0, 8), slice(3, 8))


def test_blend_contributors_weighted_mean():
    plan = _two_tile_plan(leading_shape=(1, 1))
    geom = _core.build_stitch_geom(plan)
    blend = _ramp_blend()
    kernel_cache: dict[tuple, np.ndarray] = {}

    # Constant-valued tiles so the weighted mean is hand-computable.
    # Contributor arrays are full (C, Y, X) tiles: tile 0 is (1,8,5)=10,
    # tile 1 is (1,8,5)=20.
    t0 = np.full((1, 8, 5), 10.0, dtype=np.float64)
    t1 = np.full((1, 8, 5), 20.0, dtype=np.float64)
    contribs = {0: t0, 1: t1}

    result = _core.blend_contributors(geom[0], contribs, blend, kernel_cache)
    assert result.shape == (1, 8, 8)
    assert result.dtype == np.float32

    # The ramp kernel weights are column-index+1 within each tile (1..5).
    # Non-overlap columns take a single contributor → exact value.
    #   output x[0:3]  ← tile0 local x[0:3]  → value 10
    #   output x[5:8]  ← tile1 local x[2:5]  → value 20
    assert np.allclose(result[0, :, 0:3], 10.0)
    assert np.allclose(result[0, :, 5:8], 20.0)

    # Overlap output x[3:5] has both contributors:
    #   x=3: tile0 local col 3 (w=4, v=10) + tile1 local col 0 (w=1, v=20)
    #        → (4*10 + 1*20) / (4 + 1) = 60/5 = 12
    #   x=4: tile0 local col 4 (w=5, v=10) + tile1 local col 1 (w=2, v=20)
    #        → (5*10 + 2*20) / (5 + 2) = 90/7
    assert np.allclose(result[0, :, 3], 12.0)
    assert np.allclose(result[0, :, 4], 90.0 / 7.0)


def test_blend_contributors_fill_value_outside_coverage():
    """A pixel no contributor covers comes back as the blend fill_value."""
    tile_dims = ("y", "x")
    # Single contributor covering only x[0:4] of an 8-wide output → x[4:8]
    # has zero accumulated weight.
    in0 = InputTile(tile_id=0, slices={"y": slice(0, 8), "x": slice(0, 4)})
    out0 = OutputTile(tile_id=0, slices={"y": slice(0, 8), "x": slice(0, 8)})
    plan = _Plan(
        input_tiles=[in0],
        output_tiles=[out0],
        output_to_inputs={0: [0]},
        tile_dims=tile_dims,
        leading_shape=(1, 1),
    )
    geom = _core.build_stitch_geom(plan)
    blend = _ramp_blend()

    t0 = np.full((1, 8, 4), 7.0, dtype=np.float64)
    result = _core.blend_contributors(geom[0], {0: t0}, blend, {})
    assert np.allclose(result[0, :, 0:4], 7.0)
    assert np.all(np.isnan(result[0, :, 4:8]))


def _write_region(plan_timepoint, out_c_idx, out_spatial):
    """Replicate the actor's stitch write-region convention.

    The T axis is carried by an explicit ``slice(t_off, t_off + 1)``; the C axis
    by this channel's slot ``slice(out_c_idx, out_c_idx + 1)`` in the (possibly
    multi-channel) shared output; the spatial dims by the per-output-tile
    geometry. Pinning it here guards against (a) the v6 leading-axis regression
    that folded T into the leading dims and mis-placed every multi-TP write, and
    (b) mis-placing a channel's C slot in a shared multi-channel output.
    """
    return (
        (slice(plan_timepoint, plan_timepoint + 1),)
        + (slice(out_c_idx, out_c_idx + 1),)
        + tuple(slice(lo, hi) for lo, hi in out_spatial)
    )


def test_timepoint_write_region_carries_t_axis():
    leading_shape = (5, 1)  # T=5, C=1 — multi-TP output zarr.
    plan = _two_tile_plan(leading_shape=leading_shape)
    geom = _core.build_stitch_geom(plan)
    entry = geom[0]

    # Geometry strips T: it uses leading_shape[1:] == (1,), so the blended
    # array is (C, Y, X) with NO T axis baked in.
    assert entry["n_lead"] == 1
    assert entry["out_shape"] == (1, 8, 8)

    t_off = 3
    region = _write_region(t_off, 0, entry["out_spatial"])  # single-channel, C=0

    # First element is the explicit T slot for this timepoint.
    assert region[0] == slice(3, 4)
    # Then the C slot (0 here), then the two spatial dims.
    assert region[1] == slice(0, 1)
    assert region[2] == slice(0, 8)
    assert region[3] == slice(0, 8)

    # Multi-channel shared output: channel index 1 writes the C=1 slot, T/spatial
    # unchanged. This is what lets phase+fluor share one (T, C, Z, Y, X) zarr.
    region_c1 = _write_region(t_off, 1, entry["out_spatial"])
    assert region_c1[0] == slice(3, 4)
    assert region_c1[1] == slice(1, 2)
    assert region_c1[2:] == region[2:]
    # The blended array (with a leading None for the T axis at write time)
    # has rank matching the write region.
    result = _core.blend_contributors(
        entry,
        {0: np.full((1, 8, 5), 1.0), 1: np.full((1, 8, 5), 1.0)},
        _ramp_blend(),
        {},
    )
    assert result[None].shape == (1, 1, 8, 8)
    assert len(region) == result[None].ndim


def test_build_recon_batches_invariants():
    """Batches are shape-uniform, <= bsize, and cover every tile exactly once.
    Differing shapes bucket separately (a batched FFT needs one shape)."""
    from biahub.tile_stitch.monarch.backend import build_recon_batches

    # Mixed shapes interleaved: tids 0,4,8,12 are wider in x (ragged edge), the
    # rest uniform — so they must bucket into separate batches.
    tiles, order = [], []
    for tid in range(13):
        x_hi = 4 if tid % 4 == 0 else 2
        tiles.append(
            InputTile(
                tile_id=tid, slices={"z": slice(0, 2), "y": slice(0, 2), "x": slice(0, x_hi)}
            )
        )
        order.append(tid)
    tiles_by_id = {t.tile_id: t for t in tiles}

    batches = build_recon_batches(order, tiles_by_id, 4)
    flat = [t for b in batches for t in b]
    assert sorted(flat) == sorted(order)  # every tile exactly once
    assert all(len(b) <= 4 for b in batches)  # respects bsize
    for b in batches:  # shape-uniform per batch
        assert len({tuple(tiles_by_id[t].shape) for t in b}) == 1


def test_frozen_monarch_config_pickles_through_plan():
    """Guard 3b: the frozen MonarchConfig survives the plan→pickle→load path.

    The CLI carries the resolved MonarchConfig on the pickled RunPlan; the
    actor reads it back after ``load_plan`` (init) and ``swap_to`` (per TP).
    Confirm a frozen, StrEnum-bearing config round-trips through pickle with
    identical field values — the actor must never see a mangled config.
    """
    import pickle

    from biahub.settings import CompileMode, Device, MonarchConfig

    cfg = MonarchConfig(
        gpus_per_node=4,
        recon_batch=1,
        compile_mode=CompileMode.NONE,
        device=Device.CUDA,
    )
    restored = pickle.loads(pickle.dumps(cfg, protocol=pickle.HIGHEST_PROTOCOL))
    assert restored == cfg
    assert restored.compile_mode is CompileMode.NONE
    assert restored.compile_mode == "none"
    assert restored.recon_batch == 1
    # Still frozen after unpickle.
    import pytest

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        restored.recon_batch = 8


def test_blend_contributors_float16_accumulates_in_float32():
    """A float16 recon store (monarch.recon_dtype=float16) blends in float32.

    The float16 store halves the RDMA payload, but the weighted-mean
    accumulation must stay float32 (summing value*weight in float16 would
    overflow/round badly). ``blend_contributors`` forces a float32 accumulator
    regardless of contributor dtype, so the float16-input result must match a
    float32-reference blend of the same values to well within float16 noise.
    """
    plan = _two_tile_plan(leading_shape=(1, 1))
    geom = _core.build_stitch_geom(plan)
    blend = _ramp_blend()

    # Non-trivial values (a value range float16 stores exactly enough of) so the
    # weighted mean exercises real arithmetic, not just constants.
    rng = np.random.default_rng(0)
    v0 = (rng.uniform(100.0, 4000.0, size=(1, 8, 5))).astype(np.float32)
    v1 = (rng.uniform(100.0, 4000.0, size=(1, 8, 5))).astype(np.float32)

    # float16 contributors → the path under test.
    res_f16 = _core.blend_contributors(
        geom[0], {0: v0.astype(np.float16), 1: v1.astype(np.float16)}, blend, {}
    )
    # float32 reference: same values, full precision throughout.
    res_f32 = _core.blend_contributors(geom[0], {0: v0, 1: v1}, blend, {})

    # Accumulation must be float32 regardless of the (float16) contributor dtype.
    assert res_f16.dtype == np.float32
    assert res_f32.dtype == np.float32

    # Compare only the covered region (the fill_value here is NaN).
    covered = ~np.isnan(res_f32)
    diff = res_f16[covered] - res_f32[covered]
    nrmse = np.sqrt(np.mean(diff**2)) / (np.sqrt(np.mean(res_f32[covered] ** 2)) + 1e-12)
    assert nrmse < 1e-2, f"float16 blend NRMSE {nrmse:.2e} exceeds 1e-2"


def test_recon_dtype_default_is_float32_and_float16_opt_in():
    """recon_dtype defaults to lossless float32; float16 is opt-in.

    The stored/transmitted recon dtype halves the D2H + ibverbs-MR bytes at
    float16 (lossy, ~3 sig digits), so the default must stay lossless.
    """
    from biahub.settings import MonarchConfig

    assert MonarchConfig().recon_dtype == "float32"
    assert MonarchConfig(recon_dtype="float16").recon_dtype == "float16"
