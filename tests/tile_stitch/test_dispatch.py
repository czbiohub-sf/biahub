"""Tests for the recon-dispatch order + budget. A fake int-keyed plan (the
``RunPlan`` duck type — ``tile_id`` + ``slices: dict[str, slice]`` +
``output_to_inputs`` + ``tile_dims``) is fed through ``morton_output_order``,
without building a full waveorder plan."""

from __future__ import annotations

from dataclasses import dataclass

from biahub.tile_stitch.dispatch import morton_output_order, peak_resident_tiles

TILE, OVERLAP = 8, 2
STRIDE = TILE - OVERLAP
DIMS = ("z", "y", "x")


@dataclass
class _Tile:
    tile_id: int
    slices: dict


@dataclass
class _Plan:
    output_to_inputs: dict
    output_tiles: list
    tile_dims: tuple


def _fake_plan(n: int):
    def gid(z, y, x):
        return z * n * n + y * n + x

    def ov(a, b):
        return a * STRIDE < b * STRIDE + STRIDE and a * STRIDE + TILE > b * STRIDE

    out_to_in, out_tiles = {}, []
    for oz in range(n):
        for oy in range(n):
            for ox in range(n):
                oid = gid(oz, oy, ox)
                out_to_in[oid] = [
                    gid(iz, iy, ix)
                    for iz in range(n) if ov(iz, oz)
                    for iy in range(n) if ov(iy, oy)
                    for ix in range(n) if ov(ix, ox)
                ]
                out_tiles.append(_Tile(oid, {
                    "z": slice(oz * STRIDE, oz * STRIDE + STRIDE),
                    "y": slice(oy * STRIDE, oy * STRIDE + STRIDE),
                    "x": slice(ox * STRIDE, ox * STRIDE + STRIDE),
                }))
    return _Plan(out_to_in, out_tiles, DIMS)


def test_morton_order_is_valid_permutation():
    plan = _fake_plan(6)
    order = morton_output_order(plan)
    assert sorted(order) == sorted(plan.output_to_inputs)  # no dupes/drops
    assert order != sorted(plan.output_to_inputs)  # actually Z-order reordered


def test_peak_resident_tiles_is_interval_overlap():
    # tiles' [first,last] output spans: 0:(0,0) 1:(0,1) 2:(1,2) 3:(2,2)
    # max simultaneously-live = 2 at every position.
    assert peak_resident_tiles({0: [0, 1], 1: [1, 2], 2: [2, 3]}, [0, 1, 2]) == 2
