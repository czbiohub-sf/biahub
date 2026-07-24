"""Tests for the recon-dispatch order + budget. A fake int-keyed plan (the
``RunPlan`` duck type — ``tile_id`` + ``slices: dict[str, slice]`` +
``output_to_inputs`` + ``tile_dims``) is fed through ``morton_output_order``,
without building a full waveorder plan."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from biahub.tile_stitch.dispatch import (
    SchedulerContext,
    build_dispatch_plan,
    build_dispatch_problem,
    morton_output_order,
    register_scheduler,
    scheduler_names,
)

TILE, OVERLAP = 8, 2
STRIDE = TILE - OVERLAP
DIMS = ("z", "y", "x")


@dataclass
class _Tile:
    tile_id: int
    slices: dict


@dataclass
class _InputTile:
    tile_id: int
    shape: tuple[int, ...] = (TILE, TILE, TILE)


@dataclass
class _Plan:
    output_to_inputs: dict
    output_tiles: list
    tile_dims: tuple
    input_order: list
    input_tiles: list


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
                    for iz in range(n)
                    if ov(iz, oz)
                    for iy in range(n)
                    if ov(iy, oy)
                    for ix in range(n)
                    if ov(ix, ox)
                ]
                out_tiles.append(
                    _Tile(
                        oid,
                        {
                            "z": slice(oz * STRIDE, oz * STRIDE + STRIDE),
                            "y": slice(oy * STRIDE, oy * STRIDE + STRIDE),
                            "x": slice(ox * STRIDE, ox * STRIDE + STRIDE),
                        },
                    )
                )
    input_order = sorted({tile_id for inputs in out_to_in.values() for tile_id in inputs})
    return _Plan(
        out_to_in,
        out_tiles,
        DIMS,
        input_order,
        [_InputTile(tile_id) for tile_id in input_order],
    )


def test_morton_order_is_valid_permutation():
    problem = build_dispatch_problem(_fake_plan(6))
    order = morton_output_order(problem)
    assert sorted(order) == sorted(problem.output_to_inputs)  # no dupes/drops
    assert order != sorted(problem.output_to_inputs)  # actually Z-order reordered


def test_registered_schedulers_return_complete_deterministic_permutations():
    problem = build_dispatch_problem(_fake_plan(4))
    expected = sorted(problem.input_order)
    context = SchedulerContext(window=16, recon_batch=4, n_actors=2)
    assert scheduler_names() == ("plan", "morton", "windowed_graph_ready")

    for name in scheduler_names():
        first = build_dispatch_plan(problem, name, context=context)
        second = build_dispatch_plan(problem, name, context=context)
        assert sorted(first.input_order) == expected
        assert first == second
        assert first.metrics["scheduler"] == name
        assert first.metrics["peak_resident_tiles"] >= 1
        with pytest.raises(TypeError):
            first.metrics["mutated"] = 1


def test_windowed_graph_ready_prioritizes_the_closest_output():
    plan = _Plan(
        output_to_inputs={0: [0, 1, 2, 3], 1: [4, 5]},
        output_tiles=[
            _Tile(0, {"z": slice(0, 1), "y": slice(0, 1), "x": slice(0, 1)}),
            _Tile(1, {"z": slice(0, 1), "y": slice(0, 1), "x": slice(1, 2)}),
        ],
        tile_dims=DIMS,
        input_order=[0, 1, 2, 3, 4, 5],
        input_tiles=[_InputTile(tile_id) for tile_id in range(6)],
    )

    problem = build_dispatch_problem(plan)
    context = SchedulerContext(window=6, recon_batch=1, n_actors=1)
    morton = build_dispatch_plan(problem, "morton", context=context)
    graph = build_dispatch_plan(problem, "windowed_graph_ready", context=context)

    assert morton.metrics["first_output_ready_at"] == 4
    assert graph.input_order[:2] == (4, 5)
    assert graph.metrics["first_output_ready_at"] == 2
    assert graph.metrics["mean_output_ready_at"] < morton.metrics["mean_output_ready_at"]


def test_unknown_scheduler_fails_with_registered_names():
    problem = build_dispatch_problem(_fake_plan(2))
    context = SchedulerContext(window=4, recon_batch=1, n_actors=1)
    with pytest.raises(ValueError, match="windowed_graph_ready"):
        build_dispatch_plan(problem, "missing", context=context)


def test_dispatch_problem_is_deeply_immutable():
    problem = build_dispatch_problem(_fake_plan(2))
    with pytest.raises(TypeError):
        problem.input_shapes[problem.input_order[0]] = (1, 1, 1)
    with pytest.raises(TypeError):
        problem.output_to_inputs[next(iter(problem.output_to_inputs))] = ()


@pytest.mark.parametrize("field", ["window", "recon_batch", "n_actors"])
def test_scheduler_context_requires_positive_values(field: str):
    values = {"window": 4, "recon_batch": 2, "n_actors": 1}
    values[field] = 0
    with pytest.raises(ValueError, match=field):
        SchedulerContext(**values)


def test_scheduler_registration_rejects_invalid_and_duplicate_names():
    def strategy(problem, context):
        return problem.input_order

    with pytest.raises(ValueError, match="invalid scheduler name"):
        register_scheduler("not-valid", strategy)
    with pytest.raises(ValueError, match="already registered"):
        register_scheduler("plan", strategy)


def test_problem_rejects_duplicate_input_order():
    plan = _fake_plan(2)
    plan.input_order.append(plan.input_order[0])
    with pytest.raises(ValueError, match="input_order contains duplicate"):
        build_dispatch_problem(plan)


def test_problem_rejects_unknown_and_repeated_contributors():
    plan = _fake_plan(2)
    output_id = next(iter(plan.output_to_inputs))
    plan.output_to_inputs[output_id].append(999)
    with pytest.raises(ValueError, match="unknown inputs"):
        build_dispatch_problem(plan)

    plan = _fake_plan(2)
    output_id = next(iter(plan.output_to_inputs))
    plan.output_to_inputs[output_id].append(plan.output_to_inputs[output_id][0])
    with pytest.raises(ValueError, match="repeats a contributor"):
        build_dispatch_problem(plan)


def test_problem_rejects_duplicate_output_coordinates():
    plan = _fake_plan(2)
    plan.output_tiles[1].slices = dict(plan.output_tiles[0].slices)
    with pytest.raises(ValueError, match="coordinates are not unique"):
        build_dispatch_problem(plan)


def test_problem_rejects_unconsumed_inputs():
    plan = _fake_plan(2)
    for contributors in plan.output_to_inputs.values():
        while plan.input_order[-1] in contributors:
            contributors.remove(plan.input_order[-1])
    with pytest.raises(ValueError, match="no output consumer"):
        build_dispatch_problem(plan)
