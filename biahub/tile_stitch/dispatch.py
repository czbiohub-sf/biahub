"""Validated reconstruction scheduling for tile-stitch plans.

Schedulers receive an immutable problem and return only an input order. This
module validates that order and derives output readiness, resident-set bounds,
and planner telemetry before actor work begins.
"""


import logging
import statistics

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

logger = logging.getLogger(__name__)

Metric = int | float | str


class InputTileView(Protocol):
    """Input tile metadata required by schedulers.

    Attributes
    ----------
    tile_id : int
        Unique input tile identifier.
    shape : Sequence[int]
        Positive tile dimensions.
    """

    @property
    def tile_id(self) -> int: ...

    @property
    def shape(self) -> Sequence[int]: ...


class OutputTileView(Protocol):
    """Output tile geometry required by spatial schedulers.

    Attributes
    ----------
    tile_id : int
        Unique output tile identifier.
    slices : Mapping[str, slice]
        Output bounds keyed by tile dimension.
    """

    @property
    def tile_id(self) -> int: ...

    @property
    def slices(self) -> Mapping[str, slice]: ...


class DispatchProblemSource(Protocol):
    """Structural scheduling view of a runtime plan.

    Attributes
    ----------
    input_order : Sequence[int]
        Declared reconstruction order.
    input_tiles : Sequence[InputTileView]
        Input tile metadata.
    output_tiles : Sequence[OutputTileView]
        Output tile metadata.
    output_to_inputs : Mapping[int, Sequence[int]]
        Contributor IDs for each output tile.
    tile_dims : Sequence[str]
        Ordered spatial dimensions used to derive coordinates.
    """

    @property
    def input_order(self) -> Sequence[int]: ...

    @property
    def input_tiles(self) -> Sequence[InputTileView]: ...

    @property
    def output_tiles(self) -> Sequence[OutputTileView]: ...

    @property
    def output_to_inputs(self) -> Mapping[int, Sequence[int]]: ...

    @property
    def tile_dims(self) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class SchedulerContext:
    """Runtime dimensions shared by scheduling strategies.

    Attributes
    ----------
    window : int
        Candidate look-ahead size.
    recon_batch : int
        Reconstruction batch size.
    n_actors : int
        Number of reconstruction actors.

    Raises
    ------
    ValueError
        If any value is not positive.
    """

    window: int
    recon_batch: int
    n_actors: int

    def __post_init__(self) -> None:
        for name in ("window", "recon_batch", "n_actors"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class DispatchProblem:
    """Immutable scheduling projection of a tile-stitch plan.

    Attributes
    ----------
    input_order : tuple[int, ...]
        Declared input order.
    input_shapes : Mapping[int, tuple[int, ...]]
        Input shapes keyed by tile ID.
    output_coordinates : Mapping[int, tuple[int, ...]]
        Output start coordinates keyed by tile ID.
    output_to_inputs : Mapping[int, tuple[int, ...]]
        Contributor IDs keyed by output tile ID.
    """

    input_order: tuple[int, ...]
    input_shapes: Mapping[int, tuple[int, ...]]
    output_coordinates: Mapping[int, tuple[int, ...]]
    output_to_inputs: Mapping[int, tuple[int, ...]]


class SchedulerStrategy(Protocol):
    """Callable protocol for deterministic scheduling strategies."""

    def __call__(
        self,
        problem: DispatchProblem,
        context: SchedulerContext,
        /,
    ) -> Sequence[int]:
        """Produce a reconstruction order.

        Parameters
        ----------
        problem : DispatchProblem
            Validated immutable scheduling problem.
        context : SchedulerContext
            Shared runtime dimensions.

        Returns
        -------
        Sequence[int]
            Permutation of every input tile ID.
        """
        ...


def build_dispatch_problem(source: DispatchProblemSource) -> DispatchProblem:
    """Validate and freeze the scheduling subset of a runtime plan.

    Parameters
    ----------
    source : DispatchProblemSource
        Runtime plan or compatible structural view.

    Returns
    -------
    DispatchProblem
        Immutable validated scheduling problem.

    Raises
    ------
    ValueError
        If IDs, shapes, coordinates, contributor mappings, or consumers violate
        the scheduling contract.
    """
    input_order = tuple(source.input_order)
    if len(input_order) != len(set(input_order)):
        raise ValueError("dispatch problem input_order contains duplicate tile IDs")

    input_shapes: dict[int, tuple[int, ...]] = {}
    for tile in source.input_tiles:
        if tile.tile_id in input_shapes:
            raise ValueError(f"duplicate input tile metadata for {tile.tile_id}")
        shape = tuple(int(size) for size in tile.shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError(f"input tile {tile.tile_id} has invalid shape {shape}")
        input_shapes[tile.tile_id] = shape

    declared = set(input_order)
    if declared != set(input_shapes):
        raise ValueError(
            "dispatch problem input_order and input tile metadata disagree: "
            f"missing_metadata={sorted(declared - set(input_shapes))[:5]}, "
            f"unordered_metadata={sorted(set(input_shapes) - declared)[:5]}"
        )

    dims = tuple(source.tile_dims)
    output_coordinates: dict[int, tuple[int, ...]] = {}
    for tile in source.output_tiles:
        if tile.tile_id in output_coordinates:
            raise ValueError(f"duplicate output tile metadata for {tile.tile_id}")
        try:
            coordinate = tuple(int(tile.slices[dim].start) for dim in dims)
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"output tile {tile.tile_id} lacks a start for dimensions {dims}"
            ) from exc
        output_coordinates[tile.tile_id] = coordinate
    coordinates = list(output_coordinates.values())
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("dispatch problem output coordinates are not unique")

    output_to_inputs: dict[int, tuple[int, ...]] = {}
    for output_id, contributors in source.output_to_inputs.items():
        frozen = tuple(contributors)
        if len(frozen) != len(set(frozen)):
            raise ValueError(f"output tile {output_id} repeats a contributor")
        unknown = set(frozen) - declared
        if unknown:
            raise ValueError(
                f"output tile {output_id} references unknown inputs {sorted(unknown)[:5]}"
            )
        if output_id not in output_coordinates:
            raise ValueError(f"output tile {output_id} has no geometry")
        output_to_inputs[output_id] = frozen

    referenced = {tile_id for inputs in output_to_inputs.values() for tile_id in inputs}
    if referenced != declared:
        raise ValueError(
            "dispatch problem contains inputs with no output consumer: "
            f"{sorted(declared - referenced)[:5]}"
        )

    return DispatchProblem(
        input_order=input_order,
        input_shapes=MappingProxyType(input_shapes),
        output_coordinates=MappingProxyType(output_coordinates),
        output_to_inputs=MappingProxyType(output_to_inputs),
    )


def morton_order(cells: Iterable[Sequence[int]]) -> list:
    """Sort integer coordinates in k-dimensional Morton order.

    Parameters
    ----------
    cells : Iterable[Sequence[int]]
        Integer coordinates with a common dimensionality.

    Returns
    -------
    list
        Coordinates ordered by interleaved Morton code.
    """
    cells = list(cells)
    if not cells:
        return cells
    ndim = len(cells[0])
    bits = max(1, max(max(c) for c in cells).bit_length())

    def code(c: Sequence[int]) -> int:
        r = 0
        for i in range(bits):
            for d in range(ndim):
                r |= ((c[d] >> i) & 1) << (i * ndim + d)
        return r

    return sorted(cells, key=code)


def morton_output_order(problem: DispatchProblem) -> list[int]:
    """Order output tile IDs by validated Morton coordinates.

    Parameters
    ----------
    problem : DispatchProblem
        Problem containing unique output coordinates.

    Returns
    -------
    list[int]
        Output tile IDs in Morton order.
    """
    ids = tuple(problem.output_to_inputs)
    coord_to_id = {problem.output_coordinates[output_id]: output_id for output_id in ids}
    return [
        coord_to_id[coordinate]
        for coordinate in morton_order(
            problem.output_coordinates[output_id] for output_id in ids
        )
    ]


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    """Validated reconstruction schedule and diagnostics.

    Attributes
    ----------
    scheduler : str
        Registered scheduler name.
    input_order : tuple[int, ...]
        Validated reconstruction order.
    peak_resident_tiles : int
        Simulated peak live reconstruction count.
    metrics : Mapping[str, Metric]
        Immutable planner telemetry.
    """

    scheduler: str
    input_order: tuple[int, ...]
    peak_resident_tiles: int
    metrics: Mapping[str, Metric]


def _input_to_outputs(
    out_to_in: Mapping[int, Sequence[int]],
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for output_id, inputs in out_to_in.items():
        for tile_id in inputs:
            result[tile_id].append(output_id)
    return dict(result)


def _morton_input_order(problem: DispatchProblem) -> list[int]:
    output_order = morton_output_order(problem)
    rank = {output_id: index for index, output_id in enumerate(output_order)}
    input_to_outputs = _input_to_outputs(problem.output_to_inputs)
    return sorted(
        input_to_outputs,
        key=lambda tile_id: (
            min(rank[output_id] for output_id in input_to_outputs[tile_id]),
            tile_id,
        ),
    )


def _plan_order(
    problem: DispatchProblem,
    _context: SchedulerContext,
) -> Sequence[int]:
    return problem.input_order


def _morton_order(
    problem: DispatchProblem,
    _context: SchedulerContext,
) -> Sequence[int]:
    return _morton_input_order(problem)


def _windowed_graph_ready_order(
    problem: DispatchProblem,
    context: SchedulerContext,
) -> Sequence[int]:
    """Prioritize ready outputs within a Morton look-ahead window.

    Parameters
    ----------
    problem : DispatchProblem
        Validated scheduling problem.
    context : SchedulerContext
        Window and batching dimensions.

    Returns
    -------
    Sequence[int]
        Deterministic input permutation.
    """
    out_to_in = problem.output_to_inputs
    input_to_outputs = _input_to_outputs(out_to_in)
    baseline = _morton_input_order(problem)
    baseline_rank = {tile_id: index for index, tile_id in enumerate(baseline)}
    remaining = {output_id: len(inputs) for output_id, inputs in out_to_in.items()}
    outstanding_uses = Counter(tile_id for inputs in out_to_in.values() for tile_id in inputs)
    unscheduled = list(baseline)
    result: list[int] = []

    while unscheduled:
        best_index = 0
        best_score: tuple[int, ...] | None = None
        current_shape = (
            problem.input_shapes[result[-1]]
            if result and len(result) % context.recon_batch != 0
            else None
        )
        for index, tile_id in enumerate(unscheduled[: context.window]):
            ready_outputs = [
                output_id
                for output_id in input_to_outputs[tile_id]
                if remaining[output_id] == 1
            ]
            ready_uses = Counter(
                contributor
                for output_id in ready_outputs
                for contributor in out_to_in[output_id]
            )
            releasable = sum(
                outstanding_uses[contributor] == uses
                for contributor, uses in ready_uses.items()
            )
            closest_output = min(
                remaining[output_id] for output_id in input_to_outputs[tile_id]
            )
            urgency = sum(
                len(out_to_in[output_id]) - remaining[output_id]
                for output_id in input_to_outputs[tile_id]
            )
            same_shape = int(
                current_shape is not None and problem.input_shapes[tile_id] == current_shape
            )
            score = (
                len(ready_outputs),
                releasable,
                -closest_output,
                same_shape,
                urgency,
                -baseline_rank[tile_id],
                -tile_id,
            )
            if best_score is None or score > best_score:
                best_index = index
                best_score = score

        tile_id = unscheduled.pop(best_index)
        result.append(tile_id)
        for output_id in input_to_outputs[tile_id]:
            remaining[output_id] -= 1
            if remaining[output_id] == 0:
                for contributor in out_to_in[output_id]:
                    outstanding_uses[contributor] -= 1

    return result


_SCHEDULERS: dict[str, SchedulerStrategy] = {}


def register_scheduler(name: str, strategy: SchedulerStrategy) -> None:
    """Register a scheduling strategy.

    Parameters
    ----------
    name : str
        Identifier exposed through configuration.
    strategy : SchedulerStrategy
        Deterministic scheduling callable.

    Raises
    ------
    ValueError
        If ``name`` is not an identifier or is already registered.
    """
    if not name.isidentifier():
        raise ValueError(f"invalid scheduler name {name!r}")
    if name in _SCHEDULERS:
        raise ValueError(f"scheduler already registered: {name}")
    _SCHEDULERS[name] = strategy


register_scheduler("plan", _plan_order)
register_scheduler("morton", _morton_order)
register_scheduler("windowed_graph_ready", _windowed_graph_ready_order)


def scheduler_names() -> tuple[str, ...]:
    """Return registered scheduler names.

    Returns
    -------
    tuple[str, ...]
        Names in registration order.
    """
    return tuple(_SCHEDULERS)


def _validate_order(problem: DispatchProblem, order: Sequence[int]) -> None:
    expected = set(problem.input_order)
    actual = set(order)
    if len(order) != len(actual):
        raise ValueError("dispatch scheduler returned duplicate input tile IDs")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "dispatch scheduler returned an invalid permutation: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def _readiness(
    problem: DispatchProblem,
    input_order: Sequence[int],
) -> tuple[list[int], list[int]]:
    position = {tile_id: index + 1 for index, tile_id in enumerate(input_order)}
    ready_at = {
        output_id: max(position[tile_id] for tile_id in inputs)
        for output_id, inputs in problem.output_to_inputs.items()
        if inputs
    }
    output_order = sorted(
        ready_at,
        key=lambda output_id: (ready_at[output_id], output_id),
    )
    expected_outputs = sum(bool(inputs) for inputs in problem.output_to_inputs.values())
    if len(output_order) != expected_outputs:
        raise ValueError("dispatch scheduler left outputs without a readiness step")
    return output_order, [ready_at[output_id] for output_id in output_order]


def _peak_resident_for_input_order(
    problem: DispatchProblem,
    input_order: Sequence[int],
) -> int:
    out_to_in = problem.output_to_inputs
    input_to_outputs = _input_to_outputs(out_to_in)
    remaining = {output_id: len(inputs) for output_id, inputs in out_to_in.items()}
    outstanding_uses = Counter(tile_id for inputs in out_to_in.values() for tile_id in inputs)
    live: set[int] = set()
    peak = 0
    for tile_id in input_order:
        live.add(tile_id)
        peak = max(peak, len(live))
        for output_id in input_to_outputs[tile_id]:
            remaining[output_id] -= 1
            if remaining[output_id] == 0:
                for contributor in out_to_in[output_id]:
                    outstanding_uses[contributor] -= 1
                    if outstanding_uses[contributor] == 0:
                        live.discard(contributor)
    if live or any(outstanding_uses.values()) or any(remaining.values()):
        raise ValueError("dispatch residency simulation did not consume the full problem")
    return peak


def build_dispatch_plan(
    problem: DispatchProblem,
    scheduler: str,
    *,
    context: SchedulerContext,
) -> DispatchPlan:
    """Build and validate a dispatch plan.

    Parameters
    ----------
    problem : DispatchProblem
        Validated immutable scheduling problem.
    scheduler : str
        Registered scheduler name.
    context : SchedulerContext
        Runtime dimensions supplied to the scheduler.

    Returns
    -------
    DispatchPlan
        Validated order, readiness order, resident peak, and metrics.

    Raises
    ------
    ValueError
        If the scheduler is unknown or violates the permutation, readiness, or
        residency contract.
    """
    try:
        implementation = _SCHEDULERS[scheduler]
    except KeyError as exc:
        raise ValueError(
            f"unknown dispatch scheduler {scheduler!r}; expected one of {scheduler_names()}"
        ) from exc

    input_order = tuple(implementation(problem, context))
    _validate_order(problem, input_order)
    output_order, ready_steps = _readiness(problem, input_order)
    peak = _peak_resident_for_input_order(problem, input_order)
    metrics: dict[str, Metric] = {
        "scheduler": scheduler,
        "window": context.window,
        "recon_batch": context.recon_batch,
        "n_actors": context.n_actors,
        "n_inputs": len(input_order),
        "n_outputs": len(output_order),
        "peak_resident_tiles": peak,
        "first_output_ready_at": ready_steps[0] if ready_steps else 0,
        "last_output_ready_at": ready_steps[-1] if ready_steps else 0,
        "mean_output_ready_at": statistics.fmean(ready_steps) if ready_steps else 0.0,
    }
    return DispatchPlan(
        scheduler=scheduler,
        input_order=input_order,
        peak_resident_tiles=peak,
        metrics=MappingProxyType(metrics),
    )
