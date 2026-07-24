"""Transport-independent state machine for one tile-stitch work unit."""


import asyncio
import contextlib
import logging
import time

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def build_recon_batches(
    order: Sequence[int],
    tiles_by_id: Mapping[int, Any],
    batch_size: int,
) -> list[list[int]]:
    """Group an input order into same-shape reconstruction batches."""
    from collections import OrderedDict

    by_shape: OrderedDict[tuple, list[int]] = OrderedDict()
    for tile_id in order:
        by_shape.setdefault(tuple(tiles_by_id[tile_id].shape), []).append(tile_id)
    return [
        tile_ids[offset : offset + batch_size]
        for tile_ids in by_shape.values()
        for offset in range(0, len(tile_ids), batch_size)
    ]


class ExecutionTransport(Protocol):
    """Actor operations required by :class:`TimepointExecution`."""

    async def prime(self, assignments: Mapping[int, Sequence[int]]) -> None: ...

    async def reconstruct(
        self,
        gpu: int,
        tile_ids: Sequence[int],
        timeout_s: float,
    ) -> list[Any]: ...

    def dispatch_stitch(self, output_id: int, contributors: Mapping[int, Any]) -> None: ...

    async def receive_stitch(self) -> dict[str, Any]: ...

    async def forget(self, tile_ids: Sequence[int]) -> None: ...


class ResidentGate:
    """Reserve reconstructed tiles until their final consumer exits."""

    def __init__(self, budget: int) -> None:
        self._free = budget
        self._cond = asyncio.Condition()

    @property
    def free(self) -> int:
        """Currently available tile slots."""
        return self._free

    async def acquire(self, count: int) -> None:
        """Reserve ``count`` tile slots atomically."""
        async with self._cond:
            while self._free < count:
                await self._cond.wait()
            self._free -= count

    async def release(self, count: int = 1) -> None:
        """Return tile slots and wake blocked reconstruction work."""
        async with self._cond:
            self._free += count
            self._cond.notify_all()


@dataclass(slots=True)
class ExecutionState:
    """Synchronous readiness, dispatch, and contributor-lifetime state."""

    output_to_inputs: Mapping[int, Sequence[int]]
    handles: dict[int, Any] = field(default_factory=dict)
    pending: set[int] = field(init=False)
    queued: set[int] = field(default_factory=set)
    in_flight: set[int] = field(default_factory=set)
    completed: set[int] = field(default_factory=set)
    _input_to_outputs: dict[int, tuple[int, ...]] = field(init=False)
    _remaining_consumers: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        input_to_outputs: dict[int, list[int]] = defaultdict(list)
        for output_id, input_ids in self.output_to_inputs.items():
            for tile_id in input_ids:
                input_to_outputs[tile_id].append(output_id)
        self._input_to_outputs = {
            tile_id: tuple(output_ids) for tile_id, output_ids in input_to_outputs.items()
        }
        self._remaining_consumers = {
            tile_id: len(output_ids) for tile_id, output_ids in self._input_to_outputs.items()
        }
        self.pending = set(self.output_to_inputs)

    @property
    def active_outputs(self) -> int:
        """Number of queued or actor-executing stitches."""
        return len(self.queued) + len(self.in_flight)

    def add_reconstructions(self, reconstructed: Mapping[int, Any]) -> list[int]:
        """Store newly reconstructed handles and reserve newly ready outputs."""
        duplicate = self.handles.keys() & reconstructed.keys()
        if duplicate:
            raise RuntimeError(f"duplicate reconstructed tile IDs: {sorted(duplicate)}")
        self.handles.update(reconstructed)
        candidates = {
            output_id
            for tile_id in reconstructed
            for output_id in self._input_to_outputs.get(tile_id, ())
        }
        ready = sorted(
            output_id
            for output_id in candidates
            if output_id in self.pending
            and set(self.output_to_inputs[output_id]) <= self.handles.keys()
        )
        for output_id in ready:
            self.pending.remove(output_id)
            self.queued.add(output_id)
        return ready

    def dispatch(self, output_id: int) -> dict[int, Any]:
        """Move a reserved output to actor execution and return its handles."""
        if output_id not in self.queued:
            raise RuntimeError(f"output {output_id} is not queued")
        self.queued.remove(output_id)
        self.in_flight.add(output_id)
        return {tile_id: self.handles[tile_id] for tile_id in self.output_to_inputs[output_id]}

    def complete(self, output_id: int) -> list[int]:
        """Complete one output and release final-consumer reconstruction handles."""
        if output_id not in self.in_flight:
            raise RuntimeError(f"output {output_id} completed without dispatch")
        self.in_flight.remove(output_id)
        if output_id in self.completed:
            raise RuntimeError(f"output {output_id} completed twice")
        self.completed.add(output_id)
        released: list[int] = []
        for tile_id in self.output_to_inputs[output_id]:
            remaining = self._remaining_consumers[tile_id] - 1
            self._remaining_consumers[tile_id] = remaining
            if remaining == 0:
                self.handles.pop(tile_id)
                released.append(tile_id)
        return released


class TimepointExecution:
    """Run one validated reconstruction schedule through an actor transport."""

    def __init__(
        self,
        *,
        program,
        input_order: Sequence[int],
        resident_budget: int | None,
        recon_batch: int,
        n_actors: int,
        window: int,
        transport: ExecutionTransport,
        heartbeat_s: float = 0.0,
    ) -> None:
        self.program = program
        self.input_order = tuple(input_order)
        self.resident_budget = resident_budget
        self.recon_batch = recon_batch
        self.n_actors = n_actors
        self.window = window
        self.transport = transport
        self.heartbeat_s = heartbeat_s
        self.state = ExecutionState(program.output_to_inputs)
        self.gate = ResidentGate(resident_budget) if resident_budget is not None else None
        self._stitch_slots = asyncio.Semaphore(window)
        self._released: list[int] = []
        self._summaries: list[dict[str, Any]] = []
        self._reconstruction_remaining = len(self.input_order)
        self._reconstruction_done = False
        self._stage_a_started = 0.0
        self._stage_a_s = 0.0
        config = program.monarch
        self._max_inflight = config.recon_max_inflight_per_gpu
        self._rpc_timeout_s = float(config.recon_rpc_timeout_s)
        self._rpc_retries = config.recon_rpc_retries

    async def _flush_released(self) -> None:
        if not self._released:
            return
        tile_ids = tuple(self._released)
        self._released.clear()
        await self.transport.forget(tile_ids)

    async def _receive_one(self) -> None:
        summary = await self.transport.receive_stitch()
        output_id = summary.get("out_tile_id")
        if output_id is None:
            raise RuntimeError("stitch completion omitted out_tile_id")
        released = self.state.complete(output_id)
        self._released.extend(released)
        if self.gate is not None and released:
            await self.gate.release(len(released))
        self._summaries.append(summary)
        self._stitch_slots.release()
        if len(self._released) >= 32:
            await self._flush_released()

    async def _completion_loop(self) -> None:
        while not self._reconstruction_done or self.state.active_outputs:
            if self.state.in_flight:
                await self._receive_one()
            else:
                await asyncio.sleep(0)

    async def _dispatch_ready(self, output_ids: Sequence[int]) -> None:
        for output_id in output_ids:
            await self._stitch_slots.acquire()
            contributors = self.state.dispatch(output_id)
            self.transport.dispatch_stitch(output_id, contributors)

    async def _reconstruct_rpc(self, tile_ids: Sequence[int], gpu: int) -> list[Any]:
        for attempt in range(self._rpc_retries + 1):
            actor = (gpu + attempt) % self.n_actors
            try:
                return await self.transport.reconstruct(
                    actor,
                    tile_ids,
                    self._rpc_timeout_s,
                )
            except TimeoutError:
                logger.warning(
                    "reconstruction timed out on gpu=%d (attempt %d/%d)",
                    actor,
                    attempt + 1,
                    self._rpc_retries + 1,
                )
        raise TimeoutError(f"reconstruction stuck after {self._rpc_retries + 1} attempts")

    async def _reconstruct(self, tile_ids: Sequence[int], gpu: int) -> None:
        if self.gate is not None:
            await self.gate.acquire(len(tile_ids))
        handles = await self._reconstruct_rpc(tile_ids, gpu)
        if len(handles) != len(tile_ids):
            raise RuntimeError(
                f"actor returned {len(handles)} handles for {len(tile_ids)} tiles"
            )
        ready = self.state.add_reconstructions(dict(zip(tile_ids, handles, strict=True)))
        self._reconstruction_remaining -= len(tile_ids)
        if self._reconstruction_remaining == 0:
            self._stage_a_s = time.monotonic() - self._stage_a_started
        await self._dispatch_ready(ready)

    async def _dispatch_limited(
        self,
        semaphore: asyncio.Semaphore | None,
        tile_ids: Sequence[int],
        gpu: int,
    ) -> None:
        if semaphore is None:
            await self._reconstruct(tile_ids, gpu)
            return
        async with semaphore:
            await self._reconstruct(tile_ids, gpu)

    def _work_units(self, tiles_by_id: Mapping[int, Any]) -> list[tuple[int, ...]]:
        if self.recon_batch <= 1:
            return [(tile_id,) for tile_id in self.input_order]
        return [
            tuple(batch)
            for batch in build_recon_batches(
                self.input_order,
                tiles_by_id,
                self.recon_batch,
            )
        ]

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_s)
            logger.info(
                "DRIVE hb: remaining=%d in_flight=%d gate_free=%s held=%d pending=%d done=%d",
                self._reconstruction_remaining,
                len(self.state.in_flight),
                self.gate.free if self.gate is not None else "unbounded",
                len(self.state.handles),
                len(self.state.pending),
                len(self._summaries),
            )

    async def run(self) -> tuple[float, float, list[dict[str, Any]]]:
        """Execute reconstruction and stitching to a fully drained state."""
        tiles_by_id = {tile.tile_id: tile for tile in self.program.input_tiles}
        work_units = self._work_units(tiles_by_id)
        assignments: dict[int, list[int]] = {gpu: [] for gpu in range(self.n_actors)}
        for index, tile_ids in enumerate(work_units):
            assignments[index % self.n_actors].extend(tile_ids)
        await self.transport.prime(assignments)

        limit = (
            asyncio.Semaphore(max(self._max_inflight * self.n_actors, self.recon_batch))
            if self.gate is not None and self._max_inflight > 0
            else None
        )
        self._stage_a_started = time.monotonic()
        completion = asyncio.create_task(self._completion_loop())
        heartbeat = asyncio.create_task(self._heartbeat()) if self.heartbeat_s > 0 else None
        tasks = [
            asyncio.create_task(self._dispatch_limited(limit, tile_ids, index % self.n_actors))
            for index, tile_ids in enumerate(work_units)
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            self._reconstruction_done = True
            completion.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await completion
            raise
        else:
            self._reconstruction_done = True
            await completion
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat

        await self._flush_released()
        if self._reconstruction_remaining:
            raise RuntimeError(
                f"execution ended with {self._reconstruction_remaining} reconstructions pending"
            )
        nonempty_outputs = {
            output_id
            for output_id, input_ids in self.program.output_to_inputs.items()
            if input_ids
        }
        if self.state.completed != nonempty_outputs:
            missing = sorted(nonempty_outputs - self.state.completed)
            raise RuntimeError(f"execution ended with undispatched outputs: {missing}")
        return (
            self._stage_a_s,
            time.monotonic() - self._stage_a_started,
            list(self._summaries),
        )
