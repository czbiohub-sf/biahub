"""Transport-independent contracts for the work-unit execution state machine."""

import asyncio

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from biahub.settings import MonarchConfig
from biahub.tile_stitch.monarch.execution import ExecutionState, TimepointExecution


@dataclass(frozen=True)
class _Tile:
    tile_id: int
    shape: tuple[int, ...] = (1, 4, 4, 4)


class _FakeTransport:
    """Deterministic actor transport with controllable out-of-order completion."""

    def __init__(self) -> None:
        self.prime_assignments = None
        self.reconstruction_order: list[int] = []
        self.dispatched: list[int] = []
        self.forgotten: list[int] = []
        self._completions: asyncio.Queue[dict] = asyncio.Queue()

    async def prime(self, assignments) -> None:
        self.prime_assignments = {gpu: tuple(ids) for gpu, ids in assignments.items()}

    async def reconstruct(self, gpu, tile_ids, timeout_s):
        del gpu, timeout_s
        assert self.prime_assignments is not None
        await asyncio.sleep((4 - tile_ids[0]) * 0.001)
        self.reconstruction_order.extend(tile_ids)
        return [f"handle-{tile_id}" for tile_id in tile_ids]

    def dispatch_stitch(self, output_id, contributors) -> None:
        self.dispatched.append(output_id)

        async def complete() -> None:
            await asyncio.sleep((2 - output_id) * 0.001)
            await self._completions.put(
                {
                    "out_tile_id": output_id,
                    "n_inputs": len(contributors),
                    "wall_s": 0.0,
                }
            )

        asyncio.create_task(complete())

    async def receive_stitch(self):
        return await self._completions.get()

    async def forget(self, tile_ids) -> None:
        self.forgotten.extend(tile_ids)


def _program():
    return SimpleNamespace(
        input_tiles=tuple(_Tile(tile_id) for tile_id in range(4)),
        output_to_inputs={0: (0, 1), 1: (1, 2), 2: (2, 3)},
        monarch=MonarchConfig(
            recon_batch=1,
            recon_rpc_retries=0,
            recon_max_inflight_per_gpu=0,
        ),
    )


def test_out_of_order_reconstructions_reserve_outputs_once():
    state = ExecutionState({0: (0, 1), 1: (1, 2)})

    assert state.add_reconstructions({2: "h2"}) == []
    assert state.add_reconstructions({1: "h1"}) == [1]
    assert state.add_reconstructions({0: "h0"}) == [0]
    assert state.pending == set()
    assert state.queued == {0, 1}


def test_output_dispatch_and_completion_are_exactly_once():
    state = ExecutionState({7: (1,)})
    assert state.add_reconstructions({1: "h1"}) == [7]
    assert state.dispatch(7) == {1: "h1"}

    with pytest.raises(RuntimeError, match="not queued"):
        state.dispatch(7)
    assert state.complete(7) == [1]
    with pytest.raises(RuntimeError, match="without dispatch"):
        state.complete(7)


def test_reconstruction_releases_only_after_final_consumer():
    state = ExecutionState({0: (3,), 1: (3, 4)})
    state.add_reconstructions({3: "h3", 4: "h4"})
    state.dispatch(0)
    state.dispatch(1)

    assert state.complete(0) == []
    assert 3 in state.handles
    assert state.complete(1) == [3, 4]
    assert state.handles == {}


def test_fake_transport_drives_full_lifecycle_out_of_order():
    async def run():
        transport = _FakeTransport()
        execution = TimepointExecution(
            program=_program(),
            input_order=(0, 1, 2, 3),
            resident_budget=2,
            recon_batch=1,
            n_actors=2,
            window=2,
            transport=transport,
        )
        result = await execution.run()
        return transport, execution, result

    transport, execution, (_, _, summaries) = asyncio.run(run())
    assert transport.reconstruction_order != [0, 1, 2, 3]
    assert sorted(transport.dispatched) == [0, 1, 2]
    assert len(transport.dispatched) == len(set(transport.dispatched))
    assert sorted(summary["out_tile_id"] for summary in summaries) == [0, 1, 2]
    assert sorted(transport.forgotten) == [0, 1, 2, 3]
    assert execution.state.handles == {}
    assert execution.state.completed == {0, 1, 2}


def test_stitch_failure_propagates_without_hanging_execution():
    class _FailingTransport:
        async def prime(self, assignments) -> None:
            self.assignments = assignments

        async def reconstruct(self, gpu, tile_ids, timeout_s):
            del gpu, timeout_s
            assert self.assignments
            return [f"handle-{tile_id}" for tile_id in tile_ids]

        def dispatch_stitch(self, output_id, contributors) -> None:
            assert output_id == 0
            assert contributors == {0: "handle-0"}

        async def receive_stitch(self):
            raise RuntimeError("stitch actor failed")

        async def forget(self, tile_ids) -> None:
            raise AssertionError(f"failed output must not release {tile_ids}")

    program = SimpleNamespace(
        input_tiles=(_Tile(0),),
        output_to_inputs={0: (0,)},
        monarch=MonarchConfig(
            recon_batch=1,
            recon_rpc_retries=0,
            recon_max_inflight_per_gpu=1,
        ),
    )

    async def run():
        execution = TimepointExecution(
            program=program,
            input_order=(0,),
            resident_budget=2,
            recon_batch=1,
            n_actors=1,
            window=1,
            transport=_FailingTransport(),
        )
        await execution.run()

    with pytest.raises(RuntimeError, match="stitch actor failed"):
        asyncio.run(run())


def test_backend_teardown_uses_public_shutdown(monkeypatch):
    import monarch.actor

    from biahub.tile_stitch.monarch.backend import MonarchBackend

    calls = []

    class _Shutdown:
        def get(self, *, timeout):
            calls.append(timeout)

    monkeypatch.setattr(monarch.actor, "shutdown_context", _Shutdown)
    backend = MonarchBackend()
    backend._workers = object()
    backend._procs = object()

    backend.teardown()

    assert backend._workers is None
    assert backend._procs is None
    assert len(calls) == 1
