from __future__ import annotations

import asyncio
import threading
import time

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from biahub.tile_stitch.dataflow import (
    OutputCapture,
    ReconstructionLoader,
    write_capture,
)


@dataclass(frozen=True)
class _Payload:
    tile_id: int
    nbytes: int = 16


def test_loader_delivers_batches_and_captures_bounded_state():
    loader = ReconstructionLoader(
        lambda tile_id: _Payload(tile_id),
        order=[0, 1, 2, 3],
        depth=2,
        num_workers=2,
        read_timeout_s=1,
    )
    try:
        assert loader.get_batch([0, 1]) == (_Payload(0), _Payload(1))
        assert loader.get_batch([2, 3]) == (_Payload(2), _Payload(3))
        snapshot = loader.snapshot()
        assert snapshot["delivered"] == 4
        assert snapshot["bytes_read"] == 64
        assert snapshot["peak_in_flight"] <= 2
        assert snapshot["peak_buffered_bytes"] <= 32
        assert snapshot["buffered_bytes"] == 0
    finally:
        loader.close()


def test_loader_forward_jump_abandons_stale_reads_without_growing_backlog():
    release_first = threading.Event()

    def read(tile_id: int) -> _Payload:
        if tile_id == 0:
            release_first.wait(timeout=1)
        return _Payload(tile_id)

    loader = ReconstructionLoader(
        read,
        order=list(range(10)),
        depth=2,
        num_workers=2,
        read_timeout_s=2,
    )
    try:
        result: list[_Payload | None] = []

        def consume() -> None:
            result.append(loader.get(8))

        thread = threading.Thread(target=consume)
        thread.start()
        time.sleep(0.05)
        release_first.set()
        thread.join(timeout=2)
        assert result == [_Payload(8)]
        snapshot = loader.snapshot()
        assert snapshot["frontier"] == 9
        assert snapshot["buffered_items"] <= 1
        assert snapshot["peak_buffered_bytes"] <= 32
        assert loader.get(0) is None
    finally:
        release_first.set()
        loader.close()


def test_loader_failure_transfers_none_for_synchronous_fallback():
    def read(tile_id: int) -> _Payload:
        raise ValueError(f"bad tile {tile_id}")

    loader = ReconstructionLoader(read, order=[4], depth=1, read_timeout_s=1)
    try:
        assert loader.get(4) is None
        snapshot = loader.snapshot()
        assert snapshot["failures"] == 1
        assert snapshot["delivered"] == 0
    finally:
        loader.close()


def test_output_capture_preserves_region_and_reports_committed_bytes():
    writes = []

    def write(region, payload):
        writes.append((region, payload))

    region = (slice(2, 3), slice(1, 2), slice(4, 8))
    payload = _Payload(tile_id=9, nbytes=128)

    receipt = write_capture(write, OutputCapture(tile_id=9, region=region, payload=payload))

    assert writes == [(region, payload)]
    assert receipt.tile_id == 9
    assert receipt.bytes_written == 128
    assert receipt.elapsed_s >= 0


def test_loader_preserves_source_dtype():
    source = np.arange(8, dtype=np.float16)
    loader = ReconstructionLoader(
        lambda tile_id: source.copy(),
        order=[0],
        depth=1,
        read_timeout_s=1,
    )
    try:
        delivered = loader.get(0)
        assert delivered is not None
        assert delivered.dtype == np.float16
    finally:
        loader.close()


def test_worker_casts_after_source_dtype_device_transfer(monkeypatch):
    import torch

    from biahub.tile_stitch.monarch.tile_worker import TileWorker

    source = np.arange(8, dtype=np.float16)
    calls = []

    class _DeviceTensor:
        def to(self, dtype):
            calls.append(("to", dtype))
            return self

    def as_tensor(value, **kwargs):
        calls.append(("as_tensor", value, kwargs))
        return _DeviceTensor()

    monkeypatch.setattr(torch, "as_tensor", as_tensor)
    worker = SimpleNamespace(gpu_idx=1, _loader=None, _rs_h2d_bytes=0)

    result = TileWorker._load_one(worker, 7, object(), source)

    assert isinstance(result, _DeviceTensor)
    assert calls == [
        ("as_tensor", source, {"device": "cuda:1"}),
        ("to", torch.float32),
    ]
    assert worker._rs_h2d_bytes == source.nbytes


def test_worker_binds_iohub_output_writer(monkeypatch):
    import iohub.ngff

    from biahub.tile_stitch.monarch.tile_worker import TileWorker

    calls = []
    writes = []

    class _Array:
        def __setitem__(self, region, payload):
            writes.append((region, payload))

    class _Dataset:
        data = _Array()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def open_ome_zarr(path, *, layout, mode):
        calls.append((path, layout, mode))
        return _Dataset()

    monkeypatch.setattr(iohub.ngff, "open_ome_zarr", open_ome_zarr)
    worker = SimpleNamespace(work=SimpleNamespace(output_path="/tmp/output.zarr"))
    writer = TileWorker._make_output_writer(worker)
    region = (slice(0, 1), slice(2, 3), slice(4, 8))
    payload = np.arange(4, dtype=np.float32)

    writer(region, payload)

    assert calls == [("/tmp/output.zarr", "fov", "a")]
    assert len(writes) == 1
    assert writes[0][0] == region
    np.testing.assert_array_equal(writes[0][1], payload)


def test_worker_batches_remote_contributors(monkeypatch):
    import torch

    from biahub.tile_stitch.monarch import tile_worker

    actions = []

    class _Buffer:
        backend = "ibverbs"

        def __init__(self, values):
            self.values = values

    class _Action:
        def __init__(self):
            self.reads = []
            self.timeout = None
            actions.append(self)

        def read_remote(self, destination, source):
            self.reads.append((destination, source))
            return self

        async def submit(self, *, timeout):
            self.timeout = timeout
            for destination, source in self.reads:
                destination.copy_(source.values.view(torch.uint8).flatten())

    monkeypatch.setattr(tile_worker, "RDMAAction", _Action)
    local = torch.tensor([1.0, 2.0], dtype=torch.float16)
    remote_a = torch.tensor([3.0, 4.0], dtype=torch.float16)
    remote_b = torch.tensor([5.0, 6.0], dtype=torch.float16)
    contributors = {
        0: object(),
        1: SimpleNamespace(
            shape=(2,),
            dtype_name="torch.float16",
            buffer=_Buffer(remote_a),
        ),
        2: SimpleNamespace(
            shape=(2,),
            dtype_name="torch.float16",
            buffer=_Buffer(remote_b),
        ),
    }

    arrays, metrics = asyncio.run(
        tile_worker._pull_contributors(
            {0: local},
            contributors,
            timeout_s=17,
        )
    )

    assert len(actions) == 1
    assert len(actions[0].reads) == 2
    assert actions[0].timeout == 17
    np.testing.assert_array_equal(arrays[0], local.numpy())
    np.testing.assert_array_equal(arrays[1], remote_a.numpy())
    np.testing.assert_array_equal(arrays[2], remote_b.numpy())
    assert metrics == {
        "rdma_backend": "ibverbs",
        "rdma_batches": 1,
        "rdma_ops": 2,
        "rdma_bytes": remote_a.nbytes + remote_b.nbytes,
    }


def test_worker_reports_tcp_when_ibverbs_is_forced_off(monkeypatch):
    import monarch

    from biahub.tile_stitch.monarch import tile_worker

    monkeypatch.setattr(
        monarch,
        "get_global_config",
        lambda: {"rdma_disable_ibverbs": True},
    )
    buffer = SimpleNamespace(backend="ibverbs")

    assert tile_worker._rdma_backend_name(buffer) == "tcp"


def test_worker_skips_rdma_batch_for_local_contributors(monkeypatch):
    import torch

    from biahub.tile_stitch.monarch import tile_worker

    class _UnexpectedAction:
        def __init__(self):
            raise AssertionError("local-only output must not create an RDMA action")

    monkeypatch.setattr(tile_worker, "RDMAAction", _UnexpectedAction)
    local = torch.tensor([1.0, 2.0], dtype=torch.float32)

    arrays, metrics = asyncio.run(
        tile_worker._pull_contributors(
            {4: local},
            {4: object()},
            timeout_s=60,
        )
    )

    np.testing.assert_array_equal(arrays[4], local.numpy())
    assert metrics == {
        "rdma_backend": "local",
        "rdma_batches": 0,
        "rdma_ops": 0,
        "rdma_bytes": 0,
    }


def test_worker_propagates_rdma_batch_timeout(monkeypatch):

    from biahub.tile_stitch.monarch import tile_worker

    class _Buffer:
        backend = "ibverbs"

    class _FailingAction:
        def read_remote(self, destination, source):
            del destination, source
            return self

        async def submit(self, *, timeout):
            assert timeout == 9
            raise TimeoutError("batched RDMA timed out")

    monkeypatch.setattr(tile_worker, "RDMAAction", _FailingAction)
    handle = SimpleNamespace(
        shape=(2,),
        dtype_name="torch.float16",
        buffer=_Buffer(),
    )

    with pytest.raises(TimeoutError, match="batched RDMA timed out"):
        asyncio.run(
            tile_worker._pull_contributors(
                {},
                {7: handle},
                timeout_s=9,
            )
        )
