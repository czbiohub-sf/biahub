"""Source-read deadlines, retries, terminal failures, and depth-zero reads."""

import threading
import time

from biahub.tile_stitch.dataflow import ReconstructionLoader


def test_loader_retries_transient_stall_on_fresh_executor():
    calls = 0

    def read(tile_id: int) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            time.sleep(0.4)
            return "late"
        return f"tile-{tile_id}"

    loader = ReconstructionLoader(
        read,
        order=[7],
        depth=1,
        read_timeout_s=0.1,
        retries=1,
    )
    try:
        assert loader.get(7) == "tile-7"
        assert calls == 2
        assert loader.snapshot()["retries"] == 1
    finally:
        loader.close()


def test_loader_persistent_stall_stops_after_retry_budget():
    release = threading.Event()
    calls = 0

    def read(tile_id: int) -> int:
        nonlocal calls
        calls += 1
        release.wait(timeout=30)
        return tile_id

    loader = ReconstructionLoader(
        read,
        order=[7],
        depth=1,
        read_timeout_s=0.05,
        retries=1,
    )
    try:
        assert loader.get(7) is None
        snapshot = loader.snapshot()
        assert calls == 2
        assert snapshot["retries"] == 1
        assert snapshot["stopped"] is True
    finally:
        release.set()
        loader.close()


def test_loader_real_error_is_terminal_without_retry():
    calls = 0

    def read(tile_id: int) -> int:
        nonlocal calls
        calls += 1
        raise ValueError(f"bad tile {tile_id}")

    loader = ReconstructionLoader(
        read,
        order=[3],
        depth=1,
        read_timeout_s=1,
        retries=3,
    )
    try:
        assert loader.get(3) is None
        snapshot = loader.snapshot()
        assert calls == 1
        assert snapshot["failures"] == 1
        assert snapshot["retries"] == 0
    finally:
        loader.close()


def test_depth_zero_reads_only_after_request():
    calls: list[int] = []
    loader = ReconstructionLoader(
        lambda tile_id: calls.append(tile_id) or tile_id,
        order=[1, 2],
        depth=0,
        read_timeout_s=1,
    )
    try:
        time.sleep(0.02)
        assert calls == []
        assert loader.get(1) == 1
        assert calls == [1]
    finally:
        loader.close()
