"""Bounded read-ahead and output-capture primitives for tile stitching."""

import concurrent.futures
import logging
import threading
import time

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any, Self

logger = logging.getLogger("tile_stitch.dataflow")


@cache
def _open_input_array(input_path: str):
    """Open and cache one source's level-zero array."""
    from iohub.ngff import open_ome_zarr

    return open_ome_zarr(input_path, layout="fov", mode="r").data


def read_zarr_tile(program, work, tile):
    """Read one spatial source tile with leading axes removed."""
    import numpy as np

    array = _open_input_array(work.input_path)
    region = (
        slice(work.timepoint, work.timepoint + 1),
        slice(work.channel_idx, work.channel_idx + 1),
    ) + tuple(tile.slices[dim] for dim in program.tile_dims)
    return np.asarray(array[region]).squeeze(axis=(0, 1))


@dataclass(frozen=True, slots=True)
class OutputCapture[T]:
    """A complete output payload and its destination region."""

    tile_id: int
    region: tuple[slice, ...]
    payload: T


@dataclass(frozen=True, slots=True)
class WriteReceipt:
    """Logical byte count and elapsed time for one completed output write."""

    tile_id: int
    bytes_written: int
    elapsed_s: float


def write_capture[T](
    write: Callable[[tuple[slice, ...], T], None],
    capture: OutputCapture[T],
) -> WriteReceipt:
    """Commit one capture through a write callable and record its cost."""
    started = time.monotonic()
    write(capture.region, capture.payload)
    return WriteReceipt(
        tile_id=capture.tile_id,
        bytes_written=int(getattr(capture.payload, "nbytes", 0)),
        elapsed_s=time.monotonic() - started,
    )


@dataclass(slots=True)
class LoaderStats:
    """Mutable counters for one reconstruction loader."""

    submitted: int = 0
    completed: int = 0
    delivered: int = 0
    failures: int = 0
    retries: int = 0
    bytes_read: int = 0
    wait_s: float = 0.0
    buffered_bytes: int = 0
    peak_buffered_bytes: int = 0
    peak_in_flight: int = 0


@dataclass(slots=True)
class _Entry[T]:
    value: T | None
    error: BaseException | None
    nbytes: int


class ReconstructionLoader[T]:
    """Read tiles through one bounded executor in reconstruction order.

    ``depth=0`` disables speculative read-ahead but still routes requested reads
    through the same timeout and retry lifecycle. A timed-out attempt rotates the
    executor because Python cannot cancel a running storage call.
    """

    def __init__(
        self,
        read: Callable[[int], T],
        order: Sequence[int],
        depth: int,
        *,
        num_workers: int = 1,
        read_timeout_s: float = 600.0,
        retries: int = 0,
    ) -> None:
        if len(order) != len(set(order)):
            raise ValueError("reconstruction loader order contains duplicate tile IDs")
        if depth < 0:
            raise ValueError("reconstruction loader depth must be nonnegative")
        if num_workers <= 0:
            raise ValueError("reconstruction loader num_workers must be positive")
        if read_timeout_s <= 0:
            raise ValueError("reconstruction loader read_timeout_s must be positive")
        if retries < 0:
            raise ValueError("reconstruction loader retries must be nonnegative")

        self._read = read
        self._order = tuple(order)
        self._index = {tile_id: index for index, tile_id in enumerate(order)}
        self._depth = depth
        self._read_timeout_s = read_timeout_s
        self._retries = retries
        self._num_workers = min(num_workers, max(1, depth))
        self._generation = 0
        self._executor = self._new_executor()
        self._cv = threading.Condition()
        self._entries: dict[int, _Entry[T]] = {}
        self._futures: dict[concurrent.futures.Future[T], tuple[int, int, int]] = {}
        self._consumed: set[int] = set()
        self._frontier = 0
        self._next_submit = 0
        self._requested_until = 0
        self._stopped = False
        self._closed = False
        self._stats = LoaderStats()
        with self._cv:
            self._fill_locked()

    def _new_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=self._num_workers,
            thread_name_prefix="tile-source",
        )

    def _active_ids_locked(self) -> set[int]:
        return {tile_id for tile_id, _, _ in self._futures.values()}

    def _submit_locked(self, tile_id: int, index: int) -> None:
        future = self._executor.submit(self._read, tile_id)
        self._futures[future] = (tile_id, index, self._generation)
        self._stats.submitted += 1
        self._stats.peak_in_flight = max(
            self._stats.peak_in_flight,
            len(self._futures),
        )
        future.add_done_callback(self._complete)

    def _fill_locked(self) -> None:
        if self._stopped:
            return
        window_end = min(
            len(self._order),
            max(self._frontier + self._depth, self._requested_until),
        )
        active_ids = self._active_ids_locked()
        while self._next_submit < window_end and len(self._futures) < self._num_workers:
            index = self._next_submit
            self._next_submit += 1
            tile_id = self._order[index]
            if index in self._consumed or tile_id in self._entries or tile_id in active_ids:
                continue
            self._submit_locked(tile_id, index)
            active_ids.add(tile_id)

    def _complete(self, future: concurrent.futures.Future[T]) -> None:
        try:
            value = future.result()
            error = None
            nbytes = int(getattr(value, "nbytes", 0))
        except BaseException as exc:
            value = None
            error = exc
            nbytes = 0

        with self._cv:
            info = self._futures.pop(future, None)
            if info is None or self._stopped:
                return
            tile_id, index, generation = info
            if generation != self._generation or index < self._frontier:
                self._fill_locked()
                self._cv.notify_all()
                return
            self._entries[tile_id] = _Entry(value=value, error=error, nbytes=nbytes)
            self._stats.completed += 1
            self._stats.bytes_read += nbytes
            self._stats.buffered_bytes += nbytes
            self._stats.peak_buffered_bytes = max(
                self._stats.peak_buffered_bytes,
                self._stats.buffered_bytes,
            )
            if error is not None:
                self._stats.failures += 1
            self._fill_locked()
            self._cv.notify_all()

    def _rotate_executor_locked(self, missing: set[int]) -> None:
        old = self._executor
        lost_indices = [index for _, index, _ in self._futures.values()]
        self._generation += 1
        self._futures.clear()
        self._executor = self._new_executor()
        old.shutdown(wait=False, cancel_futures=True)
        if lost_indices:
            self._next_submit = min(self._next_submit, min(lost_indices))
        self._next_submit = min(
            self._next_submit,
            min(self._index[tile_id] for tile_id in missing),
        )
        self._stats.retries += len(missing)
        self._fill_locked()

    def get_batch(self, tile_ids: Sequence[int]) -> tuple[T | None, ...]:
        """Transfer requested values in order, returning ``None`` on final failure."""
        if not tile_ids:
            return ()
        unknown = [tile_id for tile_id in tile_ids if tile_id not in self._index]
        if unknown:
            raise KeyError(f"unknown reconstruction tile IDs: {unknown}")

        started = time.monotonic()
        with self._cv:
            if self._stopped:
                return tuple(None for _ in tile_ids)
            indices = [self._index[tile_id] for tile_id in tile_ids]
            first_index = min(indices)
            if first_index > self._frontier:
                self._abandon_before_locked(first_index)
            self._requested_until = max(self._requested_until, max(indices) + 1)
            self._fill_locked()

            attempts = 0
            while True:
                missing = set(tile_ids).difference(self._entries)
                if not missing or self._stopped:
                    break
                deadline = time.monotonic() + self._read_timeout_s
                while missing and not self._stopped:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)
                    missing = set(tile_ids).difference(self._entries)
                if not missing or self._stopped:
                    break
                if attempts >= self._retries:
                    logger.warning(
                        "reconstruction loader timed out waiting for tiles %s; disabling reads",
                        sorted(missing),
                    )
                    self._stopped = True
                    break
                attempts += 1
                self._rotate_executor_locked(missing)

            values: list[T | None] = []
            for tile_id in tile_ids:
                entry = self._entries.pop(tile_id, None)
                index = self._index[tile_id]
                self._consumed.add(index)
                if entry is None:
                    values.append(None)
                    continue
                self._stats.buffered_bytes -= entry.nbytes
                if entry.error is not None:
                    logger.warning("tile read failed for %s: %s", tile_id, entry.error)
                    values.append(None)
                    continue
                self._stats.delivered += 1
                values.append(entry.value)

            self._advance_frontier_locked()
            self._stats.wait_s += time.monotonic() - started
            self._fill_locked()
            self._cv.notify_all()
            return tuple(values)

    def get(self, tile_id: int) -> T | None:
        """Transfer one requested tile."""
        return self.get_batch((tile_id,))[0]

    def _abandon_before_locked(self, index: int) -> None:
        for stale_index in range(self._frontier, index):
            self._consumed.add(stale_index)
            stale_id = self._order[stale_index]
            entry = self._entries.pop(stale_id, None)
            if entry is not None:
                self._stats.buffered_bytes -= entry.nbytes
        self._next_submit = max(self._next_submit, index)
        self._advance_frontier_locked()

    def _advance_frontier_locked(self) -> None:
        while self._frontier in self._consumed:
            self._consumed.remove(self._frontier)
            self._frontier += 1

    def snapshot(self) -> dict[str, Any]:
        """Return counters and current queue ownership."""
        with self._cv:
            result = asdict(self._stats)
            result.update(
                {
                    "depth": self._depth,
                    "frontier": self._frontier,
                    "buffered_items": len(self._entries),
                    "in_flight": len(self._futures),
                    "stopped": self._stopped,
                }
            )
            return result

    def close(self) -> None:
        """Stop new reads and release loader-owned values."""
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._stopped = True
            self._entries.clear()
            self._futures.clear()
            self._stats.buffered_bytes = 0
            self._cv.notify_all()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
