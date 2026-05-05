"""Pipelined drivers — verbatim port of legacy drive_pipelined_v6 and _v7.

Both are interleaved Stage A↔B via dask.distributed.as_completed: stitch
fires per output as its contributor batches/tiles complete. No Stage A
barrier.
"""

import logging
import time

from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def drive_pipelined_v6(
    *,
    client: Any,
    plan_path: str,
    plan,  # biahub.tile_stitch.plan.RunPlan
    retries: int,
) -> dict[str, Any]:
    """v6 CPU pipeline — K input tiles per Stage A batch, per-output stitch.

    Port of drivers/v4_dask.py:drive_pipelined_v6 minus the wandb/stages
    instrumentation. Returns a summary dict.
    """
    from distributed import as_completed

    from biahub.tile_stitch.workers import (
        reconstruct_batch_memory,
        stitch_output_tile_v6,
    )

    n_inputs = len(plan.input_tiles)
    batches = plan.input_batches or []
    o2b = plan.output_to_batches or {}
    n_batches = len(batches)
    n_outputs = len(plan.output_tiles)

    logger.info(
        "v6 pipeline: %d batches (%d total tiles, avg %.1f tiles/batch); avg %.1f batches/output → %d stitches",
        n_batches,
        n_inputs,
        n_inputs / max(n_batches, 1),
        sum(len(v) for v in o2b.values()) / max(len(o2b), 1),
        n_outputs,
    )

    t_submit = time.monotonic()
    batch_futs = {
        bidx: client.submit(
            reconstruct_batch_memory,
            bidx,
            plan_path=plan_path,
            priority=n_batches - bidx,
            retries=retries,
            pure=False,
            key=f"batch-{bidx}",
        )
        for bidx in range(n_batches)
    }
    fut_to_bidx = {f.key: bidx for bidx, f in batch_futs.items()}
    logger.info("submitted %d batch futures in %.1fs", n_batches, time.monotonic() - t_submit)

    waiting_for: dict[int, set[int]] = defaultdict(set)
    remaining_batches: dict[int, set[int]] = {}
    for ot_id, bidxs in o2b.items():
        s = set(bidxs)
        remaining_batches[ot_id] = s
        for bidx in s:
            waiting_for[bidx].add(ot_id)

    a_done = 0
    a_failures: list[str] = []
    b_done = 0
    b_results: list[dict] = []
    b_failures: list[str] = []
    stitch_futs: dict[str, int] = {}

    t_start = time.monotonic()
    t_a_complete: float | None = None
    t_b_first: float | None = None
    last_log = t_start
    log_every = 30.0

    pool = as_completed(list(batch_futs.values()), with_results=False)

    while a_done < n_batches or b_done < n_outputs:
        try:
            fut = next(pool)
        except StopIteration:
            break

        key = fut.key
        if key in fut_to_bidx:
            bidx = fut_to_bidx[key]
            if fut.status == "error":
                a_failures.append(key)
                a_done += 1
                logger.error("batch-%d failed: %s", bidx, fut.exception() or "unknown")
            else:
                a_done += 1
                for ot_id in waiting_for[bidx]:
                    remaining_batches[ot_id].discard(bidx)
                    if not remaining_batches[ot_id]:
                        needed = {cb: batch_futs[cb] for cb in o2b[ot_id]}
                        sf = client.submit(
                            stitch_output_tile_v6,
                            ot_id,
                            plan_path=plan_path,
                            batches=needed,
                            priority=0,
                            retries=retries,
                            pure=False,
                            key=f"stitch-{ot_id}",
                        )
                        stitch_futs[sf.key] = ot_id
                        pool.add(sf)
                        if t_b_first is None:
                            t_b_first = time.monotonic()
        else:
            ot_id = stitch_futs.pop(key, None)
            try:
                r = fut.result()
                b_results.append(r)
                b_done += 1
            except Exception as e:
                b_failures.append(key)
                b_done += 1
                logger.error("stitch-%s failed: %s", ot_id, e)

        if a_done == n_batches and t_a_complete is None:
            t_a_complete = time.monotonic()

        now = time.monotonic()
        if now - last_log >= log_every or (a_done == n_batches and b_done == n_outputs):
            elapsed = now - t_start
            n_hosts = len({r.get("host") for r in b_results if r.get("host")})
            logger.info(
                "v6 progress: a=%d/%d b=%d/%d failed_a=%d failed_b=%d hosts(b)=%d elapsed=%.0fs",
                a_done,
                n_batches,
                b_done,
                n_outputs,
                len(a_failures),
                len(b_failures),
                n_hosts,
                elapsed,
            )
            last_log = now

    a_wall = (t_a_complete or time.monotonic()) - t_start
    b_wall = time.monotonic() - (t_b_first or t_start)

    return {
        "stage_a_wall_s": a_wall,
        "stage_b_wall_s": b_wall,
        "stage_a_count": a_done - len(a_failures),
        "stage_b_count": b_done - len(b_failures),
        "a_failures": a_failures,
        "b_failures": b_failures,
        "wall_s": time.monotonic() - t_start,
    }


def drive_pipelined_v7(
    *,
    client: Any,
    plan_path: str,
    plan,
    retries: int,
) -> dict[str, Any]:
    """v7 GPU pipeline — per-tile Stage A returning cupy → GPU stitch."""
    from distributed import as_completed

    from biahub.tile_stitch.workers import (
        reconstruct_tile_memory_gpu,
        stitch_output_tile_v7,
    )

    n_inputs = len(plan.input_tiles)
    n_outputs = len(plan.output_tiles)
    prio = (
        {tid: n_inputs - i for i, tid in enumerate(plan.input_order)}
        if plan.input_order
        else {t.tile_id: 0 for t in plan.input_tiles}
    )

    logger.info("v7 GPU pipeline: %d recons → %d stitches", n_inputs, n_outputs)

    recon_futs = {
        t.tile_id: client.submit(
            reconstruct_tile_memory_gpu,
            t.tile_id,
            plan_path=plan_path,
            priority=prio[t.tile_id],
            retries=retries,
            pure=False,
            key=f"recon-{t.tile_id}",
        )
        for t in plan.input_tiles
    }
    fut_to_tid = {f.key: tid for tid, f in recon_futs.items()}

    waiting_for: dict[int, set[int]] = defaultdict(set)
    remaining: dict[int, set[int]] = {}
    for ot in plan.output_tiles:
        deps = set(plan.output_to_inputs[ot.tile_id])
        remaining[ot.tile_id] = deps
        for tid in deps:
            waiting_for[tid].add(ot.tile_id)

    a_done = 0
    a_failures: list[str] = []
    b_results: list[dict] = []
    b_failures: list[str] = []
    b_done = 0
    stitch_futs: dict[str, int] = {}

    t_start = time.monotonic()
    last_log = t_start
    log_every = 30.0

    pool = as_completed(list(recon_futs.values()), with_results=False)

    while a_done < n_inputs or b_done < n_outputs:
        try:
            fut = next(pool)
        except StopIteration:
            break

        key = fut.key
        if key in fut_to_tid:
            tid = fut_to_tid[key]
            if fut.status == "error":
                a_failures.append(key)
                a_done += 1
            else:
                a_done += 1
                for ot_id in waiting_for[tid]:
                    remaining[ot_id].discard(tid)
                    if not remaining[ot_id]:
                        contribs = {
                            cid: recon_futs[cid] for cid in plan.output_to_inputs[ot_id]
                        }
                        sf = client.submit(
                            stitch_output_tile_v7,
                            ot_id,
                            plan_path=plan_path,
                            contributors=contribs,
                            priority=0,
                            retries=retries,
                            pure=False,
                            key=f"stitch-{ot_id}",
                        )
                        stitch_futs[sf.key] = ot_id
                        pool.add(sf)
        else:
            ot_id = stitch_futs.pop(key, None)
            try:
                r = fut.result()
                b_results.append(r)
                b_done += 1
            except Exception as e:
                b_failures.append(key)
                b_done += 1
                logger.error("stitch-%s failed: %s", ot_id, e)

        now = time.monotonic()
        if now - last_log >= log_every or (a_done == n_inputs and b_done == n_outputs):
            logger.info(
                "v7 progress: a=%d/%d b=%d/%d failed_a=%d failed_b=%d elapsed=%.0fs",
                a_done,
                n_inputs,
                b_done,
                n_outputs,
                len(a_failures),
                len(b_failures),
                time.monotonic() - t_start,
            )
            last_log = now

    return {
        "stage_a_count": a_done - len(a_failures),
        "stage_b_count": b_done - len(b_failures),
        "a_failures": a_failures,
        "b_failures": b_failures,
        "wall_s": time.monotonic() - t_start,
    }
