"""biahub tile-stitch — unified Monarch driver.

Builds the run scaffold (TP resolve/shard, engine plan, output zarr, per-TP
plan pickles) and delegates the mesh bring-up + per-TP pipelined drive + volume
swap to :class:`MonarchBackend`. Monarch is the only distributed backend; the
durable engine knobs live in ``config.monarch`` (``MonarchConfig``) and the
SLURM-/runtime-dependent topology comes from CLI flags.
"""

import json
import logging
import math
import time

from pathlib import Path

import click
import yaml

logger = logging.getLogger(__name__)


def parse_timepoints(spec: str) -> list[int]:
    """Parse '0-9' (inclusive range), '0,3,7' (list), or '5' (single)."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",")]
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


@click.command("tile-stitch", no_args_is_help=True)
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--input", "-i", "input_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--output", "-o", "output_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--timepoint",
    type=int,
    default=None,
    help="Single TP (back-compat). Use --timepoints for multi-TP loop.",
)
@click.option(
    "--timepoints",
    type=str,
    default=None,
    help="Multi-TP spec: '0-9' (range), '0,3,7' (list), '5' (single).",
)
@click.option("--channel", type=str, default=None)
@click.option(
    "--nodes",
    type=str,
    default=None,
    help="Comma-sep worker hostnames for a multi-host HostMesh. Single node if unset.",
)
@click.option("--port", type=int, default=26000, show_default=True)
@click.option(
    "--ready-dir",
    type=str,
    default=None,
    help="Shared dir of <hostname>.ready files; the driver waits for all nodes "
    "before attaching (robust to slow cold-start on non-batch nodes).",
)
@click.option(
    "--shard-by-proc",
    is_flag=True,
    help="TP-parallel sharding: split --timepoints across SLURM tasks "
    "(SLURM_PROCID/SLURM_NTASKS), each task processes its contiguous shard "
    "on its own local GPUs. One shared output zarr; proc 0 creates it "
    "(global T), others wait. The right pattern for many TPs.",
)
def tile_stitch_cli(
    config: Path,
    input_path: Path,
    output_path: Path,
    timepoint: int | None,
    timepoints: str | None,
    channel: str | None,
    nodes: str | None,
    port: int,
    ready_dir: str | None,
    shard_by_proc: bool,
) -> None:
    """Distributed tile-stitch reconstruction over a Monarch actor mesh."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import os

    import numpy as np

    from iohub.ngff import open_ome_zarr
    from waveorder.tile_stitch._engine import build_plan as engine_build_plan

    from biahub.tile_stitch.config import TileStitchRun
    from biahub.tile_stitch.monarch.backend import MonarchBackend
    from biahub.tile_stitch.plan import from_engine_plan, write_plan

    raw = yaml.safe_load(config.read_text())
    run = TileStitchRun.model_validate(raw)
    cfg = run.monarch

    # Resolve the GLOBAL TP list. ``--timepoints`` wins; otherwise fall back to
    # single ``--timepoint`` (default 0) for backward-compatible single-TP.
    if timepoints:
        global_tps = parse_timepoints(timepoints)
    elif timepoint is not None:
        global_tps = [timepoint]
    else:
        global_tps = [0]

    # TP-parallel sharding: each SLURM task takes a contiguous slice of the
    # global TP list and runs independently on its node's local GPUs. The
    # output zarr's T dim spans the GLOBAL range; this task only writes its
    # shard's T slots (stitch writes to plan.timepoint = global TP index).
    procid = int(os.environ.get("SLURM_PROCID", "0")) if shard_by_proc else 0
    nprocs = int(os.environ.get("SLURM_NTASKS", "1")) if shard_by_proc else 1
    if shard_by_proc and nprocs > 1:
        per = math.ceil(len(global_tps) / nprocs)
        tps = global_tps[procid * per : (procid + 1) * per]
        logger.info("shard %d/%d: my TPs=%s (global=%s)", procid, nprocs, tps, global_tps)
        if not tps:
            logger.info("shard %d has no TPs; exiting", procid)
            return
    else:
        tps = global_tps
        logger.info("running TPs: %s", tps)

    run_dir = Path(run.run_dir)
    if str(output_path.parent).startswith("/hpc/projects/waveorder/tile-stitch/runs/"):
        run_dir = output_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build the engine plan once from the first TP — geometry is identical
    # across TPs (same volume shape over time), so all later TPs share it.
    src = open_ome_zarr(input_path, layout="fov", mode="r")
    ch = channel if channel is not None else src.channel_names[0]
    channel_idx = src.channel_names.index(ch)
    czyx = src.to_xarray().isel(t=tps[0]).sel(c=[ch])
    engine_plan = engine_build_plan(czyx, run.tile_stitch, batch_size=None)

    spatial_shape = tuple(engine_plan.full_shape[d] for d in engine_plan.tile_dims)
    tile_spatial = tuple(run.tile_stitch.tile.tile_size[d] for d in engine_plan.tile_dims)
    chunk_shape = (1, 1) + tile_spatial

    # Output zarr T dim spans the GLOBAL TP range so every shard's writes land
    # in the right slot; unwritten slots read back as zero (fill_value).
    t_size = max(global_tps) + 1
    full_shape = (t_size, 1) + spatial_shape

    final_output = (
        output_path if output_path.suffix == ".zarr" else output_path.with_suffix(".zarr")
    )
    if procid == 0:
        # One coordinator pre-creates the shared output mosaic: an OME-NGFF
        # FOV whose T dim spans the GLOBAL range, lazily zero-filled
        # (unwritten T slots read back as 0). T-chunk=1 so shards write
        # disjoint chunks concurrently, no races. ``create_zeros`` writes the
        # metadata + array without materializing the (100s of GB) full volume.
        with open_ome_zarr(
            final_output, layout="fov", mode="w", channel_names=[f"{ch}_recon"]
        ) as out_ds:
            out_ds.create_zeros("0", shape=full_shape, dtype=np.float32, chunks=chunk_shape)
        logger.info(
            "output zarr created: %s | full_shape=%s | chunks=%s",
            final_output,
            full_shape,
            chunk_shape,
        )
    else:
        # Other shards wait for the coordinator's zarr to appear.
        arr0 = Path(final_output) / "0"
        for _ in range(300):
            if arr0.exists():
                break
            time.sleep(1)
        logger.info("shard %d: output zarr ready at %s", procid, final_output)

    # Build (and pickle) one plan per TP. ``plan.timepoint`` selects both the
    # input TP (volume load) and the output T slot (stitch write). The resolved
    # MonarchConfig rides on each plan so the actor reads the same knobs across
    # setup + every swap_to.
    multi_tp = len(global_tps) > 1
    plan_entries: list[tuple[int, str, object]] = []
    for tp in tps:
        run_plan = from_engine_plan(
            engine_plan,
            settings=run.tile_stitch,
            input_path=str(input_path),
            output_path=str(final_output),
            channel=ch,
            channel_idx=channel_idx,
            timepoint=tp,
            monarch=cfg,
        )
        plan_filename = f"plan_t{tp}.pkl" if multi_tp else "plan.pkl"
        plan_path = write_plan(run_plan, run_dir, filename=plan_filename)
        plan_entries.append((tp, plan_path, run_plan))

    # All durable knobs come from the config (single source of truth).
    node_list = [n for n in (nodes or "").split(",") if n]

    # GPUs-per-node is an allocation fact, not a tuning knob: prefer the
    # explicit config value, else SLURM_GPUS_ON_NODE; if neither is set the
    # backend falls back to torch.cuda.device_count() (honors --gres /
    # CUDA_VISIBLE_DEVICES). This is what lets the YAML stay in sync with the
    # sbatch --gres for free.
    gpus_per_node = cfg.gpus_per_node or int(os.environ.get("SLURM_GPUS_ON_NODE") or 0) or None

    per_tp_walls: list[dict] = []
    t_total_start = time.monotonic()
    with MonarchBackend(
        gpus_per_node=gpus_per_node,
        nodes=node_list,
        port=port,
        ready_dir=ready_dir,
        window_per_actor=cfg.window_per_actor,
        device=cfg.device.value,
    ) as backend:
        backend.setup(plan_entries[0][1])

        for i, (tp, plan_path, run_plan) in enumerate(plan_entries):
            if i > 0:
                backend.swap(plan_path)

            t_wall_start = time.monotonic()
            summary = backend.drive_tp(plan_path, run_plan, recon_batch=cfg.recon_batch)
            wall = time.monotonic() - t_wall_start
            t_a = summary["stage_a_s"]
            n_completed = summary["n_outputs"]
            logger.info(
                "TP %d finished: wall=%.1fs (A=%.1fs), tiles=%d/%d",
                tp,
                wall,
                t_a,
                n_completed,
                len(run_plan.output_tiles),
            )

            try:
                for st in backend.collect_recon_stats():
                    logger.info(
                        "  actor %s gpu%d: n=%d io=%.1fs fft=%.1fs d2h=%.1fs "
                        "busy=%.1fs span=%.1fs util=%.2f",
                        st["host"],
                        st["gpu_idx"],
                        st["n_tiles"],
                        st["io_s"],
                        st["fft_s"],
                        st["d2h_s"],
                        st["busy_s"],
                        st["span_s"],
                        st["util"],
                    )
            except Exception as exc:
                logger.warning("recon_stats collection failed: %s", exc)

            per_tp_walls.append(
                {
                    "timepoint": tp,
                    "wall_s": wall,
                    "stage_a_s": t_a,
                    "n_outputs": n_completed,
                }
            )

    total_s = time.monotonic() - t_total_start
    # Per-shard walls file so concurrent shards don't clobber each other.
    walls_name = f"walls_proc{procid}.json" if shard_by_proc else "walls.json"
    walls_path = run_dir / walls_name
    walls_path.write_text(
        json.dumps(
            {
                "procid": procid,
                "nprocs": nprocs,
                "tps": [w["timepoint"] for w in per_tp_walls],
                "total_s": total_s,
                "per_tp": per_tp_walls,
            },
            indent=2,
        )
    )
    logger.info(
        "tile-stitch done: shard %d tps=%s total=%.1fs walls=%s",
        procid,
        [w["timepoint"] for w in per_tp_walls],
        total_s,
        walls_path,
    )
    click.echo(f"tile-stitch complete: {final_output}")
