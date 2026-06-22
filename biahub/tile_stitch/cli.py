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
    help="Single timepoint; use --timepoints for a range/list.",
)
@click.option(
    "--timepoints",
    type=str,
    default=None,
    help="Multi-TP spec: '0-9' (range), '0,3,7' (list), '5' (single).",
)
@click.option("--channel", type=str, default=None)
@click.option(
    "--output-channels",
    type=str,
    default=None,
    help="Comma-separated ordered list of input channels that share ONE output "
    "zarr (each reconstructed by a separate invocation with the same --output). "
    "This run writes the C slot matching its --channel; the first channel "
    "(index 0) creates the C=N store with all '<ch>_recon' names, the rest "
    "append. Omit for a single-channel output.",
)
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
    output_channels: str | None,
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
    # Reconstruction is in-place on the input voxel grid (no resampling), so the
    # output must carry the input's voxel scale. Without this the output zarr
    # defaults to all-1s and viewers render it with the wrong (isotropic) aspect
    # — anisotropic data (e.g. coarse Y) then looks squashed/"transposed".
    input_scale = list(src.scale)

    # Multi-channel shared output: all listed channels write into ONE C=N zarr,
    # each invocation filling its own C slot. The first channel (index 0) creates
    # the store with every '<ch>_recon' name; the rest append. Omitting
    # --output-channels keeps the single-channel (C=1) behavior.
    out_channels = (
        [c.strip() for c in output_channels.split(",") if c.strip()]
        if output_channels
        else [ch]
    )
    if ch not in out_channels:
        raise click.ClickException(
            f"--channel {ch!r} is not in --output-channels {out_channels}"
        )
    n_out_channels = len(out_channels)
    out_channel_idx = out_channels.index(ch)
    out_recon_names = [f"{c}_recon" for c in out_channels]
    czyx = src.to_xarray().isel(t=tps[0]).sel(c=[ch])
    engine_plan = engine_build_plan(czyx, run.tile_stitch, batch_size=None)

    spatial_shape = tuple(engine_plan.full_shape[d] for d in engine_plan.tile_dims)
    tile_spatial = tuple(run.tile_stitch.tile.tile_size[d] for d in engine_plan.tile_dims)
    # Output chunk = one tile, BUT capped so a single blosc compress stays under
    # its hard 2 GiB limit. Full-Z tiles with large YX overflow it (e.g.
    # 2368x512x512x4 = 2.48 GB > 2 GiB -> blosc_compress_ctx fails). Cap the
    # leading (depth) dim to ~1 GiB worth; each output tile owns its full depth
    # column, so a smaller depth chunk doesn't create cross-tile write races.
    CHUNK_BYTES_CAP = 1_000_000_000
    lead, *rest = tile_spatial  # (depth, ...lateral)
    rest_bytes = int(np.prod(rest)) * 4  # float32
    cap = max(1, CHUNK_BYTES_CAP // max(1, rest_bytes))
    chunk_shape = (1, 1, min(lead, cap)) + tuple(rest)

    # Output zarr T dim spans the GLOBAL TP range so every shard's writes land
    # in the right slot; unwritten slots read back as zero (fill_value). C spans
    # all shared output channels; this run fills its own C slot.
    t_size = max(global_tps) + 1
    full_shape = (t_size, n_out_channels) + spatial_shape

    final_output = (
        output_path if output_path.suffix == ".zarr" else output_path.with_suffix(".zarr")
    )
    # The first output channel (index 0) on the coordinator shard pre-creates the
    # shared output: an OME-NGFF FOV whose T spans the GLOBAL range and C spans
    # all channels, lazily zero-filled. T- and C-chunk=1 so shards/channels write
    # disjoint chunks concurrently, no races. Every other channel and shard waits
    # for it, then writes its own [t, c_idx] slot (the stitch opens mode="a").
    is_creator = procid == 0 and out_channel_idx == 0
    if is_creator:
        from iohub.ngff.models import TransformationMeta

        with open_ome_zarr(
            final_output, layout="fov", mode="w", channel_names=out_recon_names
        ) as out_ds:
            out_ds.create_zeros(
                "0",
                shape=full_shape,
                dtype=np.float32,
                chunks=chunk_shape,
                transform=[TransformationMeta(type="scale", scale=input_scale)],
            )
        logger.info(
            "output zarr created: %s | full_shape=%s | chunks=%s | channels=%s",
            final_output,
            full_shape,
            chunk_shape,
            out_recon_names,
        )
    else:
        # Other shards / channels wait for the coordinator's zarr to appear.
        arr0 = Path(final_output) / "0"
        for _ in range(600):
            if arr0.exists():
                break
            time.sleep(1)
        logger.info(
            "shard %d (channel %s -> C=%d): output zarr ready at %s",
            procid,
            ch,
            out_channel_idx,
            final_output,
        )

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
            output_channel_index=out_channel_idx,
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
                        "  actor %s gpu%d: n=%d io=%.1fs fft=%.1fs "
                        "d2h=%.1fs (copy=%.1fs rdma=%.1fs) busy=%.1fs span=%.1fs util=%.2f",
                        st["host"],
                        st["gpu_idx"],
                        st["n_tiles"],
                        st["io_s"],
                        st["fft_s"],
                        st["d2h_s"],
                        st.get("copy_s", 0.0),
                        st.get("rdma_s", 0.0),
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
