"""biahub tile-stitch — minimal driver. Build cluster, run pipeline, write zarr."""

import logging

from pathlib import Path

import click
import yaml

logger = logging.getLogger(__name__)


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
@click.option("--gpu", is_flag=True, help="Use gpu_pool (LocalCUDACluster)")
@click.option("--timepoint", type=int, default=0, show_default=True)
@click.option("--channel", type=str, default=None)
def tile_stitch_cli(
    config: Path,
    input_path: Path,
    output_path: Path,
    gpu: bool,
    timepoint: int,
    channel: str | None,
) -> None:
    """Distributed tile-stitch reconstruction over a dask cluster."""
    import time

    import numpy as np

    from distributed import Client
    from iohub.ngff import open_ome_zarr
    from waveorder.tile_stitch._engine import build_plan as engine_build_plan

    from biahub.tile_stitch.config import TileStitchRun
    from biahub.tile_stitch.dispatcher import (
        gpu_worker_setup,
        make_cpu_cluster,
        make_gpu_cluster,
        wait_for_workers,
    )
    from biahub.tile_stitch.pipeline import drive_pipelined_v6, drive_pipelined_v7
    from biahub.tile_stitch.plan import from_engine_plan, write_plan

    raw = yaml.safe_load(config.read_text())
    run = TileStitchRun.model_validate(raw)
    run_dir = Path(run.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if gpu:
        if run.gpu_pool is None:
            raise click.UsageError("--gpu specified but no gpu_pool in config")
        pool = run.gpu_pool
        batch_size = 1  # GPU path uses per-tile dispatch (legacy v7)
    else:
        if run.cpu_pool is None:
            raise click.UsageError("no cpu_pool in config (and --gpu not set)")
        pool = run.cpu_pool
        batch_size = pool.batch_size

    # Read input shape + build engine plan
    src = open_ome_zarr(input_path, layout="fov", mode="r")
    if channel is None:
        channel = src.channel_names[0]
    channel_idx = src.channel_names.index(channel)
    czyx = src.to_xarray().isel(t=timepoint).sel(c=[channel])

    engine_plan = engine_build_plan(
        czyx, run.tile_stitch, batch_size=batch_size if batch_size > 1 else None
    )

    # Pre-create the output zarr (5D: T, C, Z, Y, X). Chunks shadow
    # tile_size — one output tile = one zarr chunk = single-writer-per-chunk.
    final_output = (
        output_path if output_path.suffix == ".zarr" else output_path.with_suffix(".zarr")
    )
    full_shape = (1, 1) + tuple(engine_plan.full_shape[d] for d in engine_plan.tile_dims)
    chunk_shape = (1, 1) + tuple(
        run.tile_stitch.tile.tile_size[d] for d in engine_plan.tile_dims
    )
    out_ds = open_ome_zarr(
        final_output, layout="fov", mode="w", channel_names=[f"{channel}_recon"]
    )
    out_ds.create_image("0", np.zeros(full_shape, dtype=np.float32), chunks=chunk_shape)
    out_ds.close()

    # Compose RunPlan + pickle to disk
    run_plan = from_engine_plan(
        engine_plan,
        settings=run.tile_stitch,
        input_path=str(input_path),
        output_path=str(final_output),
        channel=channel,
        channel_idx=channel_idx,
        timepoint=timepoint,
    )
    plan_path = write_plan(run_plan, run_dir)
    logger.info("plan pickled to %s", plan_path)

    # Build cluster + run
    if gpu:
        cluster = make_gpu_cluster(pool)
    else:
        cluster = make_cpu_cluster(pool, run_dir=str(run_dir))

    t_started = time.time()
    try:
        with Client(cluster) as client:
            if gpu:
                client.run(gpu_worker_setup)
                summary = drive_pipelined_v7(
                    client=client,
                    plan_path=plan_path,
                    plan=run_plan,
                    retries=run.retries,
                )
            else:
                if pool.pool_mode == "adapt":
                    cluster.adapt(minimum=pool.workers_min, maximum=pool.workers_max)
                else:
                    cluster.scale(pool.workers_max)
                target = pool.prewarm_workers or pool.workers_min
                wait_for_workers(client, target, pool.prewarm_timeout_s)
                summary = drive_pipelined_v6(
                    client=client,
                    plan_path=plan_path,
                    plan=run_plan,
                    retries=run.retries,
                )
    finally:
        try:
            cluster.close()
        except Exception:
            pass

    n_failed = len(summary.get("a_failures", [])) + len(summary.get("b_failures", []))
    logger.info(
        "tile-stitch finished: wall=%.1fs, stage_a=%d, stage_b=%d, failed=%d",
        summary.get("wall_s", time.time() - t_started),
        summary.get("stage_a_count", 0),
        summary.get("stage_b_count", 0),
        n_failed,
    )

    if n_failed > 0:
        raise click.ClickException(
            f"tile-stitch finished with {n_failed} failed task(s). See driver log for details."
        )
    click.echo(f"tile-stitch complete: {final_output}")
