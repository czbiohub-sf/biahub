"""biahub tile-stitch — Monarch-engine tiled reconstruction.

Reconstructs each input position (a fused FOV) into the mirrored output position
over a Monarch actor mesh. Channel + timepoints come from the config; single- vs
multi-node is auto-detected from the SLURM allocation (no topology flags).

NOTE: tile-stitch is not wired into the Nextflow pipeline in this pass.
"""

import json
import logging
import os
import time

from pathlib import Path

import click
import yaml

from biahub.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath

logger = logging.getLogger(__name__)


def _resolve_time_indices(spec: int | list[int] | str, n_t: int) -> list[int]:
    """Config ``time_indices`` (``'all'`` | int | list) → an explicit list."""
    if spec == "all":
        return list(range(n_t))
    if isinstance(spec, int):
        return [spec]
    return list(spec)


@click.command("tile-stitch")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def tile_stitch_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
) -> None:
    """Distributed tiled reconstruction over a Monarch actor mesh.

    Each input position (a fused FOV) is tiled, reconstructed, and stitched into
    the mirrored output position. Channel + timepoints are set in the config;
    single- vs multi-node is auto-detected from the SLURM allocation.

    Not wired into the Nextflow pipeline this pass.

    >>> biahub tile-stitch -i ./fused.zarr/*/*/* -c ./phase.yml -o ./recon.zarr
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    import numpy as np

    from iohub.ngff import Plate, open_ome_zarr
    from iohub.ngff.models import TransformationMeta
    from waveorder.tile_stitch._engine import build_plan as engine_build_plan

    from biahub.settings import TileStitchReconSettings
    from biahub.tile_stitch.monarch.backend import MonarchBackend
    from biahub.tile_stitch.plan import from_engine_plan, write_plan

    positions = list(input_position_dirpaths)
    run = TileStitchReconSettings.model_validate(yaml.safe_load(config_filepath.read_text()))
    cfg = run.monarch
    ch = run.tile_stitch.recon.input_channel_names[0]  # channel lives in the config

    # Run artifacts (plan pickles, walls) sit next to the output store.
    final_output = (
        output_dirpath
        if output_dirpath.suffix == ".zarr"
        else output_dirpath.with_suffix(".zarr")
    )
    run_dir = final_output.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # Validate all positions share TCZYX shape (geometry + TF are reused across
    # them), and pull channel index + voxel scale from the first.
    with open_ome_zarr(str(positions[0]), mode="r") as ds0:
        shape0 = ds0.data.shape
        channel_idx = ds0.channel_names.index(ch)
        input_scale = list(ds0.scale)
    for p in positions[1:]:
        with open_ome_zarr(str(p), mode="r") as ds:
            if ds.data.shape != shape0:
                raise click.UsageError(
                    f"all positions must share TCZYX shape; {p} is {ds.data.shape} "
                    f"!= {shape0} ({positions[0]})"
                )
    tps = _resolve_time_indices(run.time_indices, shape0[0])
    logger.info("positions=%d channel=%r timepoints=%s", len(positions), ch, tps)

    # Output layout mirrors the input: HCS plate positions -> a plate; a single
    # standalone FOV -> a FOV. (No mixing.)
    def _is_plate_position(p: Path) -> bool:
        try:
            with open_ome_zarr(str(Path(p).parents[2]), mode="r") as anc:
                return isinstance(anc, Plate)
        except Exception:
            return False

    plate_mode = all(_is_plate_position(p) for p in positions)
    if not plate_mode and len(positions) > 1:
        raise click.UsageError(
            "multiple standalone FOVs are not supported; pass HCS plate positions "
            "(e.g. plate.zarr/*/*/*) or a single FOV"
        )

    # Geometry is identical across positions+tps, so build the engine plan once.
    with open_ome_zarr(str(positions[0]), mode="r") as ds0:
        czyx = ds0.to_xarray().isel(t=tps[0]).sel(c=[ch])
    engine_plan = engine_build_plan(czyx, run.tile_stitch, batch_size=None)

    spatial_shape = tuple(engine_plan.full_shape[d] for d in engine_plan.tile_dims)
    tile_spatial = tuple(run.tile_stitch.tile.tile_size[d] for d in engine_plan.tile_dims)
    # Output chunk = one tile, but cap the depth so a single blosc compress stays
    # under its hard 2 GiB limit (full-Z tiles with large YX overflow it).
    CHUNK_BYTES_CAP = 1_000_000_000
    lead, *rest = tile_spatial
    rest_bytes = int(np.prod(rest)) * 4  # float32
    cap = max(1, CHUNK_BYTES_CAP // max(1, rest_bytes))
    chunk_shape = (1, 1, min(lead, cap)) + tuple(rest)

    # One output channel per run; T spans the requested timepoints (unwritten
    # slots read back as fill_value). T/C-chunk=1 so tps write disjoint chunks.
    full_shape = (max(tps) + 1, 1) + spatial_shape
    out_recon_name = f"{ch}_recon"
    scale_5 = tuple(input_scale) if len(input_scale) == 5 else (1.0, 1.0, *input_scale[-3:])

    # Create the output up-front (single driver — no creator race), one output
    # position per input position.
    if plate_mode:
        from iohub.ngff.utils import create_empty_plate

        from biahub.cli.utils import get_output_paths

        out_targets = get_output_paths(positions, final_output)
        create_empty_plate(
            store_path=final_output,
            position_keys=[tuple(Path(p).parts[-3:]) for p in positions],
            channel_names=[out_recon_name],
            shape=full_shape,
            chunks=chunk_shape,
            scale=scale_5,
            dtype=np.float32,
        )
    else:
        with open_ome_zarr(
            final_output, layout="fov", mode="w", channel_names=[out_recon_name]
        ) as out_ds:
            out_ds.create_zeros(
                "0",
                shape=full_shape,
                dtype=np.float32,
                chunks=chunk_shape,
                transform=[TransformationMeta(type="scale", scale=list(scale_5))],
            )
        out_targets = [final_output]
    logger.info(
        "output created: %s | positions=%d | full_shape=%s | chunks=%s | channel=%s",
        final_output,
        len(positions),
        full_shape,
        chunk_shape,
        out_recon_name,
    )

    # One plan per (position, timepoint): same geometry, per-position input/output
    # path + timepoint. The resolved MonarchConfig rides on each plan so actors
    # read the same knobs across setup + every swap_to.
    plan_entries: list[tuple[Path, int, str, object]] = []
    for pi, (pos, out_target) in enumerate(zip(positions, out_targets, strict=True)):
        for tp in tps:
            run_plan = from_engine_plan(
                engine_plan,
                settings=run.tile_stitch,
                input_path=str(pos),
                output_path=str(out_target),
                channel=ch,
                channel_idx=channel_idx,
                timepoint=tp,
                output_channel_index=0,
                monarch=cfg,
            )
            plan_path = write_plan(run_plan, run_dir, filename=f"plan_p{pi}_t{tp}.pkl")
            plan_entries.append((pos, tp, plan_path, run_plan))

    # GPUs-per-node: config value, else SLURM_GPUS_ON_NODE, else backend auto
    # (torch.cuda.device_count). Topology (single vs multi node) is detected by
    # the backend from the SLURM allocation.
    gpus_per_node = cfg.gpus_per_node or int(os.environ.get("SLURM_GPUS_ON_NODE") or 0) or None

    per_unit_walls: list[dict] = []
    t_total_start = time.monotonic()
    with MonarchBackend(
        gpus_per_node=gpus_per_node,
        window_per_actor=cfg.window_per_actor,
        device=cfg.device.value,
    ) as backend:
        backend.setup(plan_entries[0][2])
        for i, (pos, tp, plan_path, run_plan) in enumerate(plan_entries):
            if i > 0:
                backend.swap(plan_path)
            t_wall_start = time.monotonic()
            summary = backend.drive_tp(plan_path, run_plan, recon_batch=cfg.recon_batch)
            wall = time.monotonic() - t_wall_start
            logger.info(
                "%s tp%d: wall=%.1fs (A=%.1fs), tiles=%d/%d",
                Path(pos).name,
                tp,
                wall,
                summary["stage_a_s"],
                summary["n_outputs"],
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
            per_unit_walls.append(
                {
                    "position": Path(pos).name,
                    "timepoint": tp,
                    "wall_s": wall,
                    "stage_a_s": summary["stage_a_s"],
                    "n_outputs": summary["n_outputs"],
                }
            )

    total_s = time.monotonic() - t_total_start
    (run_dir / "walls.json").write_text(
        json.dumps({"total_s": total_s, "units": per_unit_walls}, indent=2)
    )
    logger.info("tile-stitch done: %d units, total=%.1fs", len(plan_entries), total_s)
    click.echo(f"tile-stitch complete: {final_output}")
