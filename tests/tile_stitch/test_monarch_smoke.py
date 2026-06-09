"""GPU + Monarch end-to-end smoke — single host, 1 GPU, 1 TP, tiny phantom.

Opt-in only: set ``TILE_STITCH_GPU_SMOKE=1`` to run. Skips in CI and on dev
nodes — Monarch's RDMABuffer needs the real IB/IPC fabric, which only the SLURM
GPU allocation provides (a login/dev box with a GPU still can't open the
transport). The lead runs this as part of the GPU integration step.

Replaces the deleted ``m1_smoke`` script: it exercises the real
``MonarchBackend`` lifecycle (setup → drive_tp → teardown) over a synthetic
phantom and asserts the output zarr is finite and shaped right. The full
c0032/c0041 parity reproduction stays in the SLURM submission scripts.
"""

from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import pytest

if os.environ.get("TILE_STITCH_GPU_SMOKE") != "1":
    pytest.skip(
        "GPU smoke is opt-in (set TILE_STITCH_GPU_SMOKE=1 on a SLURM GPU node)",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("monarch.actor")

if not torch.cuda.is_available():
    pytest.skip("no CUDA device available", allow_module_level=True)


def _phantom_settings(channel: str):
    from waveorder.api.tile_stitch import TileStitchSettings

    # Tiny tiles with a small overlap so the grid has interior + edge output
    # tiles (the blend path needs at least one multi-contributor output).
    return TileStitchSettings.model_validate(
        {
            "tile": {
                "tile_size": {"z": 8, "y": 32, "x": 32},
                "overlap": {"z": 0, "y": 8, "x": 8},
            },
            "blend": {"kind": "gaussian_mean", "sigma_fraction": 0.125},
            "recon": {
                "input_channel_names": [channel],
                "reconstruction_dimension": 3,
                "phase": {
                    "transfer_function": {
                        "yx_pixel_size": 0.25,
                        "z_pixel_size": 0.25,
                        "z_padding": 2,
                        "index_of_refraction_media": 1.3,
                        "numerical_aperture_detection": 1.0,
                        "numerical_aperture_illumination": 0.5,
                        "wavelength_illumination": 0.5,
                    },
                    "apply_inverse": {
                        "reconstruction_algorithm": "Tikhonov",
                        "regularization_strength": 0.01,
                    },
                },
            },
        }
    )


def test_monarch_single_gpu_one_tp(tmp_path: Path):
    from iohub.ngff import open_ome_zarr
    from waveorder.tile_stitch._engine import build_plan as engine_build_plan

    from biahub.tile_stitch.config import MonarchConfig
    from biahub.tile_stitch.monarch.backend import MonarchBackend
    from biahub.tile_stitch.plan import from_engine_plan, write_plan

    channel = "phantom"
    z, y, x = 8, 64, 64

    # --- synthetic input FOV (1, 1, Z, Y, X) ---
    in_path = tmp_path / "input.zarr"
    rng = np.random.default_rng(0)
    vol = rng.random((1, 1, z, y, x), dtype=np.float32).astype(np.float32)
    src = open_ome_zarr(in_path, layout="fov", mode="w", channel_names=[channel])
    src.create_image("0", vol)
    src.close()

    cfg = MonarchConfig(gpus_per_node=1, recon_batch=1, prefetch_depth=0)
    settings = _phantom_settings(channel)

    src_r = open_ome_zarr(in_path, layout="fov", mode="r")
    czyx = src_r.to_xarray().isel(t=0).sel(c=[channel])
    engine_plan = engine_build_plan(czyx, settings, batch_size=None)

    spatial = tuple(engine_plan.full_shape[d] for d in engine_plan.tile_dims)
    tile_spatial = tuple(settings.tile.tile_size[d] for d in engine_plan.tile_dims)
    out_path = tmp_path / "output.zarr"
    with open_ome_zarr(
        out_path, layout="fov", mode="w", channel_names=[f"{channel}_recon"]
    ) as out_ds:
        out_ds.create_zeros(
            "0", shape=(1, 1) + spatial, dtype=np.float32, chunks=(1, 1) + tile_spatial
        )

    run_plan = from_engine_plan(
        engine_plan,
        settings=settings,
        input_path=str(in_path),
        output_path=str(out_path),
        channel=channel,
        channel_idx=0,
        timepoint=0,
        monarch=cfg,
    )
    plan_path = write_plan(run_plan, tmp_path, filename="plan.pkl")

    with MonarchBackend(gpus_per_node=1, device="cuda") as backend:
        backend.setup(plan_path)
        summary = backend.drive_tp(plan_path, run_plan, recon_batch=1)

    assert summary["n_outputs"] == len(run_plan.output_tiles)

    out = open_ome_zarr(out_path, layout="fov", mode="r")
    arr = np.asarray(out["0"])
    out.close()
    assert arr.shape == (1, 1) + spatial
    assert np.isfinite(arr).all()
    # At least some output is non-zero (real recon ran, not all fill_value).
    assert np.abs(arr).max() > 0
