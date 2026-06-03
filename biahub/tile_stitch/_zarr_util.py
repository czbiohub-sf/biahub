"""Shared zarr helpers for the Monarch and Dask tile-stitch drivers."""

from pathlib import Path

import numpy as np


def create_multi_tp_zarr(
    path: Path,
    full_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    channel_label: str,
) -> None:
    """Pre-create an OME-NGFF FOV with ``T=full_shape[0]`` slots, no host RAM.

    ``iohub.ngff.Position.create_image(data=np.zeros(full_shape))``
    materializes the full zeros array on the host, which is hundreds of
    GB for multi-TP outputs. We use a tiny placeholder to write the
    OME-NGFF metadata, then drop and recreate the underlying zarr array
    with the correct shape and ``fill_value=0`` so unwritten chunks read
    back as zero.
    """
    import shutil

    import zarr

    from iohub.ngff import open_ome_zarr

    out_ds = open_ome_zarr(
        path, layout="fov", mode="w", channel_names=[channel_label]
    )
    out_ds.create_image("0", np.zeros((1, 1, 1, 1, 1), dtype=np.float32))
    out_ds.close()

    arr_path = Path(path) / "0"
    if arr_path.exists():
        shutil.rmtree(arr_path)
    g = zarr.open_group(str(path), mode="a")
    g.create_dataset(
        "0",
        shape=full_shape,
        chunks=chunk_shape,
        dtype=np.float32,
        fill_value=0.0,
    )


def parse_timepoints(spec: str) -> list[int]:
    """Parse '0-9' (inclusive range), '0,3,7' (list), or '5' (single)."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",")]
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]
