"""Shared zarr helpers for the Monarch tile-stitch driver."""

from pathlib import Path

import numpy as np


def create_multi_tp_zarr(
    path: Path,
    full_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    channel_label: str,
) -> None:
    """Pre-create an OME-NGFF FOV with ``T=full_shape[0]`` slots, no host RAM.

    ``Position.create_zeros`` writes the OME-NGFF metadata and a lazily
    zero-filled zarr array (unwritten chunks read back as ``fill_value=0``)
    without materializing the full array on the host — which would be
    hundreds of GB for multi-TP outputs. Shards then write their disjoint
    T slots concurrently.
    """
    from iohub.ngff import open_ome_zarr

    with open_ome_zarr(
        path, layout="fov", mode="w", channel_names=[channel_label]
    ) as out_ds:
        out_ds.create_zeros(
            "0", shape=full_shape, dtype=np.float32, chunks=chunk_shape
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
