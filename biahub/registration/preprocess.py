"""Preprocessing blocks: what a config's `preprocess:` list does to the volumes.

Image blocks change a volume's pixels, in order; mask blocks say where to compare. Each
block applies to the volumes in its `target`, in that volume's own grid (see the block
models in `biahub.settings`).
"""

from __future__ import annotations

import numpy as np

from skimage import filters

from biahub.settings import (
    MASK_BLOCKS,
    ClipPercentiles,
    MaskBox,
    MaskCircle,
    PreprocessBlocks,
    Sobel,
)


def apply_image_blocks(volume: np.ndarray, blocks: PreprocessBlocks, which: str) -> np.ndarray:
    """Return `volume` (the `which` one: "moving" or "reference") after its image blocks."""
    out = np.asarray(volume, dtype=np.float32)
    for block in blocks:
        if which not in block.target or isinstance(block, MASK_BLOCKS):
            continue
        if isinstance(block, ClipPercentiles):
            low, high = np.nanpercentile(out, [block.low, block.high])
            out = np.clip(out, low, high)
        elif isinstance(block, Sobel):
            out = filters.sobel(out).astype(np.float32)
        else:
            raise TypeError(f"no image operation for {type(block).__name__}")
    return out


def build_mask(shape_zyx: tuple, blocks: PreprocessBlocks, which: str) -> np.ndarray | None:
    """Return where to compare in the `which` volume (its mask blocks combined), or None."""
    masks = [
        _mask(block, tuple(shape_zyx))
        for block in blocks
        if isinstance(block, MASK_BLOCKS) and which in block.target
    ]
    if not masks:
        return None
    return np.logical_and.reduce(masks)


def _mask(block, shape_zyx: tuple) -> np.ndarray:
    Z, Y, X = shape_zyx
    if isinstance(block, MaskCircle):
        center_y, center_x = Y // 2, X // 2
        radius = int(block.radius * min(center_y, center_x))
        y, x = np.ogrid[:Y, :X]
        disk = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
        return np.broadcast_to(disk, shape_zyx).copy()
    if isinstance(block, MaskBox):
        slices = []
        for axis, size, r in zip("zyx", shape_zyx, block.box, strict=True):
            if r == "all":
                slices.append(slice(0, size))
                continue
            if r[1] > size:
                raise ValueError(
                    f"mask_box {axis} range {r} is outside the {block.target[0]} volume "
                    f"({axis} size {size})"
                )
            slices.append(slice(r[0], r[1]))
        mask = np.zeros(shape_zyx, dtype=bool)
        mask[tuple(slices)] = True
        return mask
    raise TypeError(f"no mask for {type(block).__name__}")
