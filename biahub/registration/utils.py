"""
Registration utility functions.

Helper functions shared across registration methods (beads, ANTs, phase correlation).
Includes:
- Transform validation and interpolation across timepoints.
- Transform I/O (save/load from disk, convert between numpy and ANTs formats).
- Volume utilities: LIR cropping, padding, and shape matching.

Key conventions
---------------
- Coordinates and shapes are in ZYX order for 3D data.
- Transforms are 4x4 homogeneous matrices.
"""

import click
import largestinteriorrectangle as lir
import numpy as np

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

# TODO: see if at some point these globals should be hidden or exposed.
NA_DETECTION_SOURCE = 1.35
NA_DETECTION_TARGET = 1.35
WAVELENGTH_EMISSION_SOURCE_CHANNEL = 0.45  # in um
WAVELENGTH_EMISSION_TARGET_CHANNEL = 0.6  # in um
FOCUS_SLICE_ROI_WIDTH = 150  # size of central ROI used to find focal slice


def find_lir(registered_zyx: np.ndarray, plot: bool = False) -> tuple:
    registered_zyx = np.asarray(registered_zyx, dtype=bool)

    # Find the lir in YX at Z//2
    registered_yx = registered_zyx[registered_zyx.shape[0] // 2].copy()
    coords_yx = lir.lir(registered_yx)
    coords_yx = list(map(int, coords_yx))

    x, y, width, height = coords_yx
    x_start, x_stop = x, x + width
    y_start, y_stop = y, y + height
    x_slice = slice(x_start, x_stop)
    y_slice = slice(y_start, y_stop)

    # Iterate over ZX and ZY slices to find optimal Z cropping params
    _coords = []
    for _x in (x_start, x_start + (x_stop - x_start) // 2, x_stop - 1):
        registered_zy = registered_zyx[:, y_slice, _x].copy()
        coords_zy = lir.lir(registered_zy)
        _, z, _, depth = coords_zy
        z_start, z_stop = z, z + depth
        _coords.append((z_start, z_stop))
    for _y in (y_start, y_start + (y_stop - y_start) // 2, y_stop - 1):
        registered_zx = registered_zyx[:, _y, x_slice].copy()
        coords_zx = lir.lir(registered_zx)
        _, z, _, depth = coords_zx
        z_start, z_stop = z, z + depth
        _coords.append((z_start, z_stop))

    _coords = np.asarray(_coords)
    z_start = int(_coords.max(axis=0)[0])
    z_stop = int(_coords.min(axis=0)[1])
    z_slice = slice(z_start, z_stop)

    if plot:
        xy_corners = ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
        rectangle_yx = plt.Polygon(
            xy_corners,
            closed=True,
            fill=None,
            edgecolor="r",
        )
        # Add the rectangle to the plot
        _, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(registered_yx)
        ax[0].add_patch(rectangle_yx)

        zx_corners = ((x, z), (x + width, z), (x + width, z + depth), (x, z + depth))
        rectangle_zx = plt.Polygon(
            zx_corners,
            closed=True,
            fill=None,
            edgecolor="r",
        )
        ax[1].imshow(registered_zx)
        ax[1].add_patch(rectangle_zx)
        plt.savefig("./lir.png")

    return (z_slice, y_slice, x_slice)


def pad_to_shape(
    arr: ArrayLike,
    shape: tuple[int, ...],
    mode: str,
    verbose: bool = False,
    **kwargs,
) -> ArrayLike:
    """
    Pad or crop array to match provided shape.

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    shape : Tuple[int]
        Output shape.
    mode : str
        Padding mode (see np.pad).
    verbose : bool
        If True, print verbose output.
    kwargs : dict
        Additional keyword arguments for np.pad.

    Returns
    -------
    ArrayLike
        Padded array.
    """
    assert arr.ndim == len(shape)

    dif = tuple(s - a for s, a in zip(shape, arr.shape, strict=True))
    assert all(d >= 0 for d in dif)

    pad_width = [[s // 2, s - s // 2] for s in dif]

    if verbose:
        click.echo(
            f"padding: input shape {arr.shape}, output shape {shape}, padding {pad_width}"
        )

    return np.pad(arr, pad_width=pad_width, mode=mode, **kwargs)


def center_crop(
    arr: ArrayLike,
    shape: tuple[int, ...],
    verbose: bool = False,
) -> ArrayLike:
    """
    Crop the center of `arr` to match provided shape.

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    shape : Tuple[int, ...]
    """
    assert arr.ndim == len(shape)

    starts = tuple((cur_s - s) // 2 for cur_s, s in zip(arr.shape, shape, strict=True))

    assert all(s >= 0 for s in starts)

    slicing = tuple(slice(s, s + d) for s, d in zip(starts, shape, strict=True))
    if verbose:
        click.echo(
            f"center crop: input shape {arr.shape}, output shape {shape}, slicing {slicing}"
        )
    return arr[slicing]


def match_shape(
    img: ArrayLike,
    shape: tuple[int, ...],
    verbose: bool = False,
) -> ArrayLike:
    """
    Pad or crop array to match provided shape.

    Parameters
    ----------
    img : ArrayLike
        Input array.
    shape : Tuple[int, ...]
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    ArrayLike
        Padded or cropped array.
    """
    if np.any(shape > img.shape):
        padded_shape = np.maximum(img.shape, shape)
        img = pad_to_shape(img, padded_shape, mode="reflect")

    if np.any(shape < img.shape):
        img = center_crop(img, shape)

    if verbose:
        click.echo(f"matched shape: input shape {img.shape}, output shape {shape}")

    return img


def match_shape_origin(in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> np.ndarray:
    """Where `match_shape` puts the input's origin in its output: the leading pad minus the crop.

    Mirrors `match_shape` (pad to the larger shape, half the difference before; then centre
    crop), so two volumes resized to one shape can be compared in their own coordinates.
    """
    in_shape, out_shape = np.asarray(in_shape), np.asarray(out_shape)
    padded = np.maximum(in_shape, out_shape)
    leading_pad = (padded - in_shape) // 2
    crop_start = (padded - out_shape) // 2
    return leading_pad - crop_start


def resolve_time_indices(time_indices, n_t: int) -> list[int]:
    """Return the timepoints a `time_indices` setting names: 'all', one index, or a list.

    Refuses a selection with no timepoints or one outside the store's `n_t`, so it fails
    here rather than as an index error inside a job.
    """
    if time_indices == "all":
        return list(range(n_t))
    indices = [time_indices] if isinstance(time_indices, int) else list(time_indices)
    if not indices:
        raise click.UsageError("time_indices selects no timepoints")
    outside = sorted({t for t in indices if not 0 <= t < n_t})
    if outside:
        raise click.UsageError(
            f"time_indices {outside} out of range: the store has {n_t} timepoints "
            f"(0-{n_t - 1})"
        )
    return indices
