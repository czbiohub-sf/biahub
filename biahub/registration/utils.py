"""Registration helpers that fit no module of their own.

- Shape matching: `match_shape` pads (reflect) or centre-crops a volume to a shape, and
  `match_shape_origin` says where its origin lands (phase correlation resizes both
  volumes to one FFT shape); `pad_to_shape` and `center_crop` are its two halves.
  `center_window_yx`: the centred window a `center_crop_xy` setting asks for.
- `resolve_time_indices`: the timepoints a `time_indices` setting names, checked
  against the store.
- `RunError`: a run refused or unable to go on (settings that do not fit the data, a
  missing or mismatched run folder). The library raises it; each command shows it as a
  usage error (`biahub.cli.parsing.refusals_as_usage_errors`).
- `Remembered`: the last few results computed on given arrays (bead peaks, scores), so
  a job does not compute one twice; `transform_key` names a transform in its key.

Transforms live in `biahub.core.transform`, cropping in `biahub.estimate_crop`, the
initial transform from the two grids in the shapes method (`methods/manual/shapes.py`).
Shapes are ZYX.
"""

import weakref

from collections import OrderedDict

import click
import numpy as np

from numpy.typing import ArrayLike


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


def center_window_yx(yx_shape: tuple[int, int], crop_xy) -> tuple[slice, slice]:
    """Return the centred window of at most `crop_xy` (X, Y) pixels in a (y, x) plane.

    As (y, x) slices. A crop larger than an axis takes the whole axis; the stop is the
    start plus the size, so an odd size keeps its last pixel.
    """
    y, x = yx_shape
    cx, cy = min(crop_xy[0], x), min(crop_xy[1], y)
    y0, x0 = y // 2 - cy // 2, x // 2 - cx // 2
    return slice(y0, y0 + cy), slice(x0, x0 + cx)


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
    # each axis on its own: pad up to the larger size, then centre-crop down to `shape`
    shape = tuple(int(n) for n in shape)
    padded = tuple(max(n, m) for n, m in zip(img.shape, shape, strict=True))
    if padded != img.shape:
        img = pad_to_shape(img, padded, mode="reflect")
    if shape != img.shape:
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


class RunError(ValueError):
    """A run refused or unable to go on.

    Settings that do not fit the data, a missing or mismatched run folder. Not a
    command-line error: each command converts it to one.
    """


def resolve_time_indices(time_indices, n_t: int) -> list[int]:
    """Return the timepoints a `time_indices` setting names: 'all', one index, or a list.

    Refuses a selection with no timepoints or one outside the store's `n_t`, so it fails
    here rather than as an index error inside a job.
    """
    if time_indices == "all":
        return list(range(n_t))
    indices = [time_indices] if isinstance(time_indices, int) else list(time_indices)
    if not indices:
        raise RunError("time_indices selects no timepoints")
    outside = sorted({t for t in indices if not 0 <= t < n_t})
    if outside:
        raise RunError(
            f"time_indices {outside} out of range: the store has {n_t} timepoints "
            f"(0-{n_t - 1})"
        )
    return indices


class Remembered:
    """The last `kept` results computed on given arrays, found again by the arrays' identity.

    A result is reused only while the very arrays it was computed on are alive (they are
    held by weak reference), so a new array at a freed one's address is not mistaken for
    it. Arrays changed in place are not noticed: keep a result only for arrays a job
    reads, never writes.
    """

    def __init__(self, kept: int):
        self.kept = kept
        self._results: OrderedDict = OrderedDict()

    def __call__(self, arrays: tuple, extra: tuple, compute):
        """Return `compute()` for `arrays` and `extra` (hashable), computed once."""
        try:
            refs = tuple(weakref.ref(a) for a in arrays)
        except TypeError:  # an array without weak references (e.g. a list): not remembered
            return compute()
        key = (tuple(id(a) for a in arrays), extra)
        hit = self._results.get(key)
        if hit is not None and all(r() is a for r, a in zip(hit[0], arrays, strict=True)):
            self._results.move_to_end(key)
            return hit[1]
        result = compute()
        self._results[key] = (refs, result)
        while len(self._results) > self.kept:
            self._results.popitem(last=False)
        return result


def transform_key(transform) -> bytes | str:
    """Name `transform` in a `Remembered` key: its matrix (PREWARPED: the volume as it is)."""
    matrix = getattr(transform, "matrix", None)
    return "as is" if matrix is None else np.asarray(matrix).tobytes()
