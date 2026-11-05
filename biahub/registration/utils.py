"""Registration helpers that fit no module of their own.

- Shape matching: `match_shape` pads (reflect) or centre-crops a volume to a shape, and
  `match_shape_origin` says where its origin lands (phase correlation resizes both
  volumes to one FFT shape); `pad_to_shape` and `center_crop` are its two halves.
- `resolve_time_indices`: the timepoints a `time_indices` setting names, checked
  against the store.

Transforms live in `biahub.core.transform`, cropping in `biahub.estimate_crop`, the
approximate initial transform in `ShapesInitialTransform` (`engine.policies`). Shapes are
ZYX.
"""

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
