"""Ultrack: link a foreground/contour pair or fixed labels."""

import json
import shutil
import tempfile

from contextlib import contextmanager
from pathlib import Path

import click
import numpy as np

from numpy.typing import ArrayLike

from biahub.tracking.postprocess import filter_short_siblings

_ultrack_patched = False


def _patch_ultrack_readonly_buffer():
    """Workaround for scikit-image #6378: read-only buffer in _map_array.

    scikit-image's Cython ``_map_array`` rejects read-only memoryviews.
    Under numpy 2.x, pandas DataFrame columns and ``np.asarray`` results are
    often read-only, so every call path through ``map_array`` can trigger this.

    Patch ``skimage.util.map_array`` directly to force writable copies of
    all arrays before passing them to Cython.

    Remove when scikit-image >= 0.27 is available.
    """
    global _ultrack_patched
    if _ultrack_patched:
        return
    _ultrack_patched = True

    import skimage.util._map_array as _ma

    _original_map_array = _ma.map_array

    def _map_array_writable(input_arr, input_vals, output_vals, out=None):
        input_arr = np.array(input_arr, copy=False)
        if not input_arr.flags.writeable:
            input_arr = input_arr.copy()
        input_vals = np.array(input_vals, copy=False)
        if not input_vals.flags.writeable:
            input_vals = input_vals.copy()
        output_vals = np.array(output_vals, copy=False)
        if not output_vals.flags.writeable:
            output_vals = output_vals.copy()
        return _original_map_array(input_arr, input_vals, output_vals, out=out)

    _ma.map_array = _map_array_writable


def run_ultrack(
    tracking_config,
    database_path,
    geff_path: Path | None = None,
    geff_kwargs: dict | None = None,
    min_sibling_length: int | None = None,
    **track_kwargs,
):
    """
    Run object tracking using the Ultrack library.

    Note: ultrack is imported lazily within this function.

    This function performs object tracking on time-series image data and outputs labeled
    segmentation results, a track DataFrame, and a graph of object trajectories over time.
    The detection inputs are forwarded to ``Tracker.track()`` via ``**track_kwargs``, so it
    accepts either the foreground+contour pair (``detection``, ``edges``) or precomputed
    instance ``labels`` (with ``sigma``), plus ``scale`` and ``overwrite``.

    Parameters
    ----------
    tracking_config : MainConfig
        Ultrack configuration object defining segmentation, linking, and optimization parameters.
    database_path : Path
        Directory where tracking results, configuration files, and output data will be saved.
    geff_path : Path, optional
        If set, the tracks are also exported there as GEFF via ``Tracker.to_geff``
        (overwriting any previous export), from the same tracks table the labels are
        painted from, so the three outputs agree after the sibling filter.
    geff_kwargs : dict, optional
        Extra keyword arguments for ``Tracker.to_geff``.
    min_sibling_length : int, optional
        If set, apply :func:`filter_short_siblings` to the tracks and labels before the
        GEFF export.
    **track_kwargs
        Keyword arguments forwarded to ``Tracker.track()``. E.g. ``detection``, ``edges``
        (foreground+contour mode), or ``labels``, ``sigma`` (label-image mode), plus ``scale``
        and ``overwrite``.

    Returns
    -------
    labels : np.ndarray
        Labeled segmentation array of shape (T, Z, Y, X) or (T, Y, X), where each object instance
        is assigned a unique integer label across time.
    tracks_df : pandas.DataFrame
        DataFrame with tracking metadata including object ID, frame index, spatial coordinates,
        and parent-child relationships.
    graph : networkx.Graph
        Directed graph representing tracked object lineages. Nodes correspond to objects and
        edges represent links between objects across frames.

    Notes
    -----
    - This function modifies `tracking_config` to set the working directory (`working_dir`).
    - The configuration used is saved to `config.toml` under the `database_path`.
    """
    import toml

    from ultrack import Tracker

    _patch_ultrack_readonly_buffer()

    cfg = tracking_config

    cfg.data_config.working_dir = database_path

    click.echo(str(cfg))

    tracker = Tracker(cfg)
    tracker.track(**track_kwargs)

    tracks_df, graph = tracker.to_tracks_layer()
    labels = tracker.to_zarr(
        tracks_df=tracks_df,
    )
    if min_sibling_length is not None:
        tracks_df, labels = filter_short_siblings(tracks_df, labels, min_sibling_length)
    if geff_path is not None:
        # `tracks_df` restricts the export to these (possibly filtered) tracks and their
        # ids, which is what the labels were painted with (royerlab/ultrack: to_geff
        # from a tracks DataFrame).
        tracker.to_geff(
            str(geff_path), overwrite=True, tracks_df=tracks_df, **(geff_kwargs or {})
        )

    with open(database_path / "config.toml", mode="w") as f:
        toml.dump(cfg.dict(by_alias=True), f)

    return (
        labels,
        tracks_df,
        graph,
    )


@contextmanager
def _database_dirpath(output_dirpath: Path | None, fov: str):
    """Yield the Ultrack working directory (database and config) for one FOV.

    ``{output_dirpath}/{fov}`` when ``output_dirpath`` is given, and kept. Otherwise a
    fresh directory in this worker's ``$TMPDIR``, deleted on exit: one per FOV,
    because ``$TMPDIR`` can be shared by all of a user's jobs on a node.
    """
    if output_dirpath is not None:
        dirpath = Path(output_dirpath) / fov
        dirpath.mkdir(parents=True, exist_ok=True)
        yield dirpath
        return
    dirpath = Path(tempfile.mkdtemp(prefix=f"biahub-track-{fov}-"))
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath, ignore_errors=True)


def _resolved_ultrack_config(tracking_config) -> dict:
    """Return the full Ultrack config, defaults included, as JSON-safe provenance."""
    config = json.loads(json.dumps(tracking_config.dict(by_alias=True), default=str))
    # The working directory is the per-FOV database, often a deleted temp directory.
    config.get("data", {}).pop("working_dir", None)
    return config


def detect_foreground_segmentation(
    data_dict: dict[str, ArrayLike],
) -> tuple[ArrayLike, ArrayLike]:
    """Pull the foreground mask and contour map out of a preprocessed ``data_dict``.

    Accepts either separate ``foreground`` and ``contour`` channels or a single
    ``foreground_contour`` pair. Raises if neither is present.
    """
    channel_names = data_dict.keys()
    if "foreground" in channel_names and "contour" in channel_names:
        return data_dict["foreground"], data_dict["contour"]
    if "foreground_contour" in channel_names:
        return data_dict["foreground_contour"]
    raise ValueError("Foreground and contour channels are required for tracking.")
