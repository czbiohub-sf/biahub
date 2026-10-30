"""What happens to the tracks after linking.

The sibling filter, the one-child merge, and mapping tracks found on downsampled inputs
back to full resolution.
"""

import click
import numpy as np
import pandas as pd

from numpy.typing import ArrayLike


def filter_short_siblings(
    tracks_df: pd.DataFrame, labels: ArrayLike, min_length: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Remove short daughter tracks at divisions, in tracks and labels.

    At each division where exactly one daughter lives ``min_length`` frames or fewer
    and does not divide again, that daughter is dropped (its labels set to 0) and the
    other daughter is merged into the parent. If both daughters are that short, both
    are kept. This is ultrack's ``filter_short_sibling_tracks``, which counts a
    daughter of exactly ``min_length`` frames as short.
    """
    from ultrack.tracks import filter_short_sibling_tracks

    n_before = tracks_df.track_id.nunique()
    tracks_df, labels = filter_short_sibling_tracks(tracks_df, min_length, segments=labels)
    tracks_df = tracks_df.sort_values(["track_id", "t"]).reset_index(drop=True)
    click.echo(
        f"Sibling filter (min_length={min_length}): {n_before} -> "
        f"{tracks_df.track_id.nunique()} tracks"
    )
    return tracks_df, np.asarray(labels)


def _merge_single_children(
    tracks: pd.DataFrame, labels: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Merge every track that is its parent's only child into the parent.

    Linking across skipped frames (HOCT's ``max_delta_t``, CTC conversion at gaps)
    splits one track into parent -> one child. ultrack's tables, and the sibling filter
    built on them, allow only 0 or 2 children, so the child takes its parent's id in the
    tracks and the labels; chains collapse to their first track.
    """
    linked = tracks[tracks.parent_track_id != -1]
    kids = linked.groupby("parent_track_id").track_id.unique()
    into = {int(k[0]): int(parent) for parent, k in kids.items() if len(k) == 1}
    if not into:
        return tracks, labels

    def root(track_id: int) -> int:
        while track_id in into:
            track_id = into[track_id]
        return track_id

    original_parent = tracks.groupby("track_id").parent_track_id.first().astype(int)
    tracks = tracks.copy()
    tracks["track_id"] = tracks.track_id.map(lambda k: root(int(k)))
    tracks["parent_track_id"] = tracks.track_id.map(
        lambda k: -1 if original_parent[k] == -1 else root(int(original_parent[k]))
    )
    lut = np.arange(int(labels.max()) + 1, dtype=labels.dtype)
    for child in into:
        if child < len(lut):
            lut[child] = root(child)
    tracks = tracks.sort_values(["track_id", "t"]).reset_index(drop=True)
    return tracks, lut[labels]


def _upsample_result(
    tracked_small: np.ndarray,
    tracks: pd.DataFrame,
    labels_small: np.ndarray,
    labels: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Map tracks found on downsampled labels back to the full-resolution labels.

    Each original label id takes the track id it got on the downsampled frame, so the
    tracked labels keep the full-resolution masks; positions are rescaled by ``factor``.
    """
    tracked = np.zeros(labels.shape, dtype=np.int32)
    for t in range(labels.shape[0]):
        small, ids = labels_small[t].ravel(), tracked_small[t].ravel()
        both = (small > 0) & (ids > 0)
        if not both.any():
            continue
        pairs = np.unique(np.stack([small[both], ids[both]]), axis=1)
        lut = np.zeros(int(labels[t].max()) + 1, dtype=np.int32)
        lut[pairs[0]] = pairs[1]
        tracked[t] = lut[labels[t]]
    tracks = tracks.assign(y=tracks.y * factor, x=tracks.x * factor)
    return tracked, tracks
