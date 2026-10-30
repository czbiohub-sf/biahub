"""Trackastra: link fixed labels with a transformer and greedy or ILP linking."""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from biahub.settings import TrackastraConfig
from biahub.tracking.graph import _empty_result, _isolated_tracks
from biahub.tracking.postprocess import _merge_single_children, _upsample_result
from biahub.tracking.preprocess import _downsample


def warm_trackastra_weights(model: str) -> None:
    """Download Trackastra weights into the shared cache once (from --init)."""
    from trackastra.model import Trackastra

    Trackastra.from_pretrained(model, device="cpu")
    click.echo(f"Trackastra weights ready: {model}")


def run_trackastra(
    labels: np.ndarray, images: np.ndarray, config: TrackastraConfig
) -> tuple[np.ndarray, pd.DataFrame]:
    """Link fixed (T, Y, X) instance labels with Trackastra.

    With ``config.downsample`` > 1 the labels and image are shrunk first and the result
    is mapped back to full resolution (as :func:`run_hoct`).
    """
    factor = config.downsample
    if factor == 1:
        return _run_trackastra(labels, images, config)
    labels_small, images_small = _downsample(labels, images, factor)
    tracked, tracks = _run_trackastra(labels_small, images_small, config)
    return _upsample_result(tracked, tracks, labels_small, labels, factor)


def _run_trackastra(
    labels: np.ndarray, images: np.ndarray, config: TrackastraConfig
) -> tuple[np.ndarray, pd.DataFrame]:
    """Link fixed (T, Y, X) instance labels with Trackastra.

    ``images`` is the (T, Y, X) image the labels were segmented from. Trackastra's
    ``graph_to_ctc`` relabels the masks by track and gives each track's parent; the
    tracks table takes each track's centroid per frame.
    """
    if not labels.any():
        return _empty_result(labels.shape)
    if labels.shape[0] < 2:
        # Trackastra needs two frames to link; one frame is its detections, unlinked.
        return _isolated_tracks(labels)

    import torch

    from skimage.measure import regionprops_table
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if Path(config.model).exists():
        model = Trackastra.from_folder(Path(config.model), device=device)
    else:
        model = Trackastra.from_pretrained(config.model, device=device)
    # track() also returns a rendered mask movie; keep only the graph so it is freed.
    graph = model.track(images, labels, mode=config.mode)[0]
    ctc, tracked = graph_to_ctc(graph, labels)
    tracked = np.asarray(tracked).astype(np.int32)

    rows = []
    for t in range(tracked.shape[0]):
        props = regionprops_table(tracked[t], properties=("label", "centroid"))
        rows.append(
            pd.DataFrame(
                {
                    "track_id": props["label"],
                    "t": t,
                    "y": props["centroid-0"],
                    "x": props["centroid-1"],
                }
            )
        )
    tracks = pd.concat(rows, ignore_index=True)
    # CTC marks roots with parent 0; ultrack's tables use -1.
    # graph_to_ctc's table is CTC's res_track.txt: label, t1, t2, parent.
    parent = dict(zip(ctc["label"].astype(int), ctc["parent"].astype(int), strict=True))
    tracks["parent_track_id"] = tracks.track_id.map(lambda k: parent.get(int(k), 0) or -1)
    tracks, tracked = _merge_single_children(tracks, tracked)
    return tracked, tracks
