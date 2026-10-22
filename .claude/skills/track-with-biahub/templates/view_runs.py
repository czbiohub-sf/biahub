"""Show finished biahub track runs of one position side by side in napari.

Usage (in a desktop session):
  python view_runs.py --plate /path/<dataset>.zarr --fov A/3/000002 \
      track-ultrack/track.zarr track-trackastra/track.zarr [--channel nuclei_prediction]

One panel per run: the input channel (Z mean-projected), the run's labels, its tracks,
and a yellow circle on every division it calls (daughters' first frame, at the parent's
last position). Time and zoom are shared, so divisions one tracker calls and another does
not stand out. Reads only.
"""

import argparse

from pathlib import Path

import napari
import numpy as np
import pandas as pd

from iohub import open_ome_zarr


def division_points(tracks: pd.DataFrame) -> np.ndarray:
    """Return (t, y, x) per division: daughters' first frame, parent's last position."""
    first = tracks.groupby("track_id").agg(
        t0=("t", "min"), parent=("parent_track_id", "first")
    )
    last = tracks.sort_values("t").groupby("track_id").last()
    rows = [
        (d.t0.min(), last.loc[p, "y"], last.loc[p, "x"])
        for p, d in first[first.parent > 0].groupby("parent")
        if p in last.index
    ]
    return np.array(rows, dtype=float).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", type=Path, help="track.zarr outputs")
    ap.add_argument("--plate", type=Path, required=True, help="the plate the runs tracked")
    ap.add_argument("--fov", required=True, help="row/col/fov, e.g. A/3/000002")
    ap.add_argument("--channel", default="nuclei_prediction")
    args = ap.parse_args()

    with open_ome_zarr(args.plate / args.fov, mode="r") as ds:
        c = ds.channel_names.index(args.channel)
        image = np.stack(
            [np.asarray(ds.data[t, c]).mean(axis=0) for t in range(ds.data.shape[0])]
        )

    viewer = napari.Viewer(
        title=f"{args.fov}: " + " | ".join(r.parent.name for r in args.runs)
    )
    for run in args.runs:
        name = run.parent.name if run.name == "track.zarr" else run.name
        with open_ome_zarr(run / args.fov, mode="r") as ds:
            labels = np.asarray(ds.data[:, 0, 0])
        tracks = pd.read_csv(next((run / args.fov).glob("tracks_*.csv")))
        graph = {
            int(k): [int(p)]
            for k, p in tracks.groupby("track_id").parent_track_id.first().items()
            if p > 0
        }
        divisions = division_points(tracks)
        viewer.add_image(image, name=f"{name}: {args.channel}", colormap="gray")
        viewer.add_labels(labels, name=f"{name}: labels", opacity=0.35)
        if len(tracks):
            viewer.add_tracks(
                tracks[["track_id", "t", "y", "x"]].to_numpy(),
                graph=graph,
                name=f"{name}: tracks ({tracks.track_id.nunique()})",
                tail_length=10,
            )
        viewer.add_points(
            divisions,
            name=f"{name}: divisions ({len(divisions)})",
            size=40,
            face_color="transparent",
            border_color="yellow",
            out_of_slice_display=False,
        )
    viewer.canvas.grid.enabled = True
    viewer.canvas.grid.stride = 4 if len(viewer.layers) == 4 * len(args.runs) else 3
    viewer.canvas.grid.shape = (1, len(args.runs))
    napari.run()


if __name__ == "__main__":
    main()
