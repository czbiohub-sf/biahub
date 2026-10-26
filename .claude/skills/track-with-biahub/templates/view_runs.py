"""Show finished biahub track runs of one position side by side in napari.

Usage (in a desktop session):
  python view_runs.py --plate /path/<dataset>_proj.zarr --fov A/3/000002 \
      nuclei_ultrack nuclei_trackastra [--channel nuclei_prediction]

A run is the name of a tracked label image in the plate (the config's output_label, or
tracks_<labels> without one), found as compare_tracks.py finds it: labels/<run> and
tracks_<run>_<fov>.csv, or tracks_<fov>.csv for a run named tracks_*. One panel per run: the input channel (Z mean-projected), the run's labels, its tracks,
and a yellow circle on every division it calls (daughters' first frame, at the parent's
last position). Time and zoom are shared, so divisions one tracker calls and another does
not stand out. Reads only.
"""

import argparse

from pathlib import Path

import napari
import numpy as np
import pandas as pd
import zarr

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


def tracks_csv(position: Path, run: str) -> Path:
    """Return the run's tracks CSV: tracks_<run>_<fov>.csv, else tracks_<fov>.csv (tracks_*)."""
    fov = "_".join(position.parts[-3:])
    named = position / f"tracks_{run}_{fov}.csv"
    if named.exists() or not run.startswith("tracks_"):
        return named
    return position / f"tracks_{fov}.csv"


def read_labels(position: Path, run: str) -> np.ndarray:
    """Return labels/<run> as (T, Y, X): (T, 1, Z, Y, X) labels, or older (T, Z, Y, X)."""
    labels = zarr.open_array(position / "labels" / run / "0", mode="r")
    return np.asarray(labels[:, 0, 0] if labels.ndim == 5 else labels[:, 0])


LAYERS_PER_RUN = 4


def run_layers(
    name: str, image, labels, tracks: pd.DataFrame
) -> list[tuple[str, object, dict]]:
    """Return the run's panel as (layer type, data, kwargs): always four layers.

    A run without tracks gets an empty points layer in the tracks slot, so every run fills
    the same number of grid slots and panels never mix layers of different runs.
    """
    divisions = division_points(tracks)
    layers = [
        ("image", image, {"name": f"{name}: input", "colormap": "gray"}),
        ("labels", labels, {"name": f"{name}: labels", "opacity": 0.35}),
    ]
    if len(tracks):
        graph = {
            int(k): [int(p)]
            for k, p in tracks.groupby("track_id").parent_track_id.first().items()
            if p > 0
        }
        layers.append(
            (
                "tracks",
                tracks[["track_id", "t", "y", "x"]].to_numpy(dtype=float),
                {
                    "name": f"{name}: tracks ({tracks.track_id.nunique()})",
                    "graph": graph,
                    "tail_length": 10,
                },
            )
        )
    else:
        layers.append(("points", np.empty((0, 3)), {"name": f"{name}: tracks (0)"}))
    layers.append(
        (
            "points",
            divisions,
            {
                "name": f"{name}: divisions ({len(divisions)})",
                "size": 40,
                "face_color": "transparent",
                "border_color": "yellow",
                "out_of_slice_display": False,
            },
        )
    )
    return layers


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", help="tracked label names (output_label values)")
    ap.add_argument("--plate", type=Path, required=True, help="the plate the runs wrote into")
    ap.add_argument("--fov", required=True, help="row/col/fov, e.g. A/3/000002")
    ap.add_argument("--channel", default="nuclei_prediction")
    args = ap.parse_args()

    with open_ome_zarr(args.plate / args.fov, mode="r") as ds:
        c = ds.channel_names.index(args.channel)
        image = np.stack(
            [np.asarray(ds.data[t, c]).mean(axis=0) for t in range(ds.data.shape[0])]
        )

    position = args.plate / args.fov
    viewer = napari.Viewer(title=f"{args.fov}: " + " | ".join(args.runs))
    for run in args.runs:
        labels = read_labels(position, run)
        tracks = pd.read_csv(tracks_csv(position, run))
        for kind, data, kwargs in run_layers(run, image, labels, tracks):
            getattr(viewer, f"add_{kind}")(data, **kwargs)
    viewer.canvas.grid.enabled = True
    viewer.canvas.grid.stride = LAYERS_PER_RUN  # one panel per run
    viewer.canvas.grid.shape = (1, len(args.runs))
    napari.run()


if __name__ == "__main__":
    main()
