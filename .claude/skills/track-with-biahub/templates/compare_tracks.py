"""Compare biahub track outputs: per-tracker counts, and division scores if annotated.

Usage:
  python compare_tracks.py track-ultrack/track.zarr track-trackastra/track.zarr ...
  python compare_tracks.py RUN.zarr ... --gt "/path/annotation/{fov}/v*_tracks.csv" \
      --pixel-size 0.1133

Each run is a track.zarr with <row>/<col>/<fov>/tracks_<row>_<col>_<fov>.csv. Without
--gt it prints, per run: FOVs, tracks, divisions, cells per frame, mean track length.
With --gt (a glob, {fov} = row_col_fov, e.g. A_3_000002) it also matches divisions to
the annotated ones: a division is (the daughters' first frame, the parent's last
position); a predicted and an annotated division match when their frames differ by at
most --max-frames and their positions by at most --max-um (one-to-one, closest first).
"""

import argparse
import glob

from pathlib import Path

import numpy as np
import pandas as pd


def division_events(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per division with two daughters: t (daughters' first frame), y, x."""
    first = tracks.groupby("track_id").agg(
        t0=("t", "min"), parent=("parent_track_id", "first")
    )
    last = tracks.sort_values("t").groupby("track_id").last()
    rows = []
    for parent, daughters in first[first.parent > 0].groupby("parent"):
        if len(daughters) == 2 and parent in last.index:
            rows.append(
                (int(daughters.t0.min()), last.loc[parent, "y"], last.loc[parent, "x"])
            )
    return pd.DataFrame(rows, columns=["t", "y", "x"])


def match(pred: pd.DataFrame, gt: pd.DataFrame, max_frames: int, max_px: float) -> int:
    """Count one-to-one matches, closest pairs first."""
    pairs = []
    for i, p in pred.iterrows():
        for j, g in gt.iterrows():
            d = np.hypot(p.y - g.y, p.x - g.x)
            if abs(p.t - g.t) <= max_frames and d <= max_px:
                pairs.append((d, i, j))
    used_p, used_g = set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_p)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", type=Path, help="track.zarr outputs")
    ap.add_argument(
        "--gt", help='annotated tracks CSV glob, with {fov}, e.g. ".../{fov}/v*_tracks.csv"'
    )
    ap.add_argument(
        "--pixel-size", type=float, default=0.1133, help="um per pixel (mantis v2: 0.1133)"
    )
    ap.add_argument("--max-frames", type=int, default=2)
    ap.add_argument("--max-um", type=float, default=10.0)
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        for csv in sorted(run.glob("*/*/*/tracks_*.csv")):
            fov = csv.stem.removeprefix("tracks_")
            tracks = pd.read_csv(csv)
            events = division_events(tracks)
            row = {
                "run": run.parent.name if run.name == "track.zarr" else run.name,
                "fov": fov,
                "tracks": tracks.track_id.nunique(),
                "divisions": len(events),
                "cells_per_frame": tracks.groupby("t").size().mean(),
                "track_length": tracks.groupby("track_id").size().mean(),
            }
            if args.gt:
                matches = glob.glob(args.gt.format(fov=fov))
                if matches:
                    gt = division_events(pd.read_csv(sorted(matches)[-1]))
                    tp = match(events, gt, args.max_frames, args.max_um / args.pixel_size)
                    row.update(gt_divisions=len(gt), matched=tp)
            rows.append(row)
    if not rows:
        raise SystemExit("No tracks_*.csv found under the given runs.")

    d = pd.DataFrame(rows)
    g = d.groupby("run", sort=False)
    out = g.agg(
        fovs=("fov", "size"),
        tracks=("tracks", "sum"),
        divisions=("divisions", "sum"),
        cells_per_frame=("cells_per_frame", "mean"),
        track_length=("track_length", "mean"),
    )
    if "matched" in d:
        scored = d.dropna(subset=["matched"]).groupby("run", sort=False)
        out["gt_fovs"] = scored.size()
        out["div_precision"] = scored.matched.sum() / scored.divisions.sum()
        out["div_recall"] = scored.matched.sum() / scored.gt_divisions.sum()
        out["div_f1"] = (
            2 * out.div_precision * out.div_recall / (out.div_precision + out.div_recall)
        )
    print(out.round(3).to_string())
    print("\nper FOV:")
    print(d.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
