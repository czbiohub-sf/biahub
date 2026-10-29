"""Compare biahub track runs of one plate: per-run counts, and division scores if annotated.

Usage:
  python compare_tracks.py --plate <dataset>_proj.zarr nuclei_ultrack nuclei_trackastra ...
  python compare_tracks.py --plate PLATE RUN ... --gt "/path/annotation/{fov}/v*_tracks.csv" \
      --pixel-size 0.1133

A run is the name of a tracked label image in the plate, i.e. the config's output_label
(or tracks_<labels> for a run without one, such as the pipeline's tracks_nuclei). Its
tracks are, in every position that has labels/<run>:
  <row>/<col>/<fov>/tracks_<run>_<row>_<col>_<fov>.csv   (written with output_label)
  <row>/<col>/<fov>/tracks_<row>_<col>_<fov>.csv          (fallback, only for a run
                                                           named tracks_*: no output_label)
Without --gt it prints, per run: FOVs, tracks, divisions, cells per frame, mean track
length. With --gt (a glob, {fov} = row_col_fov, e.g. A_3_000002) it also matches divisions
to the annotated ones: a division is (the daughters' first frame, the parent's last
position); a predicted and an annotated division match when their frames differ by at
most --max-frames and their positions by at most --max-um (one-to-one, closest first).
"""

import argparse
import glob

from pathlib import Path

import numpy as np
import pandas as pd

from natsort import natsorted


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


def run_positions(plate: Path, run: str) -> list[Path]:
    """Return the plate's positions that have the tracked label image labels/<run>."""
    return sorted(p.parents[1] for p in plate.glob(f"*/*/*/labels/{run}"))


def tracks_csv(position: Path, run: str) -> Path | None:
    """Return the run's tracks CSV in a position, or None (rule in the module docstring)."""
    fov = "_".join(position.parts[-3:])
    named = position / f"tracks_{run}_{fov}.csv"
    if named.exists():
        return named
    default = position / f"tracks_{fov}.csv"
    if run.startswith("tracks_") and default.exists():
        return default
    return None


def cells_per_frame(tracks: pd.DataFrame, n_frames: int) -> float:
    """Return detections per frame over the whole movie, empty frames included."""
    return len(tracks) / n_frames


def n_frames(position: Path, tracks: pd.DataFrame) -> int:
    """Return the movie length from the position's image, else from the tracks."""
    try:
        from iohub import open_ome_zarr

        with open_ome_zarr(position, mode="r") as ds:
            return int(ds.data.shape[0])
    except Exception:
        return int(tracks.t.max()) + 1


def latest(paths: list[str]) -> str:
    """Return the newest annotation version (natural order: v10 after v9)."""
    return natsorted(paths)[-1]


def division_f1(matched: int, predicted: int, annotated: int) -> float:
    """Return division F1 = 2 * matched / (predicted + annotated); 0 when nothing matches."""
    total = predicted + annotated
    return 2 * matched / total if total else float("nan")


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
    ap.add_argument("--plate", type=Path, required=True, help="the plate the runs wrote into")
    ap.add_argument("runs", nargs="+", help="tracked label names (output_label values)")
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
        csvs = [c for p in run_positions(args.plate, run) if (c := tracks_csv(p, run))]
        if not csvs:
            raise SystemExit(f"No tracks CSV for run {run!r} in {args.plate}.")
        for csv in csvs:
            fov = "_".join(csv.parent.parts[-3:])
            tracks = pd.read_csv(csv)
            events = division_events(tracks)
            row = {
                "run": run,
                "fov": fov,
                "tracks": tracks.track_id.nunique(),
                "detections": len(tracks),
                "divisions": len(events),
                "cells_per_frame": cells_per_frame(tracks, n_frames(csv.parent, tracks)),
                "track_length": tracks.groupby("track_id").size().mean(),
            }
            if args.gt:
                matches = glob.glob(args.gt.format(fov=fov))
                if matches:
                    gt = division_events(pd.read_csv(latest(matches)))
                    tp = match(events, gt, args.max_frames, args.max_um / args.pixel_size)
                    row.update(gt_divisions=len(gt), matched=tp)
            rows.append(row)
    out = summarize(pd.DataFrame(rows))
    print(out.round(3).to_string())
    print("\nper FOV:")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """Return per-run totals; division scores for runs with annotated FOVs (NaN otherwise)."""
    g = d.groupby("run", sort=False)
    out = g.agg(
        fovs=("fov", "size"),
        tracks=("tracks", "sum"),
        divisions=("divisions", "sum"),
        detections=("detections", "sum"),
        cells_per_frame=("cells_per_frame", "mean"),
    )
    # Mean length of the run's tracks: weighted by each FOV's track count.
    out["track_length"] = out.detections / out.tracks
    if "matched" in d:
        scored = d.dropna(subset=["matched"]).groupby("run", sort=False)
        matched, predicted = scored.matched.sum(), scored.divisions.sum()
        annotated = scored.gt_divisions.sum()
        out["gt_fovs"] = scored.size()
        out["div_precision"] = matched / predicted
        out["div_recall"] = matched / annotated
        out["div_f1"] = pd.Series(
            {
                run: division_f1(matched[run], predicted[run], annotated[run])
                for run in matched.index
            }
        )
    return out


if __name__ == "__main__":
    main()
