# Comparing tracking runs

## With annotations (preferred)

TrackEdit-corrected tracks are the reference. Known annotated mantis-v2 FOVs:

- 2026_04_28_A549_SEC61B_DENV: `5-track/annotation/2026_04_28_A549_SEC61B_DENV_config_tracking/<fov>/v*_tracks.csv`
  for `A_2_001002`, `A_2_002002` (mock) and `A_3_000002`, `A_3_001002` (DENV); 78 divisions.
  Read-only: never write in an annotation folder.

`templates/compare_tracks.py --gt "<dir>/{fov}/v*_tracks.csv"` matches divisions: a division
is the daughters' first frame and the parent's last position; predicted and annotated
divisions match within 2 frames and 10 um, one-to-one. On the A549 FOVs it reproduces the
benchmark's scorer within 0.01 F1.

How to read it:
- **precision** low: false divisions (a virtual-staining hallucination or a neighbour linked as
  a daughter, a lobed nucleus split for a frame). The sibling filter removes the short ones.
- **recall** low: missed divisions (daughters linked as one track, or the mitotic cell lost).
- report mock and infected wells separately: infected wells usually have more false
  divisions.
- the annotations were curated from an ultrack run, which may favour ultrack slightly.

## Without annotations

Counts only, so say so in the report:
- `cells_per_frame` should agree across trackers (they share the labels); a large difference
  means a tracker drops or duplicates objects.
- `tracks` vs cells per frame: more tracks per cell means more fragmentation.
- `divisions`: compare with the expected division rate (A549 divides roughly once a day); a tracker
  with far more divisions than the others is likely calling false ones.
- `track_length`: longer is better, if divisions and cells per frame look right.

## Visual check

On one FOV per condition, open the finished runs side by side:

```bash
python templates/view_runs.py --plate <plate> --fov <row/col/fov> track-*/track.zarr
```

One panel per run (the input channel, the run's labels, its tracks, and a yellow circle on
every division it calls), with shared time and zoom. Look at the divisions one tracker calls
and another does not: ten of those say more about which tracker to trust than the counts.
To tune one tracker on a single FOV, edit its `track-<tracker>.yml`, rerun just that position
into a new output, and view it next to the previous run:

```bash
biahub track --init -i <plate>/<row/col/fov> -o track-<tracker>-v2/track.zarr -c track-<tracker>.yml
biahub track --cluster debug -i <plate>/<row/col/fov> -o track-<tracker>-v2/track.zarr -c track-<tracker>.yml
python templates/view_runs.py --plate <plate> --fov <row/col/fov> track-<tracker>/track.zarr track-<tracker>-v2/track.zarr
```

(`scripts/debug_track.py` predates the current track settings and runs ultrack only.)
