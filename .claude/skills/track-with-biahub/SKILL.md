---
name: track-with-biahub
description: >-
  Track nuclei in a mantis plate with `biahub track` and choose the tracker: ultrack
  (default), tracksdata (ILP), HOCT (learned edge scores + ILP) or Trackastra
  (transformer). Finds the plate, builds one config per tracker from templates, runs a
  few positions per tracker on the Bruno GPU partition, compares tracks and divisions
  (and scores divisions against TrackEdit annotations when they exist), checks the
  result visually, and recommends a tracker and settings. Use when asked to "track",
  "compare trackers", "try trackastra / HOCT / tracksdata", "which tracker is better
  for", or to tune tracking on a dataset. For a full pipeline run (reconstruction
  through tracking), use reconstruct-with-nextflow instead.
---

# Track and compare trackers with biahub track

Invocation: `/track-with-biahub <DATASET_OR_PLATE> [--trackers ultrack,trackastra,...]
[--positions R/C/FOV ...]`, or any request to track a plate or compare trackers.

Every tracker reads the same input: cellpose labels of the (projected) nuclei channel.
`ultrack` re-segments from them and links; `tracksdata`, `hoct` and `trackastra` link the
labels as they are. All write the same output (labels with value = track id, and
`tracks_<fov>.csv`) and go through the sibling filter (`min_sibling_length`), so outputs
compare directly.

## 0. Load context

- `references/trackers.md`: what each tracker does, its settings, solver and licence notes,
  what the A549 benchmark found, known pitfalls.
- `references/evaluation.md`: how to compare runs, with and without annotations, and the
  visual check.

## 1. Confirm the environment

- On Bruno (`squeue` works); the biahub checkout on `main`, `source .venv/bin/activate`,
  installed with the `track` extra (`uv sync --extra track`): `python -c "import ultrack,
  tracksdata, hoct, trackastra"` succeeds.
- Run launches from tmux or as `sbatch` jobs, never in a login shell that may drop.

## 2. Find the input plate

The plate `biahub track` reads: the assembled plate of a mantis-v2 run
(`<project>/1-preprocess/<N>-assemble/<dataset>.zarr`), with a nuclei channel
(`nuclei_prediction` for virtual staining) and `Phase3D` for focus finding. Check the
channel names (`iohub info <plate>/<R>/<C>/<FOV>`) against the config: some datasets use
`raw `-prefixed fluorescence names. Read one chunk to confirm the store is intact.

## 3. Pick positions and trackers

- Positions: 2-4 FOVs covering each condition (mock and infected wells behave differently;
  infected wells have more false divisions). Prefer FOVs with TrackEdit annotations
  (`references/evaluation.md`), so divisions can be scored.
- Trackers: those asked for; otherwise all four, with ultrack as the reference.

## 4. Build the configs

Copy `templates/track-<tracker>.yml` for each tracker into the run directory. They share
everything but `tracker:` and its block (the mantis-v2 A549 settings: 60-plane Phase3D
focus window, cellpose cpsam_v2 diameter 80 / min_size 2500, `min_sibling_length: 3`), so
differences come from the tracker alone. Adapt per dataset: channel names, cellpose
`diameter`/`min_size` for other cell lines or magnifications, and distances (in um) for
other pixel sizes. A test keeps the templates on the current schema.

## 5. Present the plan

Show: the plate and positions (with conditions), the trackers and their non-default
settings, the run directory, cost (one GPU job per tracker; ~10-15 min per A549 FOV
including cellpose; tracksdata and HOCT solve with SCIP on the GPU nodes, see
`references/trackers.md`), and whether annotations exist for scoring. Wait for a go.

## 6. Scaffold and launch

In a run directory (e.g. `<project>/tracking-comparison/<date>/`), copy the configs and
`templates/run_track.sh`, fill in `BIAHUB`, `PLATE`, `POSITIONS`, and submit one job per
tracker:

```bash
for c in track-*.yml; do sbatch --job-name=track_$(basename $c .yml) run_track.sh $c; done
```

Each writes `<run dir>/track-<tracker>/track.zarr`. Jobs run on `gpu`: tracking cannot
resume, so `preempted` would restart it from scratch.

## 7. Monitor

`squeue -u $USER -n track_track-ultrack,...`; the job logs (`track_*_<jobid>.out`) print
`Tracking with <tracker>`, the cells per frame, and one `Sibling filter (min_length=3):
N -> M tracks` line per FOV when done.

## 8. Compare

```bash
python templates/compare_tracks.py track-*/track.zarr \
  [--gt "<annotation dir>/{fov}/v*_tracks.csv" --pixel-size 0.1133]
```

Per tracker: tracks, divisions, cells per frame, mean track length; with annotations,
division precision / recall / F1 (`references/evaluation.md` for how to read them).
Then check divisions visually on one FOV per condition (`references/evaluation.md`).

## 9. Report and recommend

Report the comparison table per condition, what the visual check showed, and a
recommendation: which tracker, and which settings to tune next. With no annotations, say
the comparison is counts only. Write the result next to the runs (`RESULT.md`).

## 10. Handle errors

See "Known pitfalls" in `references/trackers.md` (solver fallback, model downloads,
one-child links, 2D only). Do not delete a run directory to retry without asking.
