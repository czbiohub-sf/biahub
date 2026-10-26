---
name: track-with-biahub
description: >-
  Track nuclei in a mantis plate with `biahub track` and choose the tracker: ultrack
  (default), tracksdata (ILP), HOCT (learned edge scores + ILP) or Trackastra
  (transformer). Finds the segmented projected plate, builds one config per tracker from
  templates, runs a few positions per tracker on Bruno into that plate side by side,
  compares tracks and divisions
  (and scores divisions against TrackEdit annotations when they exist), checks the
  result visually, and recommends a tracker and settings. Use when asked to "track",
  "compare trackers", "try trackastra / HOCT / tracksdata", "which tracker is better
  for", or to tune tracking on a dataset. For a full pipeline run (reconstruction
  through tracking), use reconstruct-with-nextflow instead.
---

# Track and compare trackers with biahub track

Invocation: `/track-with-biahub <DATASET_OR_PLATE> [--trackers ultrack,trackastra,...]
[--positions R/C/FOV ...]`, or any request to track a plate or compare trackers.

The flow is `biahub project` (the Z=1 plate) -> `biahub segment` (cellpose labels,
`labels/nuclei`, written into that plate) -> `biahub track` once per tracker -> compare.
The A549 pipeline does the first two (and one ultrack run, `labels/tracks_nuclei`); this
skill runs the trackers. Track neither projects nor segments: every tracker reads the same
`labels: nuclei`. `ultrack` re-segments from them and links; `tracksdata`, `hoct` and
`trackastra` link the labels as they are. Each run writes INTO the plate, named by its
config's `output_label` (`nuclei_<tracker>`): `labels/nuclei_<tracker>` (value = track id)
and `tracks_nuclei_<tracker>_<fov>.csv` per position; ultrack also writes
`tracks_nuclei_ultrack.geff`, the other three no GEFF yet. All go through the sibling
filter (`min_sibling_length`), so outputs compare directly.

## 0. Load context

- `references/trackers.md`: what each tracker does, its settings, solver and licence notes,
  what the A549 benchmark found, known pitfalls.
- `references/evaluation.md`: how to compare runs, with and without annotations, and the
  visual check.

## 1. Confirm the environment

- On Bruno (`squeue` works); the biahub checkout on `main`, `source .venv/bin/activate`,
  installed with the `track` (trackers) extra, and `segment` (cellpose) if the plate still
  needs segmenting (`uv sync --extra segment --extra track`): `python -c "import ultrack,
  tracksdata, hoct, trackastra"` succeeds.
- Run launches from tmux or as `sbatch` jobs, never in a login shell that may drop.

## 2. Find the input plate

The plate `biahub track` reads: the projected (Z=1) plate of a mantis-v2 run,
`<project>/<N>-assemble/<dataset>_proj.zarr`, with the segment label image
`labels/nuclei` in every position (`ls <plate>/<R>/<C>/<FOV>/labels`). A549 pipeline runs
have both. Without the labels, segment the chosen positions first with the A549 segment
config (`nextflow/configs/a549/segment.yml`; `run_track.sh` has the two commands); without
the projected plate, run `biahub project` (`nextflow/configs/a549/project.yml`) on the
assembled plate first. hoct and Trackastra also read the image the labels were segmented
from (the first channel segment recorded, e.g. `nuclei_prediction`): check it is in the
plate (`iohub info <plate>/<R>/<C>/<FOV>`). Read one chunk to confirm the store is intact.

## 3. Pick positions and trackers

- Positions: 2-4 FOVs covering each condition (mock and infected wells behave differently;
  infected wells have more false divisions). Prefer FOVs with TrackEdit annotations
  (`references/evaluation.md`), so divisions can be scored.
- Trackers: those asked for; otherwise all four, with ultrack as the reference.

## 4. Build the configs

Copy `templates/track-<tracker>.yml` for each tracker into the run directory. They share
`labels: nuclei`, `output_mode: "2D"` and `min_sibling_length: 3`, and differ only in
`tracker:`, its block and `output_label: nuclei_<tracker>`, so differences come from the
tracker alone. Adapt per dataset: `labels` (the segment model name), distances (in um) for
other pixel sizes, and `image_channel` in `hoct_config` / `trackastra_config` when the
label image was not written by `biahub segment` or another channel should be read
(default: the first channel segment recorded in the labels' `biahub-segment`
provenance). Segmentation settings (cellpose model, diameter, min_size) belong to the
segment config, not here. An `output_label` already in the plate is reused and
overwritten: pick a new one per run to keep. A test keeps the templates on the current
schema.

## 5. Present the plan

Show: the plate and positions (with conditions), the trackers and their non-default
settings and `output_label`s (the label images that will be added to the plate), whether
the positions need segmenting first, the run directory, cost (one job per tracker;
~10-15 min per A549 FOV when track still ran cellpose, less now; tracksdata and HOCT
solve with SCIP, see `references/trackers.md`), and whether annotations
exist for scoring. Wait for a go.

## 6. Scaffold and launch

In a run directory (e.g. `<project>/tracking-comparison/<date>/`), copy the configs and
`templates/run_track.sh`, fill in `BIAHUB`, `PLATE`, `POSITIONS`, run the `--init`s one
after another (concurrent `--init`s on one plate can drop each other's entry from the
positions' label list), then submit one job per tracker:

```bash
for c in track-*.yml; do INIT_ONLY=1 bash run_track.sh $c; done
for c in track-*.yml; do sbatch --job-name=track_$(basename $c .yml) run_track.sh $c; done
```

Each writes into the plate (`labels/<output_label>` and its CSVs); the run directory
holds the configs, the script and the job logs. No `-o`: it would only keep ultrack's
databases. hoct and Trackastra want the GPU; ultrack and tracksdata are CPU only, so
their jobs may go to `cpu` (drop `--gres`). Not `preempted`: tracking cannot resume, so a
preemption restarts it from scratch.

## 7. Monitor

`squeue -u $USER -n track_track-ultrack,...`; the job logs (`track_*_<jobid>.out`) print
`Tracking with <tracker>`, the cells per frame, and one `Sibling filter (min_length=3):
N -> M tracks` line per FOV when done.

## 8. Compare

```bash
python templates/compare_tracks.py --plate <plate> \
  nuclei_ultrack nuclei_tracksdata nuclei_hoct nuclei_trackastra [tracks_nuclei] \
  [--gt "<annotation dir>/{fov}/v*_tracks.csv" --pixel-size 0.1133]
```

Runs are named by their tracked label (`output_label`; `tracks_nuclei` is the pipeline's
own ultrack run). Per run: tracks, divisions, cells per frame, mean track length; with
annotations, division precision / recall / F1 (`references/evaluation.md` for how to
read them). Then check divisions visually on one FOV per condition with
`templates/view_runs.py` (`references/evaluation.md`).

## 9. Report and recommend

Report the comparison table per condition, what the visual check showed, and a
recommendation: which tracker, and which settings to tune next. With no annotations, say
the comparison is counts only. Write the result next to the runs (`RESULT.md`).

## 10. Handle errors

See "Known pitfalls" in `references/trackers.md` (solver fallback, model downloads,
one-child links, 2D only). Do not delete a run directory or a label image in the plate
to retry without asking.
