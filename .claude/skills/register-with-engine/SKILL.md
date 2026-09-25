---
name: register-with-engine
description: >-
  Estimate the light-sheet -> label-free registration of a mantis dataset with the
  registration engine (`biahub estimate-transform`: beads matching with the spectral
  arm, adaptive flagging, repair, SLURM fan-out with --resume) on the Bruno HPC
  cluster. Locates the dataset's deskewed light-sheet and reconstructed phase stores
  under /hpc/projects/tlg2_mantis, identifies the beads well, builds the config from
  the production template, launches the run in a tmux session, monitors the
  per-timepoint records, reads the report and run journal, compares against any
  previous registration_settings.yml, and reports a verdict. Optionally applies the
  result with `biahub stabilize`. Use when asked to "register", "estimate the
  registration for", or "run registration on" a named mantis dataset.
---

# Register a mantis dataset with the registration engine

Invocation: `/register-with-engine <DATASET_NAME> [--well R/C/FOV] [--apply]`, or any
request to estimate or run registration for a mantis dataset by name.

Registration is estimated once per timepoint on the **beads well** (both optical arms
see the beads) and applied to every embryo well. The engine's own quality score is a
bead-overlap ratio (quantized at ~1/N per bead); treat differences of one bead as noise.

## 0. Load context

- Memory `registration-engine-refactor` (design, conventions, validated numbers) and
  `.local/registration-refactor/CONTEXT.md` if present.
- The transform direction rule: everything the engine returns is forward
  (moving -> reference); everything on disk (`registration_settings.yml`,
  `approx_transform`) is the legacy pull direction. Cross that boundary only with
  `Transform.from_legacy_pull` / `Transform.to_legacy_pull`.

## 1. Confirm the environment

- On Bruno (`hostname` shows `gpu-*`/`cpu-*`/login node; `squeue` works).
- The biahub checkout: run from `main` once the engine stack (#366-#378) has merged;
  until then from `refactor-registration-engine/12-cleanup` (or the top of the stack).
  `git status` clean apart from untracked scratch; `source .venv/bin/activate`;
  `biahub estimate-transform --help` renders.
- Run the driver inside tmux. SSH drops have killed sessions before.

## 2. Find the inputs

Under `/hpc/projects/tlg2_mantis/<DATASET>/1-preprocess/`:

- moving: `light-sheet/raw/0-deskew/<DATASET>.zarr` -- channel `GFP EX488 EM525-45`
- reference: `label-free/0-reconstruct/<DATASET>.zarr` -- channel `Phase3D`

Both stores must exist and be intact for the beads well (check `.zarray`/`zarr.json`
and one chunk read). Several intracellular_dashboard datasets have had `1-preprocess`
intermediates cleaned up; say so and stop if either store is missing.

## 3. Identify the beads well

Convention on tlg2 mantis plates: `0/8/000000`. Verify rather than assume: read
`1-register/README.md` or `PROVENANCE.txt` if a previous registration exists, or run
`detect_peaks` on one timepoint of the candidate well with the template's
`source_peaks_settings` and confirm tens of peaks (an embryo well gives none or a few).
If `--well` was given, use it.

## 4. Build the config

Start from `templates/estimate-transform-beads.yml` in this skill. It is the
production beads config in the engine's `EstimateTransformSettings` schema (an old
`estimate-registration` config converts with `biahub convert-settings`), with `spectral_arm: always` (measured on 2025_09_18: median score 0.875 vs
0.857 without, 187/240 timepoints identical to production).

Adjust:
- `approx_transform`: reuse the previous run's seed if `1-register/estimate-registration-beads.yml`
  exists for this dataset (same instrument geometry); otherwise the template's seed
  from 2025_09_18 is a reasonable start -- the vote seed correction
  (`seed_correction_settings.mode: votefit`) recovers tens of voxels of drift.
- `time_indices`: `"all"` for production; a short list (e.g. `[0, 82, 200]`) for a smoke.
- Do NOT add fields from the hardened production YAML (`beads_strategy`,
  `repair_pass_settings`, `sweep_fallback_settings`, `hungarian_match_settings.qc_settings`):
  the model is `extra="forbid"` and will reject them.

Write it to `<DATASET>/1-preprocess/light-sheet/raw/1-register-engine/estimate-transform-beads.yml`
(a new directory next to the legacy `1-register`, never overwriting it).

## 5. Present the plan

Show: dataset, beads well, moving/reference stores and channels, T and volume shape, the
config path with the non-default fields, the output directory, expected cost
(~2 min per timepoint per job, 100 concurrent; repair jobs afterwards, one per flagged
timepoint), and what will be compared against (a previous `registration_settings.yml`
if one exists). Wait for a go.

## 6. Launch in tmux

```bash
tmux new-session -d -s register-<DATASET> -c /hpc/mydata/taylla.theodoro/repo/biahub
tmux send-keys -t register-<DATASET> "source .venv/bin/activate && time biahub estimate-transform --cluster slurm \
  -m <deskew.zarr>/<beads well> -r <reconstruct.zarr>/<beads well> \
  -c <config> -o <out>/registration_settings.yml 2>&1 \
  | grep -v 'FutureWarning\|transform.estimate(mov_peaks\|Please use' | tee <out>/estimate_transform.log" Enter
```

`--resume` on a relaunch keeps finished timepoints (records in `<out>/timepoints/`,
`<out>/repairs/`). Do not run two drivers on the same output directory.

## 7. Monitor

- `ls <out>/timepoints | wc -l` vs T; `squeue -u $USER -h -o "%j %t" | grep estimate_transform | sort | uniq -c`.
- Phase 2 starts when the driver prints `repair: N of T timepoints flagged`; one job per
  flagged timepoint. Done when `<out>/estimate_transform_report.json` exists.
- A record with `"error"` set is a timepoint whose estimate failed (usually too few
  beads); the repair phase tries to rescue it. `"arm"` says which estimator won.
- Job failures are recorded per timepoint, never fatal; look at
  `<out>/slurm_output/*_log.err` only if many timepoints report `job failed`.

## 8. Read the result

From `estimate_transform_report.json` and `run_journal.json`:
- score distribution (median, min), `flagged`, `repairs` (`accepted`, `source`,
  before -> after from the journal, `candidate_failures`), `filled_from_neighbour`
  (timepoints with no transform at all -- these got a neighbour's matrix).
- per-timepoint `metrics.median_residual` (voxels): should sit well under 1 for good
  timepoints; `n_matched` should be close to the detected bead count.

If a previous `1-register/registration_settings.yml` exists: per-timepoint |dT| against
it (median, p95; both files are pull-direction so compare directly), and per-timepoint
scores against `quality_scores.csv` or `xyz_transforms/*.score` when present. Equal
scores with different matrices are expected at the metric's resolution; a systematic
|dT| above ~2 voxels on unflagged timepoints needs a look.

## 9. Optional: apply

`--apply`: `biahub apply-transform -m <deskew.zarr>/*/*/* -r <reconstruct.zarr>/*/*/* -c <out>/transforms.yml -o <dataset>/1-preprocess/light-sheet/raw/1-register-engine/<DATASET>.zarr`
registers every light-sheet position onto the phase grid with the estimated per-timepoint
transforms (reference channels copied, moving channels transformed; canvas = overlap shared by
the applied transforms, `--keep-overhang` to keep the full grid). Without `-r` the same
command stabilizes a store onto its own grid.

## 10. Wrap up

Report: beads well, T, wall time, score median/min, flagged and rescued counts (with
the largest before -> after), any `filled_from_neighbour`, the comparison against the
previous registration, and the output paths. Leave the tmux session for inspection;
kill it only after the summary. Record anything surprising in memory
(`registration-engine-refactor`) with the dataset name.
