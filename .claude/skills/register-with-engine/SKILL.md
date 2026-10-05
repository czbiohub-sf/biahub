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
  result with `biahub apply-transform`. Use when asked to "register", "estimate the
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
  `approx_transform`) is the legacy inverse direction. Cross that boundary only with
  `Transform.from_inverse` / `Transform.to_inverse`.

## 1. Confirm the environment

- On Bruno (`hostname` shows `gpu-*`/`cpu-*`/login node; `squeue` works).
- The biahub checkout: run from `main` once the registration refactor has merged; until
  then from `refactor-registration-engine/integration`.
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
- `transform.seed`: reuse the previous run's `approx_transform` (inverse direction, as on
  disk) if `1-register/estimate-registration-beads.yml` exists for this dataset (same
  instrument geometry) -- `biahub convert-settings -c <that file> -o <new>.yml` converts
  the whole config; otherwise the template's seed from 2025_09_18 is a reasonable start
  -- the vote seed correction (`seed_correction_settings.mode: votefit`) recovers tens of
  voxels of drift.
- `transform.seed_from`: `input` (default) estimates every timepoint independently from
  the seed, fanned out one job per timepoint. `previous_timepoint` starts each timepoint
  from the previous one's result (the input seed still competes on the first pass), as
  legacy `use_prev_t_transform: true` did -- what most production configs used; it runs
  as ONE sequential job, ~5 min per timepoint by default.
- `time_indices`: `"all"` for production; a short list (e.g. `[0, 82, 200]`) for a smoke
  (with `previous_timepoint` the list must be contiguous).
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
tmux new-session -d -s register-<DATASET> -c <biahub checkout>
tmux send-keys -t register-<DATASET> "source .venv/bin/activate && time biahub estimate-transform --cluster slurm \
  -m <deskew.zarr>/<beads well> -r <reconstruct.zarr>/<beads well> \
  -c <config> -o <out>/transforms.yml 2>&1 \
  | grep -v 'FutureWarning\|transform.estimate(mov_peaks\|Please use' | tee <out>/estimate_transform.log" Enter
```

`--resume` on a relaunch keeps finished timepoints (records in `<out>/timepoints/`,
`<out>/repairs/`) and refuses if the settings or inputs changed; without it a run
starts clean. Do not run two drivers on the same output directory.

Time limits: an `-sb` sbatch file that sets `--time` wins over every phase's default,
including the sequential `previous_timepoint` job -- give large volumes enough
(e.g. 8 min per timepoint for 100x2048x1252 A549 volumes). Keep the number of jobs you
have on the queue modest: an overloaded partition makes jobs hit their time limit.

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
  before -> after from the journal, `candidate_failures`), `stand_ins` (timepoints with
  no transform of their own and what was written instead).
- In `transforms.yml` each entry has a `status`: `accepted`, or `unreliable` (no good
  transform -- the pipeline's best result, or a stand-in when `filled_from` is set,
  e.g. `seed` = the approximate transform, with the error in `note`). Empty frames
  (no data) are reported as such. A timepoint is never dropped.
- per-timepoint `metrics.median_residual` (voxels): should sit well under 1 for good
  timepoints; `n_matched` should be close to the detected bead count.

If a previous `1-register/registration_settings.yml` exists: per-timepoint |dT| against
it (median, p95; both files are inverse-direction so compare directly), and per-timepoint
scores against `quality_scores.csv` or `xyz_transforms/*.score` when present. Equal
scores with different matrices are expected at the metric's resolution; a systematic
|dT| above ~2 voxels on unflagged timepoints needs a look.

## 9. Optional: apply

`--apply`: `biahub apply-transform -m <deskew.zarr>/*/*/* -r <reconstruct.zarr>/*/*/* -c <out>/transforms.yml -o <dataset>/1-preprocess/light-sheet/raw/1-register-engine/<DATASET>.zarr`
registers every light-sheet position onto the phase grid with the estimated per-timepoint
transforms: reference channels copied, and every moving channel transformed
(`--channels` to pick); canvas = overlap shared by the applied transforms,
`--keep-overhang` to keep the full grid. Timepoints written with a non-accepted
transform are printed and recorded in the output metadata. Without `-r` the same command
stabilizes a store onto its own grid; estimating several `-m` positions writes a
transforms list per position, applied per position.

To replace a few bad timepoints with another method's result (e.g. beads failed, manual
or ants worked): estimate those timepoints alone, then
`biahub substitute-transforms -c <out>/transforms.yml -s <other>/transforms.yml -o <final>.yml`.

## 10. Wrap up

Report: beads well, T, wall time, score median/min, flagged and rescued counts (with
the largest before -> after), any `stand_ins` / unreliable timepoints, the comparison against the
previous registration, and the output paths. Leave the tmux session for inspection;
kill it only after the summary. Record anything surprising in memory
(`registration-engine-refactor`) with the dataset name.
