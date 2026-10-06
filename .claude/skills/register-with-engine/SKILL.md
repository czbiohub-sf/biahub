---
name: register-with-engine
description: >-
  Estimate and apply the light-sheet -> label-free registration (or the stabilization)
  of a mantis dataset with biahub's registration engine (`biahub estimate-transform`
  then `biahub apply-transform`) on the Bruno HPC cluster, through the Nextflow workflow
  `nextflow/registration.nf` (or the plain CLI for quick or interactive runs). Finds the
  deskewed light-sheet and reconstructed phase stores, identifies the beads well, builds
  the config from a template, scaffolds a run script, launches it in tmux, monitors it,
  reads the report (scores, flagged and unreliable timepoints), compares against a
  previous registration if one exists, and optionally applies the result. Also moves an
  old estimate-registration / register / stabilize setup to the engine. Use when asked
  to "register", "estimate the registration for", "stabilize", "run registration on" a
  named mantis dataset, or to migrate an old registration config or script.
---

# Register a mantis dataset with the registration engine

Invocation: `/register-with-engine <DATASET_NAME> [--well R/C/FOV] [--apply]`, or any
request to estimate, run or apply registration or stabilization for a mantis dataset.

Registration is estimated per timepoint on the **beads well** (both optical arms see the
beads) and applied to every sample well. Stabilization is estimated per position on the
sample wells themselves.

Which positions are estimated is always the user's choice (the positions passed): one
position gives one transform list shared by every position (registration, whatever the
method -- for ants, pick one FOV with good structure); several give one list each
(stabilization, or a registration that varies across the plate).

**Two ways to run it** (same code, same results):

- `templates/run_registration_nf.sh` -> `nextflow/registration.nf` (default for production):
  Nextflow fans out one task per position x timepoint, retries preempted / timed-out /
  OOM tasks, sizes resources from the plan, and `-resume`s; it can apply in the same run.
- `templates/run_estimate_transform.sh` / `run_apply_transform.sh` -> the plain CLI, which
  submits its own SLURM jobs: quick single-position runs, and **manual** registration
  (interactive: napari plus a terminal, so it cannot run in Nextflow).

## 0. Load context

- `references/methods.md`: which method for which job, `seed_from` (propagation),
  thresholds and their caveats.
- `references/transforms-file.md`: forward vs inverse direction, statuses, shared vs
  per-position lists.
- `references/datasets.md`: where the stores live and the known beads wells.
- `references/recovery.md`: read before acting on any failure.

## 1. Confirm the environment

- On Bruno (`hostname` shows `gpu-*` / `cpu-*` / a login node; `squeue` works).
- The biahub checkout runs from `main`, clean apart from untracked scratch;
  `source .venv/bin/activate`; `biahub estimate-transform --help` renders.
- Nextflow: `module load nextflow` (needs Java 17+), then `nextflow -version`.
- Run every launch inside tmux: SSH drops have killed sessions before.

## 2. Find the inputs

- moving: the deskewed light-sheet store, beads channel (`GFP EX488 EM525-45` or
  `mCherry EX561 EM600-37`, whichever shows the beads).
- reference: the reconstructed phase store, channel `Phase3D`.

Locations differ per project family (`references/datasets.md`). Check both stores exist for
the beads well and read one chunk. If the project's `1-preprocess` intermediates were
cleaned, regenerate just the beads FOV first (deskew and reconstruct that position with the
project's own `deskew_settings.yml` / `phase_config.yaml`, into a separate folder; give each
reconstruct its own folder, since they share the transfer-function file name).

## 3. Identify the beads well

Use `--well` if given; otherwise `references/datasets.md`. Verify rather than assume: a
previous `1-register/estimate-registration*.sh` names it, or run bead detection on one
timepoint of the candidate with the template's `source_peaks_settings` and confirm tens of
peaks (a sample well gives none or a few).

## 4. Build the config

Start from a template in `templates/` (all load with the current schema; a test keeps them
so):

- `estimate-transform-beads.yml`: beads registration (production settings, spectral arm on).
- `estimate-transform-focus-finding.yml`, `estimate-transform-pcc.yml`: stabilization.

If the project has an old `estimate-registration` / `estimate-stabilization` config,
convert it instead: `biahub convert-settings -c <old>.yml -o <new>.yml` (it keeps the seed,
the detection settings and, for beads, propagation). Then set:

- `transform.seed`: the previous run's `approx_transform` (inverse direction, as on disk)
  when the instrument geometry is the same.
- `transform.seed_from`: `previous_timepoint` for drifting bead series (what most
  production configs used), `input` to estimate every timepoint independently.
- `time_indices`: `all` for production, a short contiguous range for a smoke run.

## 5. Present the plan

Show: dataset, beads well, moving/reference stores and channels, T and volume shape, the
config path and its non-default fields, the positions to estimate on and to apply to, how
it runs (Nextflow or the plain CLI), the output directory, expected cost (independent: one
task per position x timepoint, ~2 min each, then one per flagged timepoint to repair;
propagation: one sequential task per position, ~5 min per timepoint by default), and what
will be compared against. Wait for a go.

## 6. Scaffold the output directory

Under the project (e.g. `<project>/1-preprocess/light-sheet/raw/1-register-engine/`, next to
any legacy `1-register/`, never overwriting it): copy the config and the run script --
`templates/run_registration_nf.sh` (Nextflow), or `run_estimate_transform.sh` (plain CLI)
-- and fill in its variables. The script is the run's provenance record; keep it there.

## 7. Launch in tmux

```bash
tmux new-session -d -s register-<DATASET> -c <output dir> \
  "./run_registration_nf.sh 2>&1 | tee registration_nf.log"
#  or, plain CLI: "./run_estimate_transform.sh 2>&1 | tee estimate_transform.log"
```

Relaunching the same script resumes: Nextflow's `-resume` skips finished tasks (the script
passes it), the plain CLI's `--resume` keeps finished timepoints; both refuse if the
settings or inputs changed. The run's records go in a folder named after the output
(`<out>/transforms.yml` -> `<run> = <out>/transforms/`), so several estimates can share
`<out>`; never run two launches on the same output at once. Nextflow retries preempted and
timed-out tasks itself (doubling the time after a time limit); for the plain CLI on large
volumes set a time limit in an sbatch file (`-sb`), which wins over every default
(`references/recovery.md`).

## 8. Monitor

- `ls <run>/timepoints | wc -l` against T (per position: `<run>/positions/<r>/<c>/<f>/`).
- Nextflow: the progress table in the tmux pane; `<out>/.nextflow.log`; per-task logs in
  `<out>/nextflow/slurm_output/{estimate_transform,apply_transform}/`; `nextflow log` for
  past runs; `<out>/nextflow/report.html` and `trace.txt` when it ends.
- Plain CLI: `squeue -u $USER -h -o "%j %t" | grep estimate_transform | sort | uniq -c`; the
  repair phase starts when it prints `repair: N of T timepoints flagged`.
- Done when `<run>/estimate_transform_report.json` exists (and, applying, the plate).
- A record with `"error"` set is a timepoint with no transform of its own (too few beads, an
  empty frame, or a cancelled job); its entry in `transforms.yml` is `unreliable` and its
  `note` says which.

## 9. Read the result

From `transforms.yml`, `<run>/estimate_transform_report.json`, `<run>/run_journal.json` and
the per-timepoint records `<run>/timepoints/<t>.json`:

- score distribution (median, min), `flagged`, `repairs` (accepted, source, before ->
  after), `stand_ins`, and the `unreliable` entries with their notes.
- per-timepoint `metrics`, in the records (`median_residual` in voxels should sit well under 1 on good
  timepoints; `n_matched` close to the detected bead count).
- If a previous registration exists: per-timepoint translation difference (read both in the
  inverse direction: `references/transforms-file.md`). Equal scores with slightly different
  matrices are expected at the score's resolution (~1/N beads); a systematic difference
  above ~2 voxels on accepted timepoints needs a look.

## 10. Optional: apply

`--apply`: with Nextflow, set `APPLY=true` (and `APPLY_POSITIONS`) in
`run_registration_nf.sh`, or apply an existing file with `TRANSFORMS=<file>`; with the plain
CLI, copy `templates/run_apply_transform.sh` next to the transforms file, fill it in and
run it in tmux. The output keeps the full reference grid by default (one wrong transform
cannot shrink every timepoint's field of view); `--crop-to-overlap` crops to the box every
transform covers. With `-r` it registers every moving position onto the reference grid
(reference channels copied, every moving channel transformed -- but a channel the
reference store also has is copied unless it is in the file's `moving_channels`, so with
both arms in one store only the estimated channel moves; `--channels` to pick);
without `-r` it stabilizes each store onto its own grid. Timepoints written with a
non-accepted transform are printed and recorded in the output metadata.

To replace a few bad timepoints with another method's result (beads failed, manual or ants
worked): estimate those timepoints alone (manual: `time_indices: <t>`, plain CLI, with a
display), then
`biahub substitute-transforms -c <out>/transforms.yml -s <other>/transforms.yml -o <final>.yml`,
and apply `<final>.yml` (Nextflow: `TRANSFORMS=<final>.yml`).

## 10b. Migrate an old setup

When asked to move an old `estimate-registration` / `estimate-stabilization` / `register`
/ `stabilize` setup (config or script) to the engine:

1. Read the old script for its inputs, positions and options, and the old config.
2. `biahub convert-settings -c <old>.yml -o <new>.yml` (per-position stabilize files fold
   into one: `-c "<dir>/xyz_stabilization_settings/*.yml"`); read the notes it prints
   (dropped fields, `--crop-to-overlap` for a register that cropped).
3. Write the run script (step 6) with the same inputs and positions: the old
   `estimate-registration` and beads `estimate-stabilization` estimated on the first
   position only; the other stabilizations on every position but `skip_beads_fov`.
4. Present it (step 5) before running. The old command names still run as deprecated
   aliases and print the new commands, but they are not the way forward.

## 11. Handle errors

Follow `references/recovery.md`. Do not delete records or outputs to "clean up" without
asking: `--resume` depends on them.

## 12. Wrap up

Report: beads well, T, wall time, score median/min, flagged and rescued counts (with the
largest before -> after), unreliable timepoints and why, the comparison against any previous
registration, and the output paths. Leave the tmux session for inspection; kill it only after
the summary.
