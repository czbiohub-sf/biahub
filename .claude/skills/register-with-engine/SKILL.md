---
name: register-with-engine
description: >-
  Estimate and apply the fluorescence -> label-free registration (or the stabilization)
  of a mantis or DragonFly dataset with biahub's registration engine (`biahub
  estimate-transform` then `biahub apply-transform`) on the Bruno HPC cluster, through the
  Nextflow workflow `nextflow/registration_wf.nf` (or the plain CLI for quick or
  interactive runs). Finds the moving and reference stores, picks the method (beads on a
  beads well, ants on the cells when there are no beads), builds the config from a
  template, scaffolds a run script, launches it in tmux, monitors it, reads the report,
  checks the result, compares against a previous registration if one exists, and
  optionally applies it. Also moves an old estimate-registration / register / stabilize
  setup to the engine. Use when asked to "register", "estimate the registration for",
  "stabilize", "run registration on" a named dataset, or to migrate an old registration
  config or script.
---

# Register a dataset with the registration engine

Invocation: `/register-with-engine <DATASET_NAME> [--well R/C/FOV] [--apply]`, or any
request to estimate, run or apply registration or stabilization for a dataset.

Registration is estimated per timepoint on the **beads well** (both optical arms or
cameras see the beads) and applied to every sample well; without beads, it is estimated on
cells with `ants` (one FOV with good structure). Stabilization is estimated per position
on the sample wells themselves.

Which positions are estimated is always the user's choice (the positions passed): one
position gives one transform list shared by every position (registration, whatever the
method -- for ants, pick one FOV with good structure); several give one list each
(stabilization, or a registration that varies across the plate).

**Two ways to run it** (same code, same results):

- `templates/run_registration_nf.sh` -> `nextflow/registration_wf.nf` (default for production):
  Nextflow fans out one task per position x timepoint, retries preempted / timed-out /
  OOM tasks, sizes resources from the plan, and `-resume`s; it can apply in the same run.
- `templates/run_estimate_transform.sh` / `run_apply_transform.sh` -> the plain CLI, which
  submits its own SLURM jobs: quick single-position runs, and **manual** registration
  (interactive: napari plus a terminal, so it cannot run in Nextflow).

## 0. Load context

- `references/methods.md`: which method for which job, the `preprocess` blocks, the
  `score` (metric, blocks, report), ANTs and its seed, `seed_from` (propagation),
  thresholds and their caveats.
- `references/transforms-file.md`: forward vs inverse direction, statuses, shared vs
  per-position lists.
- `references/datasets.md`: where the stores live and the known beads wells.
- `references/recovery.md`: read before acting on any failure.

## 1. Confirm the environment

- On Bruno (`hostname` shows `gpu-*` / `cpu-*` / a login node; `squeue` works).
- The biahub checkout runs from `main`, clean apart from untracked scratch;
  `source .venv/bin/activate` (a checkout without its own venv: another biahub venv plus
  `export PYTHONPATH=<checkout>`, as the templates' `VENV` does); check with
  `python -c "import biahub; print(biahub.__file__)"` run outside the repo, and
  `biahub estimate-transform --help` renders.
- Nextflow: `module load nextflow` (needs Java 17+), then `nextflow -version`.
- Run every launch inside tmux: SSH drops have killed sessions before.

## 2. Find the inputs

- moving: the fluorescence store -- mantis: the deskewed light-sheet, beads channel (`GFP
  EX488 EM525-45` or `mCherry EX561 EM600-37`, whichever shows the beads); DragonFly: a
  `*-Prime_*` channel of the converted store.
- reference: the reconstructed phase store (`Phase3D`), or, registering on cells, a virtual
  staining store (`nuclei_prediction` against a nuclear label).

Locations differ per project family (`references/datasets.md`). Check both stores exist for
the beads well and read one chunk. If the project's `1-preprocess` intermediates were
cleaned, regenerate just the beads FOV first (deskew and reconstruct that position with the
project's own `deskew_settings.yml` / `phase_config.yaml`, into a separate folder; give each
reconstruct its own folder, since they share the transfer-function file name).

## 3. Identify the beads well (or decide there are none)

Use `--well` if given; otherwise `references/datasets.md`. Verify rather than assume: a
previous `1-register/estimate-registration*.sh` names it, or run bead detection on one
timepoint of the candidate with the template's `source_peaks_settings` and confirm tens of
peaks (a sample well gives none or a few). Without beads, pick one FOV with clear structure
in both channels for `ants`, and plan the seed (`references/methods.md`, "ANTs").

## 4. Build the config

Start from a template in `templates/` (all load with the current schema; a test keeps them
so):

- `estimate-transform-bead_matching.yml`: beads registration (production settings, spectral arm on).
- `estimate-transform-ants.yml`: registration without usable beads (intensity-based, e.g.
  fluorescence onto virtual staining); its header lists the fields to decide.
- `estimate-transform-manual.yml`: a hand-picked registration in napari (plain CLI only).
- `estimate-transform-focus-finding.yml`, `estimate-transform-pcc.yml`: stabilization.

If the project has an old `estimate-registration` / `estimate-stabilization` config,
convert it instead: `biahub convert-settings -c <old>.yml -o <new>.yml` (it keeps the seed,
the detection settings and, for beads, propagation; old ANTs fields become blocks). Then
set:

- `transform.seed`: the previous run's `approx_transform` (inverse direction, as on disk)
  when the instrument geometry is the same.
- `transform.seed_from`: `previous_timepoint` for drifting bead series (what most
  production configs used), `input` to estimate every timepoint independently.
- `time_indices`: `all` for production, a short contiguous range for a smoke run.
- `preprocess`: what to do to the volumes before the method, chosen from what you see
  (`references/methods.md`: a symptom -> block table); empty for beads.
- `score`: the metric, its own blocks, and `report` metrics to read next to it (worth
  setting for any correlation-scored method).
- `fallback.repair: null` when there is one timepoint.

## 5. Present the plan

Show: dataset, beads well, moving/reference stores and channels, T and volume shape, the
config path and its non-default fields, the positions to estimate on and to apply to, how
it runs (Nextflow or the plain CLI), the output directory, expected cost (independent: one
task per position x timepoint, ~2 min each, then one per flagged timepoint to repair;
propagation: one sequential task per position, budgeted ~10 min per timepoint; ants: ~10-30
min and ~32 GB per estimate on a 2048 x 2048 x 66 pair), and what will be compared against
and how the result will be checked. Wait for a go.

## 6. Scaffold the output directory

Under the project (mantis: e.g. `<project>/1-preprocess/light-sheet/raw/1-register-engine/`,
next to any legacy `1-register/`; DragonFly: e.g. `<project>/2-register-engine/`, next to an
existing `2-register/` or `3-register/`; never overwriting either): copy the config and the run script --
`templates/run_registration_nf.sh` (Nextflow), or `run_estimate_transform.sh` (plain CLI)
-- and fill in its variables. The script is the run's provenance record; keep it there.

## 7. Launch in tmux

```bash
tmux new-session -d -s register-<DATASET> -c <output dir> \
  "./run_registration_nf.sh 2>&1 | tee registration_nf.log"
#  or, plain CLI: "./run_estimate_transform.sh 2>&1 | tee estimate_transform.log"
```

Relaunching the same script resumes: Nextflow's `-resume` skips finished tasks (the script
passes it) and, if the config or transforms file changed, reruns what reads it (from init:
a changed config starts the estimate over); the plain CLI's `--resume` keeps finished
timepoints and refuses if the settings or inputs changed. Either way a timepoint whose
job failed is redone. The run's records go in a folder named after the output
(`<out>/transforms.yml` -> `<run> = <out>/transforms/`), so several estimates can share
`<out>`; never run two launches on the same output at once. Nextflow retries preempted and
timed-out tasks itself (doubling the time after a time limit); for the plain CLI on large
volumes set a time limit in an sbatch file (`-sb`), which wins over every default
(`references/recovery.md`). The plain CLI submits to the `preempted` partition unless the
sbatch file sets `#SBATCH --partition=...`; set only what you need there -- a
`--mem-per-cpu` applies per CPU, and apply-transform asks for up to 8 CPUs per task.

## 8. Monitor

- `ls <run>/timepoints | wc -l` against T (per position: `<run>/positions/<r>/<c>/<f>/`).
- Nextflow: the progress table in the tmux pane; `<out>/.nextflow.log`; per-task logs in
  `<out>/nextflow/slurm_output/{estimate_transform,apply_transform}/`; `nextflow log` for
  past runs; `<out>/nextflow/report.html` and `trace.txt` when it ends.
- Plain CLI: `squeue -u $USER -h -o "%j %t" | grep estimate_transform | sort | uniq -c`; the
  flagging has run when it prints `flag: N of T timepoints flagged` (repair and sweep follow,
  when they are on).
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
- `report` metrics next to each score (the config's `score.report`), compared with the
  unregistered pair.
- Correlation-scored methods (ants): do not trust the score or the `unreliable` status
  alone. Look at overlays -- an XY max projection and an XZ section, on FOVs not used for
  the estimate -- and check the transform is plausible (scale near the expected camera
  ratio, z scale ~1, small shear).
- If a previous registration exists: the difference between the two as the largest
  displacement over the volume (read both in the same direction:
  `references/transforms-file.md`). Equal scores with slightly different matrices are
  expected at the score's resolution (~1/N beads); a systematic difference above ~2 voxels
  on accepted timepoints needs a look.

## 9b. Evaluate

Scores rank candidates; they do not say how far off a registration is. Measure it in µm,
independently of the score, with `biahub.registration.evaluate` (a library, no CLI), on
one timepoint of the position it was estimated on and of one or two held-out positions
(for a beads registration, sample wells: real structure, as the data will be used):

```python
import json
from biahub.core.transform import Transform
from biahub.registration.evaluate import evaluate
from biahub.registration.evaluate_view import plot_evaluation
from biahub.settings import load_transform_settings

transform = Transform(load_transform_settings("<out>/transforms.yml").matrix_for(t, "forward"))
# mov_zyx, ref_zyx: the moving and reference channels at t; *_um: each store's voxel size
record = evaluate(transform, mov_zyx, ref_zyx, mov_um, ref_um, landmarks="nuclei")
plot_evaluation(record, path="<out>/evaluation/<fov>_t<t>.png", title="<dataset> <fov> t=<t>")
json.dump(record.to_dict(), open("<out>/evaluation/<fov>_t<t>.json", "w"), indent=1)
```

Landmarks detected or given are `Landmarks` (`.points_vox`: N x 3 voxel z, y, x, plus
`.voxel_um`); `evaluate(..., landmarks=(moving, reference))` takes a pair you built.

- `landmarks`: `"beads"` on a beads well (`detect_kwargs={"threshold_abs": ...}`, from the
  config's peak settings), `"nuclei"` when the moving channel shows nuclei (DAPI, H2B onto
  virtual-stained nuclei), `None` otherwise (images only).
- Run it as a SLURM job (`cpu` partition, 4 CPUs, ~48 GB for a 2048 x 2048 stack): one warp
  per position, a few minutes.

Read every number against `record.baseline` (the unregistered pair) and across positions;
these are not pass/fail thresholds:

- `tiles` (yx, from max projections) and `points` medians: what is left. Well registered
  cell data sits around a few tenths of a µm, against microns unregistered.
- `points["affine"]`: a yx scale (`scale_zyx[1:]`) away from 1 or a `rotation_yx_deg`
  away from 0 is a scale or rotation the transform did not take out (e.g. `euclidean`
  cannot hold a camera scale: try `similarity` or `affine`). Reported only with enough
  matched points. Ignore its z scale when the landmarks lie in one thin layer (a cell
  monolayer): nothing constrains it there.
- `z_um` and `z_from`: z from points; the image crops can be biased on cross-modal pairs,
  and `z_from` says when there were too few points to trust.
- A held-out position much worse than the estimated one: the transform does not carry
  across the plate (estimate on another position, or one list per position).
- Registered no better than the baseline: it failed; go back to the method and config.

Look at the figure yourself (read the PNG): short arrows in random directions are a good
registration; arrows all one way are a shift; pointing in or out, growing to the edges, a
scale; turning around the centre, a rotation.

Then offer the user a look in napari, on a node with a display (launch it with output to a
file, not piped):

```python
from biahub.registration.evaluate_view import visual_check
visual_check(mov_zyx, ref_zyx, transform, ref_um, phase=phase_zyx)  # phase: optional
```

It shows the reference, the moving volume as it was and registered, and phase underneath
when given. When the window closes, ask whether the registration is good; record the
answer. Never call a registration good without it. On request, a video comparing
transforms (e.g. unregistered, a previous registration, this one):
`evaluate_view.animate_evaluation(mov_zyx, ref_zyx, {name: transform}, "<out>/evaluation/<fov>.mp4")`.

## 10. Optional: apply

`--apply`: with Nextflow, set `APPLY=true` (and `APPLY_POSITIONS`) in
`run_registration_nf.sh`, or apply an existing file with `TRANSFORMS=<file>`; with the plain
CLI, copy `templates/run_apply_transform.sh` next to the transforms file, fill it in and
run it in tmux. The output keeps the full reference grid by default (one wrong transform
cannot shrink every timepoint's field of view); `--crop-to-overlap` crops to the box every
transform covers. With `-r` it registers every moving position onto the reference grid
(reference channels copied, every moving channel transformed -- but a channel the
reference store also has is copied unless it is in the file's `moving_channels`, so with
both arms in one store only the estimated channel moves; `--channels` to pick -- and
always pick when the moving store holds channels of both cameras, e.g. DragonFly's BF);
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
asking: `--resume` depends on them. The plain CLI exits 3 when a job crashed (e.g. out of
memory): the file is written, but those timepoints hold the seed -- do not apply it.

## 12. Wrap up

Report: beads well, T, wall time, score median/min, flagged and rescued counts (with the
largest before -> after), unreliable timepoints and why, the comparison against any previous
registration, the evaluation (per position: tile and point medians against the unregistered
pair, residual scale, z and where it came from) with the user's verdict, and the output
paths. Leave the tmux session for inspection; kill it only after
the summary.
