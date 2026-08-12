---
name: reconstruct-with-nextflow
description: >-
  Run a mantis-v2 microscope dataset through the biahub Nextflow reconstruction
  pipeline (nextflow/mantis-v2.nf: flat-field → deskew → reconstruct →
  virtual-stain → assemble → track) on the Bruno HPC cluster. Locates the raw
  acquisition under /hpc/instruments/cm.mantis, picks the output project
  directory, scaffolds configs, launches the run in a tmux session in the
  foreground so it can be watched, recovers from Lustre/torn-shard I/O errors,
  and reports a summary. Use when asked to "reconstruct", "run the pipeline
  on", or "process" a named mantis-v2 dataset.
---

# Reconstruct a mantis-v2 dataset with Nextflow

Invocation: `/reconstruct-with-nextflow <DATASET_NAME>`, or any request to
run/reconstruct/process a mantis-v2 dataset by name.

**Never launch before the user approves a plan.** Steps 1–5 are read-only
investigation; step 6 presents the plan; nothing is written to disk until the
user says go.

## 0. Load context

Read `references/datasets.md` first — it holds the directory conventions, the
dataset-family rules, and the naming convention. Read `references/caveats.md`
before writing any config. The other references are read on demand:
`references/recovery.md` when a run fails, `references/monitoring.md` when the
run is live.

## 1. Confirm you are on Bruno

`hostname` — Bruno nodes are `login-*`, `gpu-sm*`, `cpu-*`, `preempted-*`. If the
shell is not on Bruno, stop and tell the user to connect; do not attempt to ssh
somewhere on their behalf.

Nextflow's head process is lightweight (it only submits SLURM jobs), so a **login
node is the right place for it**. Reserve a compute node only if the user asks,
or if a login-node policy kills long-lived processes — see
`references/monitoring.md`.

## 2. Find the raw acquisition

Raw mantis-v2 data lives in `/hpc/instruments/cm.mantis/<DATASET_NAME>/`. That
directory holds one or more `*.ome.zarr` stores plus `config.yaml`, `logs/`,
`pos_list.pos`.

```bash
ls -1 /hpc/instruments/cm.mantis/ | grep -i <DATASET_NAME>
ls -1 /hpc/instruments/cm.mantis/<DATASET_NAME>/
```

Micro-Manager appends an acquisition index, so a re-started acquisition leaves
`<NAME>_1.ome.zarr`, `<NAME>_2.ome.zarr`, … side by side.

**Same stem, several indices → take the highest index.** The earlier ones are
aborted or abandoned attempts; the last one is the acquisition that ran.

```bash
ls -1d /hpc/instruments/cm.mantis/<DATASET>/*.ome.zarr | sort -V | tail -1
```

Do **not** `du -sh` the store to decide. It walks millions of chunk files on
Lustre and takes minutes, and its answer does not mean what it looks like: a
large aborted acquisition outweighs a small complete one, so size does not
distinguish them. The index does.

**Different stems, on the other hand, are ambiguous — stop and ask.** The
convention is `YYYY_MM_DD_<description>` and the store name usually matches the
directory name; when it does not, say so explicitly and ask which store to use.
Directory `2026_08_04_smart_fov_selection_test` contains
`2026_08_04_test_fov_selection_*.ome.zarr`; directory `2026_08_05_dynatrack_2dpf`
contains `2026_08_05_dynatrack_2df_*.ome.zarr` (missing `p`). Where `_epi`,
`_prescan` or `_fov_debug` variants sit alongside the real store, they are
different acquisitions, not indices of one. Never silently pick between stems.

`/hpc/instruments` is **read-only for this workflow**. Never write into it.

## 3. Determine the layout — HCS plate or flat positions

Read the store's root `zarr.json`:

- `attributes.ome.plate` present → a real HCS plate. Feed it to the pipeline
  directly.
- Positions named `{R}Pos{C}` at the root and `bioformats2raw.layout: 3` → a flat
  acquisition with **no HCS plate**. This is the usual case for
  neuromast/zebrafish/dynatrack. It must be converted into a plate at
  `<OUTPUT>/0-convert/<DATASET>.zarr` before the pipeline can fan out over
  positions. See `references/caveats.md` §1.

## 4. Choose the output project directory

Apply the family rule in `references/datasets.md`, then **confirm it in the
plan** — it is a proposal, not a decision:

| dataset family | output root |
|---|---|
| zebrafish / neuromast / dynatrack | `/hpc/projects/tlg2_mantis/<DATASET>` |
| A549 / cell-line / organelle | `/hpc/projects/intracellular_dashboard/organelle_dynamics/<DATASET>` |

If the target directory already exists with step outputs in it, that is a
**resume or a rerun** — say which you think it is and let the user decide before
touching anything. The established convention for a fresh reprocess of the same
data is a `<DATASET>_rerun` sibling.

## 5. Start from the configs on `main`

**The configs in the biahub checkout are the source of truth. Start there.**

```bash
<BIAHUB>/nextflow/configs/{a549,zebrafish}/
```

They live next to `mantis-v2.nf`, so a schema change to the pipeline and the
matching change to its configs land in one commit, and they are reviewed. A
previous run's `configs/` directory is a **fallback**, for when no template
exists for the family or you need to see what a specific run actually used:

```bash
ls -dt /hpc/projects/tlg2_mantis/*/configs                                  # zebrafish
ls -dt /hpc/projects/intracellular_dashboard/organelle_dynamics/*/configs   # A549
```

Treat a previous run's configs as evidence, not as the baseline — they are
unreviewed snapshots that drift with the schema and often carry dataset-specific
edits nobody intended to generalize.

Then adjust for *this* dataset. Every value that must be checked per dataset is
called out in `references/caveats.md` §3 — pixel size, scan step, BF channel
name, VS checkpoint, and the tracking schema. Confirm the checkout is current
(`git log -1 --oneline origin/main -- nextflow/configs`) before copying.

### When the template itself is wrong, fix the template

If a dataset needs a change that is **not** dataset-specific — a schema
migration, a corrected default, a value the template simply has wrong — do not
leave the fix in the run directory where the next dataset will miss it. Open a PR
against `nextflow/configs/<family>/` that:

- makes the change,
- states in the description **which dataset exposed it and what failed**, and
- says whether it applies to the other family too.

Genuinely per-dataset values (pixel size, scan step, checkpoint choice for a
one-off model) stay in the run directory and do **not** go back to `main`. The
test is whether the next dataset in this family would want the same value.

## 6. Present the plan

Do not run anything yet. Show the user, concretely:

1. Resolved input store, its size, position count, channel names, and
   `(T, C, Z, Y, X)` shape.
2. Whether a `0-convert` plate build is needed.
3. Output project directory, and the full step layout that will be created.
4. That the configs come from `<BIAHUB>/nextflow/configs/<family>/` at commit
   `<sha>`, every value you changed for this dataset, and any change you intend
   to send back to `main` as a PR.
5. Pipeline steps that will run: flat-field → deskew → reconstruct →
   virtual-stain → assemble → track (the full `mantis-v2.nf` workflow).
   **For neuromast/zebrafish, say that the deliverable is `5-assemble` and that
   `4-track` is a discarded by-product** — tracking is an A549 step, but
   `mantis-v2.nf` cannot skip it today. See `references/caveats.md` §4.
6. Known caveats that apply to this dataset (from `references/caveats.md`).
7. Rough wall-time and whether `-resume` is on.

Get explicit approval.

## 7. Scaffold the output directory

```bash
mkdir -p <OUTPUT>/configs <OUTPUT>/nextflow
cp <BIAHUB>/nextflow/configs/<family>/*.yml <OUTPUT>/configs/
```

Then edit the copies for this dataset. Copy `templates/run_mantis_v2.sh` to
`<OUTPUT>/run_mantis_v2.sh`, fill in `DATASET`, `DATA_DIR`, `PROJECT_DIR`,
`BIAHUB_PROJECT`, and `chmod +x` it. Keep the script in the output directory —
it is the run's provenance record.

The pipeline itself creates `0-flatfield/ 1-deskew/ 2-reconstruct/
3-virtual-stain/ 4-track/ 5-assemble/` and `nextflow/{work,slurm_output,
report.html,timeline.html,trace.txt,dag.html}` under `--output`.

If a plate build is needed, do it now via the **build-hcs-plate** agent (see
`references/caveats.md` §1), and verify the plate opens with iohub before
launching.

## 8. Launch in tmux, in the foreground

The user watches the run, so it must be attachable and in the foreground of its
pane. Create a detached session, send the command, and tell the user how to
attach — do **not** attach yourself.

```bash
SESSION="nf_<DATASET>"
tmux new-session -d -s "$SESSION" -c "<OUTPUT>"
tmux send-keys -t "$SESSION" "bash ./run_mantis_v2.sh 2>&1 | tee -a nextflow/run_$(date +%Y%m%dT%H%M%S).log" Enter
```

Tell the user: `tmux attach -t nf_<DATASET>` to watch, `Ctrl-b d` to detach.

**Run from `<OUTPUT>`** — that is what `-c "<OUTPUT>"` above is for. The *work*
dir does not depend on it: `nextflow.config` pins
`workDir = "${params.output}/nextflow/work"` regardless of where you launch
(override with `-work-dir` to put it on faster scratch). What the cwd does
control is where `.nextflow.log` and the `.nextflow/` **resume cache** land.
Launching from somewhere else scatters those away from the run, and relaunching
from a different directory means `-resume` finds no cache and recomputes
everything.

**Do not edit the biahub checkout while a run is live** — it changes Nextflow
task hashes and invalidates `-resume`. (Editing this skill is fine; it is not
part of the pipeline.)

**The environment is resolved once, before launch, and inherited.** The pipeline
calls `biahub` and `viscy` as bare commands — there is no per-task `uv run`
wrapper and no `--biahub_project` parameter. `run_mantis_v2.sh` already does
this; it is here so you can recognize it:

```bash
uv sync --project <BIAHUB>
source <BIAHUB>/.venv/bin/activate
nextflow run <BIAHUB>/nextflow/mantis-v2.nf ...
```

sbatch exports the submit environment (`--export=ALL` is the default) and the
`.venv` is on shared storage, so every compute node resolves the same paths.
`mantis-v2.nf` calls `check_environment()` and aborts at launch with an
actionable message if `biahub`/`viscy` are missing, before any task is submitted.

`uv sync` here is expected and safe — see `references/caveats.md` §11 for the one
environment-specific hazard to check for first.

## 9. Monitor

Follow `references/monitoring.md`. In short: poll `squeue -u $USER`, the tail of
`<OUTPUT>/.nextflow.log`, and the per-step SLURM logs at
`<OUTPUT>/nextflow/slurm_output/<step>/*_<jobid>.out`. Poll at a few minutes'
interval, not seconds — these runs take hours to days.

Notify the user via Slack on: pipeline completion, a terminating error, and a
step that has retried past a reasonable threshold. Use
`templates/notify.sh`, which posts to `$BIAHUB_SLACK_WEBHOOK` and falls back to
printing in the terminal when the variable is unset.

## 10. Handle errors

When a task fails, classify before acting — `references/recovery.md` has the
decision table. The one-line version:

- **Exit 130–145** (preemption, timeout, OOM): Nextflow's `errorStrategy` already
  retries up to 5 times. Do nothing unless retries are exhausted.
- **Lustre EIO / `RuntimeError: the checksum is invalid` / torn shard**: a killed
  task left a half-written shard. **Rerun with `-resume` first** — iohub#455
  replaces a torn shard rather than reading it back and resumes per unit, which
  fixes this without any repair. Only if it fails *identically* again, write up a
  repair proposal for the user (paths, `du -sh`, what is lost) or hand it to the
  **job-io-error-repair** agent. **Never delete zarr data from this skill.**
- **Exit 1/2 with a Python traceback**: a real bug or a bad config. Nextflow
  terminates on purpose. Read the traceback, fix, relaunch with `-resume`.

Restarts are always `bash ./run_mantis_v2.sh` — the script already passes
`-resume`.

## 11. Wrap up

When the run finishes:

1. **Normalize channel names on the assembled plate** —
   `templates/rename_channels.py`, run against
   `<OUTPUT>/5-assemble/<DATASET>.zarr`. This applies the biahub#291 convention
   and is idempotent, so it is safe to re-run. See `references/caveats.md` §2 —
   and check first whether a `rename-channels` CLI has landed on main, which
   would make the manual step obsolete.
2. Verify the assembled store opens with iohub and report its shape, channels,
   and size on disk. For neuromast/zebrafish this is the deliverable; report
   `4-track` only as a discarded by-product.
3. Report the Nextflow summary: per-step task counts, failures, retries, and
   total wall time, from `<OUTPUT>/nextflow/trace.txt`. Point the user at
   `<OUTPUT>/nextflow/report.html` and `timeline.html`.
4. Send a final Slack message.
5. Flag anything that needs a human eye: positions that only passed after many
   retries, steps that ran far longer than the reference run, an assembled
   channel count that does not match expectations.
