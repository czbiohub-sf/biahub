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

Read `references/datasets.md` (directory and naming conventions) and
`references/caveats.md` (pipeline-specific pitfalls) before writing any config.
Read `references/recovery.md` when a run fails and `references/monitoring.md`
when the run is live.

## 1. Confirm the environment

### 1a. Bruno

`hostname` — Bruno nodes are `login-*`, `gpu-sm*`, `cpu-*`, `preempted-*`. If
not on Bruno, stop and tell the user to connect; do not ssh on their behalf.
The Nextflow head process is lightweight, so a login node is the right place
for it (see `references/monitoring.md` for the compute-node exception).

### 1b. The biahub checkout — run from `main`, up to date

The pipeline, step CLIs, and config templates version together, so runs should
come off a clean, current `main`:

```bash
cd <BIAHUB> && git fetch origin
git rev-parse --abbrev-ref HEAD               # current branch
git rev-list --count HEAD..origin/main        # commits behind main
git status --porcelain                        # uncommitted changes
git log -1 --format='%h %s (%cr)'
```

| state | what to do |
|---|---|
| on `main`, 0 behind, clean | Proceed. Report it in the plan. |
| on `main`, N behind | Suggest pulling first; show `git log --oneline HEAD..origin/main`, calling out anything under `nextflow/`, `biahub/`, `pyproject.toml`, `uv.lock`. Do not pull unasked. |
| not on `main` | Running a feature branch is how pipeline changes get tested, but it must be a deliberate, stated choice. Get explicit confirmation. |
| uncommitted changes | Flag them and list the files — the run's provenance points at a commit, so uncommitted work is not reproducible. |

To pull: `git pull --ff-only origin main && uv sync --project <BIAHUB>`
(`--ff-only` because a non-fast-forward means local commits — stop and ask).

**Never pull, switch branches, or edit the checkout while a run is live** — it
invalidates `-resume` (`references/caveats.md` §10).

## 2. Find the raw acquisition

Raw data lives in `/hpc/instruments/cm.mantis/<DATASET_NAME>/` — one or more
`*.ome.zarr` stores plus `config.yaml`, `logs/`, `pos_list.pos`.

- **Same stem, several indices** (`<NAME>_1.ome.zarr`, `<NAME>_2.ome.zarr`, …):
  take the highest index — earlier ones are aborted attempts.
  `ls -1d .../*.ome.zarr | sort -V | tail -1`. Do not `du -sh` to decide: it
  walks millions of Lustre files and size does not distinguish an aborted
  acquisition from a complete one.
- **Different stems: stop and ask.** See `references/datasets.md` for real
  examples. Never silently pick between stems.

`/hpc/instruments` is **read-only for this workflow**. Never write into it.

## 3. Determine the layout — HCS plate or flat positions

Read the store's root `zarr.json`:

- `attributes.ome.plate` present → an HCS plate; feed it to the pipeline.
- Positions named `{R}Pos{C}` at the root (`bioformats2raw.layout: 3`) → flat
  acquisition (usual for neuromast/zebrafish/dynatrack). Must be converted to a
  plate at `<OUTPUT>/0-convert/<DATASET>.zarr` first — `references/caveats.md` §1.

## 4. Choose the output project directory

Apply the family rule in `references/datasets.md`, then confirm it in the plan:

| dataset family | output root |
|---|---|
| zebrafish / neuromast / dynatrack | `/hpc/projects/tlg2_mantis/<DATASET>` |
| A549 / cell-line / organelle | `/hpc/projects/intracellular_dashboard/organelle_dynamics/<DATASET>` |

If the target already exists with step outputs, that is a resume or a rerun —
say which you think it is and let the user decide. A fresh reprocess goes to a
`<DATASET>_rerun` sibling.

## 5. Start from the configs on `main`

The templates in `<BIAHUB>/nextflow/configs/{a549,zebrafish}/` are the source
of truth — they version with the pipeline and are reviewed. A previous run's
`configs/` directory is a fallback only: an unreviewed snapshot that drifts
with the schema and carries dataset-specific edits.

Adjust for this dataset — every per-dataset value is listed in
`references/caveats.md` §3 (pixel size, scan step, BF channel name, VS
checkpoint, tracking schema). If step 1b found the checkout behind, check
whether the missing commits touch `nextflow/configs/`.

**If the template itself is wrong** (schema migration, bad default — anything
the next dataset in the family would also need), fix it in a PR against
`nextflow/configs/<family>/`, stating which dataset exposed it. Genuinely
per-dataset values stay in the run directory.

## 6. Present the plan

Do not run anything yet. Show the user:

1. The biahub checkout: path, branch, HEAD commit, behind-count, working-tree
   state. Make anything other than *clean `main`, up to date* a visible caveat
   with a recommendation.
2. Resolved input store, its size, position count, channel names, and
   `(T, C, Z, Y, X)` shape.
3. Whether a `0-convert` plate build is needed.
4. Output project directory and the step layout.
5. That configs come from `<BIAHUB>/nextflow/configs/<family>/` at commit
   `<sha>`, every value changed for this dataset, and any template fix headed
   for a PR.
6. Pipeline steps: flat-field → deskew → reconstruct → virtual-stain →
   assemble → track. **For neuromast/zebrafish, say that `5-assemble` is the
   deliverable and `4-track` is a discarded by-product** (`caveats.md` §4).
7. Known caveats that apply to this dataset.
8. Rough wall-time and that `-resume` is on.

Get explicit approval.

## 7. Scaffold the output directory

```bash
mkdir -p <OUTPUT>/configs <OUTPUT>/nextflow
cp <BIAHUB>/nextflow/configs/<family>/*.yml <OUTPUT>/configs/
```

Edit the copies for this dataset. Copy `templates/run_mantis_v2.sh` to
`<OUTPUT>/run_mantis_v2.sh`, fill in `DATASET`, `DATA_DIR`, `PROJECT_DIR`,
`BIAHUB_PROJECT`, `chmod +x`. The script stays in the output directory as the
run's provenance record.

If a plate build is needed, do it now via the **build-hcs-plate** agent
(`caveats.md` §1) and verify the plate opens with iohub before launching.

## 8. Launch in tmux, in the foreground

Create a detached session, send the command, and tell the user how to attach —
do **not** attach yourself:

```bash
SESSION="nf_<DATASET>"
tmux new-session -d -s "$SESSION" -c "<OUTPUT>"
tmux send-keys -t "$SESSION" "bash ./run_mantis_v2.sh 2>&1 | tee -a nextflow/run_$(date +%Y%m%dT%H%M%S).log" Enter
```

Tell the user: `tmux attach -t nf_<DATASET>` to watch, `Ctrl-b d` to detach.

**Launch from `<OUTPUT>`** (the `-c` above): the cwd controls where
`.nextflow.log` and the `.nextflow/` resume cache land — relaunching from a
different directory finds no cache and recomputes everything. The work dir is
pinned to `${params.output}/nextflow/work` regardless.

**The environment is resolved once, before launch, and inherited** — tasks
call `biahub`/`viscy` bare (ENVIRONMENT CONTRACT in
`nextflow/modules/common.nf`). `run_mantis_v2.sh` does
`uv sync --project <BIAHUB> && source <BIAHUB>/.venv/bin/activate` before
`nextflow run`; `check_environment()` aborts at launch if the tools are
missing. Before syncing, run the out-of-band-package check in `caveats.md` §11.

## 9. Monitor

Follow `references/monitoring.md`: poll `squeue -u $USER`, the tail of
`<OUTPUT>/.nextflow.log`, and `<OUTPUT>/nextflow/slurm_output/<step>/*_<jobid>.out`
at a few minutes' interval — these runs take hours to days. Notify via
`templates/notify.sh` (Slack webhook, falls back to terminal) on completion, a
terminating error, or a step retrying past a reasonable threshold.

## 10. Handle errors

Classify before acting — `references/recovery.md` has the decision table:

- **Exit 130–145** (preemption, timeout, OOM): Nextflow retries up to 5 times.
  Do nothing unless retries are exhausted.
- **Checksum / Lustre EIO / torn shard**: rerun with `-resume` first — the
  pinned iohub replaces torn shards and resumes per unit. Only if it fails
  identically again, write a repair proposal for the user or hand it to the
  **job-io-error-repair** agent. **Never delete zarr data from this skill.**
- **Exit 1/2 with a Python traceback**: real bug or bad config. Fix, relaunch
  with `-resume`.

Restarts are always `bash ./run_mantis_v2.sh` — the script passes `-resume`.

## 11. Wrap up

1. Normalize channel names on the assembled plate with
   `templates/rename_channels.py` (idempotent; `caveats.md` §2 — check first
   whether a `rename-channels` CLI has landed on main, making this obsolete).
2. Verify `5-assemble/<DATASET>.zarr` opens with iohub; report shape, channels,
   size on disk. For neuromast/zebrafish this is the deliverable; report
   `4-track` only as a discarded by-product.
3. Report per-step task counts, failures, retries, and wall time from
   `<OUTPUT>/nextflow/trace.txt`; point at `report.html` and `timeline.html`.
4. Send a final Slack message.
5. Flag anything needing a human eye: positions that passed only after many
   retries, steps far slower than the reference run, unexpected channel counts.
