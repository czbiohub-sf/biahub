---
name: reconstruct-with-nextflow
description: >-
  Run a mantis-v2 microscope dataset through the biahub Nextflow reconstruction
  pipeline (nextflow/mantis-v2.nf: flat-field → deskew → reconstruct →
  virtual-stain → assemble → track → QC) on the Bruno HPC cluster. Assemble,
  track and QC are optional and selected per dataset family: A549 runs assemble
  + track + QC, neuromast runs assemble + QC and no tracking. Locates the raw
  acquisition under /hpc/instruments/cm.mantis, picks the output project
  directory, scaffolds configs, launches the run in a tmux session in the
  foreground so it can be watched, recovers from Lustre/torn-shard I/O errors,
  and reports a summary including the QC verdict. Use when asked to "reconstruct", "run the pipeline
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

### 1a-bis. Slack notifications — optional, offer to set up

```bash
printenv BIAHUB_SLACK_WEBHOOK >/dev/null && echo "webhook set" || echo "webhook MISSING"
printenv BIAHUB_SLACK_ID      >/dev/null && echo "slack id set" || echo "slack id MISSING"
```

**Both are optional and neither gates the run** — without them every message is
printed instead of posted and nothing else changes. Never block a launch on this.

If either is missing, say so once, explain that they only enable Slack
notifications, and **offer to append them to `~/.bashrc`**. The webhook is a
credential the user cannot self-serve: tell them to **ask a biahub developer
(Ivan, Taylla)** for it. For the ID, point them at Slack profile → **⋮ / More** →
**Copy member ID**, and give the format explicitly — `U0A2ZH9CS8S`, not
`@Ivan Ivanov`, which never pings. Write to `~/.bashrc` only if they agree, append
rather than rewrite, and tell them to `source ~/.bashrc` before launching, since
the value is read from the launching shell. Full instructions:
`references/monitoring.md` § Setting it up.

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
of truth — they version with the pipeline and are reviewed. QC configs are
shared across families and live in `<BIAHUB>/nextflow/configs/qc/`, one
directory per store kind (`assemble/`, `track/`) — copy the tree, do not
flatten it: each directory must hold only its own step's config, or every
report tab renders the same one. A previous run's
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
6. Pipeline steps. Reconstruction proper — flat-field → deskew → reconstruct →
   virtual-stain — always runs; assemble, track and QC run only if their config
   is passed (biahub#306), so name the set this run performs:

   | family | steps after virtual-stain |
   |---|---|
   | A549 / cell-line / organelle | assemble, track, QC (image + tracking) |
   | neuromast / zebrafish / dynatrack | assemble, QC (image) — **no tracking** |

   For neuromast/zebrafish say that the assembled store is the deliverable and
   that tracking is simply not run — there is no tracking by-product any more
   (`caveats.md` §4).

   **State the directory numbers this run will produce.** The number is the
   step's position among the steps performed, not a fixed label, so a neuromast
   run writes `4-assemble` as its last directory and an A549 run writes
   `4-assemble` then `5-track`. Older A549 runs on disk say `5-assemble` /
   `4-track`, from when the numbers were fixed — say so if the user is comparing
   against one.
7. Known caveats that apply to this dataset.
8. Rough wall-time and that `-resume` is on.
9. Whether Slack notifications are on, and who will be @-mentioned at run end.
   If `$BIAHUB_SLACK_WEBHOOK` or `$BIAHUB_SLACK_ID` is missing, note it here as a
   caveat with the offer from §1a-bis — not as a blocker.

Get explicit approval.

## 7. Scaffold the output directory

```bash
mkdir -p <OUTPUT>/configs <OUTPUT>/nextflow
cp <BIAHUB>/nextflow/configs/<family>/*.yml <OUTPUT>/configs/
cp -r <BIAHUB>/nextflow/configs/qc <OUTPUT>/configs/qc     # keeps assemble/ and track/
```

Then set `STEPS` in the run script to the family's set from §6 — that is the
whole mechanism for skipping a step. For a neuromast run, drop `track` and
`qc_track`; `track.yml` need not exist at all.

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
tmux send-keys -t "$SESSION" "bash ./run_mantis_v2.sh" Enter
```

Tell the user: `tmux attach -t nf_<DATASET>` to watch, `Ctrl-b d` to detach.

**Do not pipe the launch through `tee`** (or anything else) — a pipe costs the
live progress table for nothing, since the two files below already carry the
console's content.

**The live table also needs `CLAUDECODE` unset.** Nextflow 26.04 has an "agent
mode" that replaces the table with one static `[PROCESS …]` line per task, and it
turns itself on whenever `CLAUDECODE`, `AGENT`, or `NXF_AGENT_MODE` is truthy. A
tmux session created from a Claude Code tool call inherits `CLAUDECODE=1`, so the
run a human then watches for days renders as agent output. `run_mantis_v2.sh`
does `unset CLAUDECODE` before `nextflow run`; nothing else works, because agent
mode ORs the three variables (`NXF_AGENT_MODE=false` cannot disable it) and
`-ansi-log true` is accepted and silently dropped. Reported upstream as
[nextflow#7478](https://github.com/nextflow-io/nextflow/issues/7478); drop the
`unset` once it is fixed. Confirm after launching:

```bash
tmux capture-pane -p -t "$SESSION" | grep -q '^\[PROCESS ' && echo "agent mode — table lost"
```

Two files carry what the console used to:

- `<OUTPUT>/.nextflow.log` — the run record you read to follow progress (step 9).
- `<OUTPUT>/nextflow/provenance.txt` — written by `run_mantis_v2.sh` at each
  launch: branch, full commit, input store, host, Nextflow version, and whether
  the checkout was dirty or off `main`. Appended, so a `-resume` relaunch adds an
  entry instead of erasing the first one. This is what answers "which commit
  produced this output?" after the tmux pane is gone — `.nextflow.log` records the
  launch command line but not the git state.

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
at a few minutes' interval — these runs take hours to days.

**The pipeline notifies Slack itself** — run start, each step's completion, and
run end (`nextflow/modules/notify.nf`), so do not re-send those. What is left for
you is a terminating error you have diagnosed, a position on attempt 4 of 5, and
the wrap-up; send those with `templates/notify.sh`. Notifications need
`$BIAHUB_SLACK_WEBHOOK` and `$BIAHUB_SLACK_ID`; without them messages print
instead of posting and nothing fails. For a `--max_positions 1` smoke test,
launch with `env -u BIAHUB_SLACK_WEBHOOK` to keep the channel quiet.

## 10. Handle errors

Classify before acting — `references/recovery.md` has the decision table:

- **Exit 130–145** (preemption, timeout, OOM): Nextflow retries up to 5 times.
  Do nothing unless retries are exhausted.
- **Checksum / Lustre EIO / torn shard**: rerun with `-resume` first — iohub
  replaces torn shards and resumes per unit. Only if it fails
  identically again, write a repair proposal for the user or hand it to the
  **job-io-error-repair** agent. **Never delete zarr data from this skill.**
- **Exit 1/2 with a Python traceback**: real bug or bad config. Fix, relaunch
  with `-resume`.

Restarts are always `bash ./run_mantis_v2.sh` — the script passes `-resume`.

## 11. Wrap up

1. Normalize channel names on the assembled plate with
   `templates/rename_channels.py` (idempotent; `caveats.md` §2 — check first
   whether a `rename-channels` CLI has landed on main, making this obsolete).
2. Verify the assembled store (`<N>-assemble/<DATASET>.zarr`, `4-assemble`
   unless earlier steps were skipped) opens with iohub; report shape, channels,
   size on disk. This is the deliverable for every family; tracking, when it
   ran, is reported beside it rather than as a by-product.
3. If QC ran, report its verdict: the `QC_SUMMARY` line per store from the
   pipeline log (`pass=`/`fail=`/`gates_fail=`), and point at the report at
   `<OUTPUT>/qc/report/index.html` — one page, one tab per QC'd store. A gate
   failure does NOT fail the run: `imaging-qc gate` exits 0 either way, so a
   failing verdict is only visible in the summary line, the report, and the
   `tables/qc/` parquet inside each store.
4. Report per-step task counts, failures, retries, and wall time from
   `<OUTPUT>/nextflow/trace.txt`; point at `report.html` and `timeline.html`.
5. Confirm the pipeline's automatic run-end message landed, then send a wrap-up
   only for what the pipeline cannot know: the channel-rename result, the iohub
   verification, and size on disk.
6. Flag anything needing a human eye: positions that passed only after many
   retries, steps far slower than the reference run, unexpected channel counts.
