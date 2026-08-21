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

### 1a-bis. GitHub access for the QC dependency — walk the user through it

QC runs the external `imaging-qc` CLI, which comes from
`czbiohub-sf/imaging-qc-pipeline`, a **private** org repo. `uv sync --extra qc`
clones it over HTTPS, so the user needs a GitHub credential — but **not an SSH
key**. Check before scaffolding, because the failure otherwise arrives as a raw
git error in the middle of an install:

```bash
gh auth status 2>&1 | head -3
```

If it reports a logged-in account, run one more command to make git use it, and
move on:

```bash
gh auth setup-git      # idempotent; installs gh as git's credential helper
```

If it reports no account, walk them through it — two commands, no key to
generate or upload, and `gh` is already at `/usr/bin/gh` on Bruno:

```bash
gh auth login          # choose GitHub.com -> HTTPS -> login with a web browser
gh auth setup-git
```

`gh auth login` prints a one-time code and a URL to open on their laptop; the
Bruno session does not need a browser. If the org enforces SAML SSO the same
browser flow authorizes it. Verify before continuing:

```bash
git ls-remote https://github.com/czbiohub-sf/imaging-qc-pipeline HEAD >/dev/null \
  && echo "QC dependency reachable" || echo "still no access"
```

Still refused after logging in means their account lacks access to that repo,
which no local setup can fix — **tell them to ask a biahub developer (Ivan,
Taylla) to be added**, and offer to continue without QC by dropping `--qc_config`
and `--qc_track_config` (§7). The rest of the pipeline needs no GitHub
credential at all, so this never blocks a reconstruction.

A user who already uses SSH keys for GitHub needs nothing here; if they would
rather keep using them, one local rewrite makes the HTTPS URL resolve over ssh:

```bash
git config --global url."ssh://git@github.com/".insteadOf "https://github.com/"
```

### 1a-ter. Slack notifications — optional, offer to set up

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
6. **The steps this run will perform, listed in order.** Reconstruction proper —
   flat-field → deskew → reconstruct → virtual-stain — always runs; assemble,
   track and QC run only if their config is passed (biahub#306). Defaults by
   family, which is what to present unless the user says otherwise:

   | family | steps |
   |---|---|
   | A549 / cell-line / organelle | flat-field → deskew → reconstruct → virtual-stain → **assemble → track → QC** |
   | neuromast / zebrafish / dynatrack | flat-field → deskew → reconstruct → virtual-stain → **assemble → QC** (no tracking) |

   Write the list out in the plan rather than naming the family, and say which
   optional steps are being skipped and why. If the user asks to skip anything
   else, reflect that here — the plan is where the step set is agreed.

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
   caveat with the offer from §1a-ter — not as a blocker.
10. If QC is in the step list, that the GitHub credential from §1a-bis is in
    place. If it is not and cannot be, say the run will proceed without QC
    rather than silently dropping it — the step set in item 6 has to match what
    is actually going to run.

Get explicit approval.

## 7. Scaffold the output directory

```bash
mkdir -p <OUTPUT>/configs <OUTPUT>/nextflow
cp <BIAHUB>/nextflow/configs/<family>/*.yml <OUTPUT>/configs/
cp -r <BIAHUB>/nextflow/configs/qc <OUTPUT>/configs/qc     # keeps assemble/ and track/
```

The run script passes all four optional configs — `--concatenate_config`,
`--track_config`, `--qc_config`, `--qc_track_config` — so an A549 run needs no
edit. **For a neuromast/zebrafish run, or any step the user asked to skip,
DELETE that flag's line** from the `nextflow run` call and note the skip in a
comment above it, so the script still records what this run did. A neuromast run
deletes `--track_config` and `--qc_track_config`; `track.yml` need not exist at
all, and there is no `qc` config directory to copy for a step that is not run.

Delete rather than comment: a `#` inside a backslash-continued command does not
start a comment line — the continuation swallows it, every flag below is dropped
including `-resume`, and bash then tries to run the remainder as a command.

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
