---
name: reconstruct-with-nextflow
description: >-
  Run a mantis-v2 microscope dataset through the biahub Nextflow reconstruction
  pipeline (nextflow/mantis-v2.nf: flat-field → deskew → reconstruct →
  virtual-stain → assemble → project → segment → track → pyramid → QC) on the Bruno
  HPC cluster. Assemble, project, segment, track and QC are optional and selected per
  dataset family: A549 runs all of them, neuromast runs assemble + QC and no
  segmentation or tracking; pyramids are on whenever assemble runs. Locates the raw
  acquisition under /hpc/instruments/cm.mantis, picks the output project
  directory, scaffolds configs, launches the run in a tmux session in the
  foreground so it can be watched, monitors it unattended every 30 minutes
  (relaunching on transient Lustre/torn-shard I/O errors), and reports a
  summary including the QC verdict. Use when asked to "reconstruct", "run the pipeline
  on", or "process" a named mantis-v2 dataset.
---

# Reconstruct a mantis-v2 dataset with Nextflow

Invocation: `/reconstruct-with-nextflow <DATASET_NAME>`, or any request to
run/reconstruct/process a mantis-v2 dataset by name.
`/reconstruct-with-nextflow monitor <OUTPUT>` runs one monitoring pass over a
launched run and nothing else (§9a); the 30-minute loop of §9 fires it.

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
(§7). The rest of the pipeline needs no GitHub
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
changes code under live tasks or invalidates `-resume` (`references/caveats.md` §10).

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
of truth — they version with the pipeline and are reviewed. The QC config sits with
them, one self-contained `qc.yaml` for the assembled store; A549's
`qc_project.yaml` also counts cells in the tracked label image tracking writes
into the projected plate. It carries `.yaml`
rather than `.yml` because imaging-qc loads a stage config through Hydra, which strips the
extension and looks for `<stem>.yaml`. A previous run's
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

### 5a. Check the channel couplings: QC filters, project.yml, segment.yml, track.yml

Once the configs are edited, run this from the biahub checkout:

```bash
python .claude/skills/reconstruct-with-nextflow/templates/check_qc_channels.py \
    <RAW_STORE.zarr> <OUTPUT>/configs
```

It exits 1 and names every `channels:` entry that the store will not have.

The stores QC runs over do not exist yet — the pipeline's init phase creates
them — so the script predicts their channels from the configs that produce
them: raw channel names for the deskewed data, waveorder's
`output_channel_names` for the phase channel, virtual-stain's `target_channel`
for the predictions, and `project.yml`'s keys for the projected plate.

**`project.yml` against the assembled plate.** When `configs/project.yml`
exists the script also checks it both ways. A channel in `project.yml` that the
assembled plate will not have is a `MISS` (exit 1): `biahub project --init`
refuses it, so fix the name now — the raw fluorescence names are the camera
strings of this acquisition and are the usual mismatch. An assembled channel
that `project.yml` does not list is printed as `NOT PROJECTED`; that is a
warning, not a failure (`biahub project` drops it with a warning), **but the
intention of the pipeline is that every assembled channel is projected unless
the user decided otherwise**, so carry the list into the pre-run plan (step 6
of §7) and ask the user whether to add each one or leave it out on purpose.
The script also checks that `segment.yml`'s model `channels` are channels
`project.yml` writes, since segment reads the projected plate, and that
`track.yml`'s `labels` is a `segment.yml` model name, since track reads the
label image segment wrote (`labels/<model>`).

**`qc_project.yaml` against the projected plate, `segment.yml` and `track.yml`
— the coupling that actually breaks.** Segment writes `labels/<model>` into each
position of the plate it reads and tracking writes `labels/tracks_<model>`
beside it. `qc_project.yaml`'s `instance_count` names the tracked one, `label:
tracks_nuclei`, and its anchor `channels: [nuclei_prediction]`, the channel the
labels were segmented from. The label only resolves if `track.yml` says `labels:
nuclei` and `segment.yml` has a model `nuclei`; the anchor must be that model's
channel and a channel `project.yml` writes — four files with no shared source of
truth, and changing one is the ordinary way to desynchronise them. The script
checks all of it when `qc_project.yaml` exists.

Since imaging-qc#226 the pipeline refuses an unknown channel name itself, at
`plan-stage`, so a mistake fails during the init phase rather than silently
skipping a metric. This check is the earlier one: it catches the mistake while
you are still editing, instead of costing a launch. A `MISS` is authoritative;
a pass is not a proof, because the script assumes `concatenate.yml` takes `all`
from each source (both shipped families do — a config selecting a subset will
over-predict).

### 5b. Check the reconstruct pixel sizes against the deskew geometry

`reconstruct.yml`'s `yx_pixel_size` / `z_pixel_size` must describe the
deskewed store, which `deskew.yml` determines (`references/caveats.md` §3).
waveorder does not read them from the zarr, so a stale or missing value
silently yields a wrong transfer function. Run from the biahub checkout:

```bash
.venv/bin/python - <OUTPUT>/configs <<'EOF'
import math, sys, yaml
d = yaml.safe_load(open(f"{sys.argv[1]}/deskew.yml"))
tf = yaml.safe_load(open(f"{sys.argv[1]}/reconstruct.yml"))["phase"]["transfer_function"]
px = d["pixel_size_um"]
want = {"yx_pixel_size": px,
        "z_pixel_size": d.get("average_n_slices", 3) * math.sin(math.radians(d["ls_angle_deg"])) * px}
bad = [f"{k}: config={tf.get(k)} expected={v:.4f}" for k, v in want.items()
       if tf.get(k) is None or not math.isclose(tf[k], v, rel_tol=0.01)]
print("\n".join(bad) or "pixel sizes OK"); sys.exit(bool(bad))
EOF
```

It exits 1 on a missing value or one more than 1% off. Fix `reconstruct.yml`,
not `deskew.yml` — the deskew values come from the acquisition (§5, caveats
§3). If the deskew geometry differs from the template, that is a per-dataset
edit to both files.

### 5c. Check that every position has the same number of time points

Assemble concatenates every position of the deskew, reconstruct and
virtual-stain stores, and with `time_indices: all` (both shipped templates)
`biahub concatenate` **refuses sources with different numbers of time
points**: it no longer crops or pads silently, so the choice has to be written
into `concatenate.yml`. Each step keeps a position's full T, so a mismatch in the raw store
reaches assemble and fails the run during the init phase. Check the raw store
now. From the biahub checkout, with the position glob for the layout from §3:
`'*/*/*'` for an HCS plate, `'*'` for flat positions:

```bash
.venv/bin/python - <RAW_STORE.zarr> '<POSITION_GLOB>' <<'EOF'
import collections, json, pathlib, sys
store, pattern = pathlib.Path(sys.argv[1]), sys.argv[2]
by_t = collections.defaultdict(list)
for meta in sorted(store.glob(f"{pattern}/0/zarr.json")):
    by_t[json.loads(meta.read_text())["shape"][0]].append(str(meta.parent.parent.relative_to(store)))
for t, positions in sorted(by_t.items()):
    print(f"T={t}: {len(positions)} positions, e.g. {', '.join(positions[:3])}")
sys.exit(len(by_t) != 1)
EOF
```

It exits 1 if the positions disagree, or if none were found (check the glob).
**If they disagree, stop and ask the user.** A short position usually means the
acquisition was stopped partway through a time point. Show the counts per T and
offer:

- **Crop in time to what every position has.** In
  `<OUTPUT>/configs/concatenate.yml`, replace `time_indices: all` with

  ```yaml
  time_indices: {start: 0, stop: <smallest T>}   # stop is exclusive
  ```

  The extra time points of the longer positions are left out of the assembled
  store (they stay in the intermediates), and tracking and QC see only the
  cropped range (`references/caveats.md` §5). If the user also wants to
  subsample, add `step`, e.g. `{start: 0, stop: <smallest T>, step: 2}` for
  every other time point. (`time_indices: min` crops the same way, but the
  range records the number in the config.) Use the smallest T as `stop`: a
  range that runs past the shorter positions pads them, as below.
- **Keep every time point and pad with zeros**: `time_indices: max`, or
  equivalently `{start: 0, stop: <largest T>}` (add `step` to subsample). The
  assembled store takes the longest T, and the missing end of each shorter
  position stays at zero. Nothing is lost, but the padded frames are blank:
  tracks end there, QC may flag them, and nothing in the store marks them as
  padding — say which positions are padded from which time point in the plan,
  and the assemble log lists them (`time_indices ...: padding ...`).
- **Stop and investigate** the acquisition before reconstructing.

Do not pick for them, and record the choice and the reason in the plan (§6).
The same refusal at assemble's init, `time_indices is 'all' but the sources
have different numbers of time points`, means this check was skipped or the
stores no longer match the raw data; it is answered the same way.

## 6. Present the plan

Do not run anything yet. Show the user:

1. The biahub checkout: path, branch, HEAD commit, behind-count, working-tree
   state. Make anything other than *clean `main`, up to date* a visible caveat
   with a recommendation.
2. Resolved input store, its size, position count, channel names, and
   `(T, C, Z, Y, X)` shape. Say that every position has the same T (§5c), or
   how many differ and the `time_indices` the user chose for that.
3. Whether a `0-convert` plate build is needed.
4. Output project directory and the step layout.
5. That configs come from `<BIAHUB>/nextflow/configs/<family>/` at commit
   `<sha>`, every value changed for this dataset, and any template fix headed
   for a PR. If QC is in the step list, that §5a passed — and if it did not,
   what was mismatched and how it was resolved.
6. **The steps this run will perform, listed in order.** Reconstruction proper —
   flat-field → deskew → reconstruct → virtual-stain — always runs; assemble,
   project, segment, track and QC run only if their config is passed
   (biahub#306). Project turns the assembled 3D plate into the 2D (Z=1) plate
   that 2D segmentation and tracking read (neither selects or projects
   Z-planes); segment writes the cellpose labels into that plate and track
   links them (`biahub track` no longer segments), so `--segment_config` and
   `--track_config` go with `--project_config`. Defaults by family, which is
   what to present unless the user says otherwise:

   | family | steps |
   |---|---|
   | A549 / cell-line / organelle | flat-field → deskew → reconstruct → virtual-stain → **assemble → project → segment → track → pyramid → QC** |
   | neuromast / zebrafish / dynatrack | flat-field → deskew → reconstruct → virtual-stain → **assemble → pyramid → QC** (no projection, segmentation or tracking) |

   Pyramid is on by default whenever assemble runs: it adds two coarser levels
   (3 in total, DCA v0.2) to every image and label image of the assembled and
   projected plates, in place, once segmentation and tracking have finished
   writing into them. QC is the very last step and waits for the pyramids.
   `--pyramid_levels 1` skips them.

   Write the list out in the plan rather than naming the family, and say which
   optional steps are being skipped and why. If the user asks to skip anything
   else, reflect that here — the plan is where the step set is agreed.

   For neuromast/zebrafish say that the assembled store is the deliverable and
   that tracking is simply not run — there is no tracking by-product any more
   (`caveats.md` §4).

   **When project runs, list what it projects.** State each assembled channel
   with its `project.yml` method, and name any assembled channel the config
   does not list (§5a's `NOT PROJECTED` lines). Those need a decision from the
   user before launch: add the channel to `project.yml`, or confirm it is left
   out on purpose. Do not launch with an unprojected channel the user has not
   seen — the pipeline's intention is that every assembled channel is
   projected unless the user explicitly decided otherwise.

   **State the directory numbers this run will produce.** The number is the
   step's position among the steps performed, not a fixed label, so a neuromast
   run and an A549 run both end at `4-assemble`: project writes its plate
   beside the assembled one as `4-assemble/<DATASET>_proj.zarr`, segment
   writes its label images into that plate and tracking its tracked labels and
   tracks; none of the three has a directory of its own. Older A549 runs on
   disk also have `5-track` (or `5-assemble` /
   `4-track`, from when the numbers were fixed) — say so if the user is
   comparing against one.
7. Known caveats that apply to this dataset.
8. Rough wall-time and that `-resume` is on.
9. Whether Slack notifications are on, and who will be @-mentioned at run end.
   If `$BIAHUB_SLACK_WEBHOOK` or `$BIAHUB_SLACK_ID` is missing, note it here as a
   caveat with the offer from §1a-ter — not as a blocker.
10. If QC is in the step list, that the GitHub credential from §1a-bis is in
    place. If it is not and cannot be, say the run will proceed without QC
    rather than silently dropping it — the step set in item 6 has to match what
    is actually going to run.
11. **Whether intermediates will be deleted** (`CLEANUP_INTERMEDIATES` in the run
    script, biahub#292). With cleanup on, once the last step (QC) finishes the
    pipeline deletes the flat-field, deskew, reconstruct and virtual-stain
    directories, the `slurm_output/` placeholders and `.iohub-progress/` resume
    markers beside the final stores, and the Nextflow work directory. Only the
    assembled and projected stores (with the segmentation and tracking labels
    and the tracks inside the latter), the QC report
    and the logs under `nextflow/` remain, and the run is final: any rerun
    recomputes from raw. The template's `CLEANUP_INTERMEDIATES="auto"` lets the
    pipeline decide at launch (`cleanup_decision` in
    `nextflow/modules/cleanup.nf`); check `concatenate.yml` now so the plan says
    what it will decide:
    - **No cropping** — `time_indices`, `channel_names`, `X_slice`, `Y_slice`
      and `Z_slice` are all absent or `all` (or a per-source list of `all`),
      with `time_indices: max` also counting as no cropping (it pads, §5c):
      leave `auto`; the pipeline turns cleanup on. Say so in the plan.
    - **Any cropping**: `auto` would leave the intermediates, but set
      `CLEANUP_INTERMEDIATES=false` explicitly so the script records the
      decision. Name the cropping fields and **ask the user explicitly** whether
      to delete the intermediates anyway — the data cropped out of the assembled
      store exists nowhere else. Set `true` only on a clear yes; no answer means
      `false`.

Get explicit approval.

## 7. Scaffold the output directory

```bash
mkdir -p <OUTPUT>/configs <OUTPUT>/nextflow
cp <BIAHUB>/nextflow/configs/<family>/*.yml <BIAHUB>/nextflow/configs/<family>/*.yaml \
   <OUTPUT>/configs/
chmod g+w <OUTPUT>/configs/*
```

Both globs: the step configs are `.yml` and the QC configs are `.yaml`.

**Every file you place in `<OUTPUT>` must be group-writable** so the rest of
the lab can edit configs and relaunch. The project directories are setgid with
a default ACL: setgid supplies the group, but the ACL overrides umask, so a
copied file's group bits follow the *source* file's mode — `cp` from the repo
(644) yields `rw-r--r--`. Hence the `chmod` after every copy. Plain `cp` only:
`cp -p`/`-a` would also carry over the checkout's group.

The run script passes all six optional configs — `--concatenate_config`,
`--project_config`, `--segment_config`, `--track_config`, `--qc_config`
(`qc.yaml`), `--qc_project_config` (`qc_project.yaml`) — so an A549 run needs no
edit. **For a neuromast/zebrafish run, or any step the user asked to skip,
DELETE that flag's line** from the `nextflow run` call and note the skip in a
comment above it, so the script still records what this run did. A neuromast
run deletes `--project_config`, `--segment_config`, `--track_config` and
`--qc_project_config`; none of `project.yml`, `segment.yml`, `track.yml` or
`qc_project.yaml` need exist for that family, and none does for zebrafish.
`--qc_track_config` is gone: passing it errors at launch.

Delete rather than comment: a `#` inside a backslash-continued command does not
start a comment line — the continuation swallows it, every flag below is dropped
including `-resume`, and bash then tries to run the remainder as a command.

Edit the copies for this dataset. Copy `templates/run_mantis_v2.sh` to
`<OUTPUT>/run_mantis_v2.sh`, fill in `DATASET`, `DATA_DIR`, `PROJECT_DIR`,
`BIAHUB_PROJECT`, `chmod 775` (not `+x` — it must also be group-writable).
The script stays in the output directory as the run's provenance record, and
re-grants `g+w` on itself and the configs at every launch.

Set `CLEANUP_INTERMEDIATES` as agreed in §6 item 11: leave `auto` when
`concatenate.yml` does not crop, `false` when it crops and the user did not opt
in, `true` only when they did. `provenance.txt` records the requested value;
the pipeline logs what it resolved at launch, as
`cleanup_intermediates: on|off (<reason>)` in `.nextflow.log` and on the
`cleanup:` line of the run-start Slack message — check it matches the plan.

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

## 9. Monitor — unattended, every 30 minutes

As soon as §8's launch check passes, start the monitoring loop yourself, without
waiting to be asked (Skill tool, skill `loop`):

```
/loop 30m /reconstruct-with-nextflow monitor <OUTPUT>
```

Each firing runs one monitoring pass (§9a) and ends. A run is never left
unwatched: a run that dies at 3 a.m. on a transient storage error is relaunched
by the next pass instead of waiting for someone to notice it. Tell the user the
loop is running and how to stop it. **The loop lives in this Claude Code
session.** If the session exits, monitoring stops; re-arm it from a new session
with the same `/loop` line. The pass is cheap: one status script, one line of
output when nothing is wrong.

**The pipeline notifies Slack itself** — run start, each step's completion, and
run end (`nextflow/modules/notify.nf`), so do not re-send those. What is left for
you is a relaunch or a terminating error the pass has diagnosed, a position on
attempt 4 of 5, and the wrap-up; send those with `templates/notify.sh`.
Notifications need `$BIAHUB_SLACK_WEBHOOK` and `$BIAHUB_SLACK_ID`; without them
messages print instead of posting and nothing fails. For a `--max_positions 1`
smoke test, launch with `env -u BIAHUB_SLACK_WEBHOOK` to keep the channel quiet.

### 9a. Monitoring pass

Invoked as `/reconstruct-with-nextflow monitor <OUTPUT>`. Skip §1–8 and do only
this:

1. Run the read-only status script and read its output:

   ```bash
   bash <SKILL_DIR>/templates/run_status.sh <OUTPUT>
   ```

   On the first pass of a session, also read `references/monitoring.md` and
   `references/recovery.md`.
2. Read `<OUTPUT>/nextflow/monitor.log` (absent until the first action). It
   records what earlier passes did, so a pass never repeats a relaunch or a
   notification.
3. Act on the `state` line:

   | state | action |
   |---|---|
   | `running` | Nothing. Notify once per position about any `attempt4` line not yet in `monitor.log`. Print one line of progress. |
   | `completed` | §11 wrap-up, then stop the loop. |
   | `aborted`, `head-lost` | Relaunch or escalate, by the rules below. |
   | `not-started` | Nothing if the launch is under 10 minutes old; otherwise tell the user. |

4. Append one line per action to `<OUTPUT>/nextflow/monitor.log`, as
   `<ISO time> <state> <relaunched|escalated|notified> <signature>`, and
   `chmod g+w` it. The signature is `<failed task> | <error line>` from the
   status output, or `head-lost`.

**Relaunch automatically** only when all of these hold:

- there is no `cleaned yes` line, since a cleaned run is final (§10);
- the cause is transient: the `error` line is an `OSError`, `IOError` or
  "Input/output error", or a torn-shard signature (`the checksum is invalid`,
  `encoded shard is smaller than`, `blosc encoded value is invalid`), or the
  state is `head-lost` (the head was killed: allocation ended, node lost);
- `monitor.log` has no `relaunched` line with the same signature. One automatic
  relaunch per signature: the same failure twice is persistent, not transient.

Relaunch exactly as §10 does, in the run's tmux session. Recreate the session
first if it is gone (`tmux new-session -d -s nf_<DATASET> -c <OUTPUT>`), then:

```bash
tmux send-keys -t nf_<DATASET> "bash ./run_mantis_v2.sh" Enter
```

Send `notify.sh "🔁 <DATASET>: relaunched with -resume after <error> on <task>"`
without `--ping`, and confirm on the next pass that the state is `running`.

nextflow.config already retries a bare `OSError` inside the run, after
`params.oserror_retry_delay_minutes` (10), sharing the 5 retries with
preemption. An `OSError` that still reaches `aborted` has failed every one of
those retries, so the pass's relaunch is the last automatic attempt.

**Escalate everything else** without relaunching. That covers any other
traceback, including `FileNotFoundError` and other `OSError` subclasses, a config
error, retries exhausted on exits 130–145, a repeated signature, and a cleaned
run. Diagnose with `references/recovery.md`, send
`notify.sh --level error --ping` with the cause, the error line and the work
directory, record it in `monitor.log`, stop the loop, and give the user the
diagnosis. Never delete data from a pass.

To stop the loop, delete its scheduled job: `CronList` to find it, then
`CronDelete`. If it runs self-paced, use `ScheduleWakeup` with `stop: true`.

## 10. Handle errors

Classify before acting — `references/recovery.md` has the decision table:

- **Exit 130–145** (preemption, timeout, OOM): Nextflow retries up to 5 times.
  Do nothing unless retries are exhausted.
- **Exit 1 ending in a bare `OSError`** (Lustre EIO, `[Errno 71] Protocol
  error` from fsync): Nextflow retries it itself after 10 minutes. A run that
  still aborts on one is relaunched once by the monitoring pass (§9a).
- **Checksum / Lustre EIO / torn shard**: rerun with `-resume` first — iohub
  replaces torn shards and resumes per unit. Only if it fails
  identically again, write a repair proposal for the user or hand it to the
  **job-io-error-repair** agent. **Never delete zarr data from this skill.**
  The pipeline's own `--cleanup_intermediates` step is the one sanctioned
  deletion, and it only runs after every step has succeeded.
- **Exit 1/2 with a Python traceback**: real bug or bad config. Fix, relaunch
  with `-resume`.

Restarts are always `bash ./run_mantis_v2.sh` — the script passes `-resume`.

**Once cleanup has run, the run is final.** Its intermediates, work directory
and resume markers are gone, so relaunching the script reruns everything from
raw and rewrites the final stores. To reprocess, prefer a fresh output
directory (`<DATASET>_rerun`) over relaunching in place.

## 11. Wrap up

1. Normalize channel names on the assembled plate, and on the projected plate
   beside it when project ran (`DATASET=<DATASET>_proj`, same directory), with
   `templates/rename_channels.py` (idempotent; `caveats.md` §2 — check first
   whether a `rename-channels` CLI has landed on main, making this obsolete).
2. Verify the assembled store (`<N>-assemble/<DATASET>.zarr`, `4-assemble`
   unless earlier steps were skipped) opens with iohub; report shape, channels,
   size on disk. This is the deliverable for every family. When segmentation
   and tracking ran, their output is inside the projected plate: check a
   position has `labels/nuclei` (the segment config's model name) and
   `labels/tracks_nuclei` (`tracks_` plus the track config's `labels`; both
   uint32 (T, 1, Z, Y, X) at the image's full Z), `tracks.geff` and
   `tracks_<row>_<col>_<fov>.csv`, and report them beside the image data.
3. If QC ran, report its verdict: the `QC_SUMMARY` line from the
   pipeline log (`pass=`/`fail=`/`gates_fail=`), and point at the report at
   `<OUTPUT>/qc/report/index.html` — one page, one tab: the assembled store
   (A549's includes the cell counts). A gate failure does NOT fail the run:
   `imaging-qc gate` exits 0 either way, so a failing verdict is only visible
   in the summary line, the report, and the `tables/qc/` parquet inside the
   store.
4. Report per-step task counts, failures, retries, and wall time from
   `<OUTPUT>/nextflow/trace.txt`; point at `report.html` and `timeline.html`.
   If cleanup was on, report what `<OUTPUT>/nextflow/intermediates_cleaned.txt`
   records — the decision and one `removed`/`absent` line per target — and
   confirm `<OUTPUT>/nextflow/work` holds no task directories (Nextflow leaves
   its empty two-character prefix directories behind; that is expected). Do
   not `du` the run directory to report
   space freed — it runs to terabytes; use `df` on the filesystem if a number
   is wanted.
5. Confirm the pipeline's automatic run-end message landed, then send a wrap-up
   only for what the pipeline cannot know: the channel-rename result, the iohub
   verification, and size on disk.
6. Flag anything needing a human eye: positions that passed only after many
   retries, steps far slower than the reference run, unexpected channel counts.
