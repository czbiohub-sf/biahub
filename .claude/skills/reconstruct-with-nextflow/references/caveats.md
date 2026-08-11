# Pipeline-specific caveats

Read this before writing configs or launching. Each item has bitten a real run.

---

## 1. Neuromast / zebrafish acquisitions have no HCS plate

The pipeline fans out over plate positions (`biahub nf list-positions`), so its
input **must** be an HCS plate. A549/cell-line acquisitions already are one — the
root `zarr.json` carries `attributes.ome.plate` with rows `B`, `C`, …

Neuromast, zebrafish and dynatrack acquisitions are **not**. They are flat
`bioformats2raw.layout: 3` stores with positions named `{R}Pos{C}` at the root
(`1Pos0`, `1Pos1`, `2Pos0`, …). Feeding one to the pipeline directly fails at
`list_positions`.

**Fix:** build a plate at `<OUTPUT>/0-convert/<DATASET>.zarr` that *symlinks* the
raw arrays into a row/column/field layout — no pixel copy. Mapping is
`{R}Pos{C}` → `0/{R}/{C}`.

Use the **build-hcs-plate** agent for this; it handles the mapping, the
group metadata, the provenance symlinks, and recovery of positions whose group
`zarr.json` was left empty by an interrupted instrument write (common). Canonical
scripts to adapt:

- `/hpc/projects/tlg2_mantis/2026_06_25_dynatrack_48hpf/build_plate_48hpf.py`
  (includes the corrupt-metadata recovery fallback)
- `/hpc/projects/tlg2_mantis/2026_07_24_dynatrack/build_plate.py`

Verify before launching:

```bash
<BIAHUB>/.venv/bin/python -c "
from iohub.ngff import open_ome_zarr
with open_ome_zarr('<OUTPUT>/0-convert/<DATASET>.zarr', mode='r') as p:
    print(len(list(p.positions())), 'positions', p.channel_names)"
```

Detection check, cheap and decisive:

```bash
python3 -c "
import json; a=json.load(open('<STORE>/zarr.json'))['attributes']
print('HCS plate' if 'plate' in a.get('ome',{}) else 'FLAT — needs 0-convert')"
```

---

## 2. Channel renaming after assemble

The assembled plate inherits whatever names its sources emitted: `BF - Oblique`
from deskew, `Phase3D*` from reconstruct, `nuclei`/`membrane` from virtual-stain,
and camera/filter strings like `mCherry EX561 EM600-37` from the raw fluorescence
channels. Downstream steps then depend on acquisition-specific strings that differ
between experiments.

**The convention is defined in [biahub#291](https://github.com/czbiohub-sf/biahub/issues/291)**
and implemented by `templates/rename_channels.py`. First match wins:

| incoming | renamed to |
|---|---|
| `BF - Oblique` | `BF` |
| `Phase3D*` | `Phase3D` |
| `nuclei` | `nuclei_prediction` |
| `membrane` | `membrane_prediction` |
| already canonical, or already `raw `-prefixed | left alone |
| anything else | `raw <original>` |

The rule is **idempotent** — the passthrough row is what makes re-running safe,
so it never produces `raw raw mCherry ...`.

Run once, after `5-assemble`, from the directory holding the store (the script
reads the stem from `$DATASET` and appends `.zarr`):

```bash
cd <OUTPUT>/5-assemble
DATASET=<DATASET> <BIAHUB>/.venv/bin/python rename_channels.py
```

**Why `nuclei` → `nuclei_prediction`:** `biahub virtual-stain` names its outputs
verbatim from `data.init_args.target_channel`, dropping the `_prediction` suffix
that viscy's `HCSPredictionWriter` used to append — a regression tracked in
[biahub#288](https://github.com/czbiohub-sf/biahub/issues/288). The rest of biahub
still keys off the suffix (`TrackingSettings` defaults, `track.py`'s
`mem_nuc_contour`, viscy's airtable `filter_raw_channels`), so the rename bridges
the gap.

**⚠ Ordering.** This script runs *after* the whole Nextflow pipeline, but `4-track`
runs *inside* it and reads the assembled plate — so **track sees the pre-rename
names.** Any channel a track config references must exist under its un-renamed
name at track time. The A549 configs work around this by putting the suffix
directly in the VS `target_channel` (`nuclei_prediction`/`membrane_prediction`),
which makes the rename rule a no-op safety net for those channels. The shipped
templates do the same.

**Stopgap status:** #291 proposes doing this inside `concatenate`, which already
resolves the output channel list, so the assembled plate would be correct the
first time. PRs [#260](https://github.com/czbiohub-sf/biahub/pull/260) and
[#250](https://github.com/czbiohub-sf/biahub/pull/250) carry a `rename-channels`
CLI and Nextflow subworkflow. **Neither has landed on main** — check before running
this by hand, and drop the manual step once one of them merges.

---

## 3. Config values that must be checked per dataset

Copying configs from a reference run is the right move, but these five drift:

**`deskew.yml` — `pixel_size_um` and `scan_step_um`.** `pixel_size_um` is the
lateral (YX) pixel size of the *input* store; `DeskewSettings` needs it to derive
`px_to_scan_ratio = pixel_size_um / scan_step_um`, and it is no longer read from
the zarr. `scan_step_um` comes from the acquisition — read
`/hpc/instruments/cm.mantis/<DATASET>/config.yaml`, do not assume 0.150.

**`reconstruct.yml` — `input_channel_names`.** Must name the brightfield channel
exactly as it appears in the deskewed store (`["BF - Oblique"]` today). A mismatch
fails at `compute_transfer_function`, after flat-field and deskew have already
burned hours.

**`virtual_stain.yml` — `ckpt_path` and the model geometry.** Two different
models are in use, and the model/data fields must agree with the checkpoint:

| family | checkpoint | `in_stack_depth` / `z_window_size` | `stem_kernel_size` | targets |
|---|---|---|---|---|
| neuromast / zebrafish | `/hpc/projects/comp.micro/virtual_staining/models/fcmae-3d/fit_v2/pretrain_end2end/lightning_logs/finetune_VS_end2end_v1_test6_prefetch2_nopersistwork_restart_2/checkpoints/epoch=64-step=24960.ckpt` | 21 | `[7, 4, 4]` | `nuclei_prediction`, `membrane_prediction` |
| A549 | `/hpc/projects/organelle_phenotyping/models/VSCyto3D-A549-infection-finetune/4gpu_bf16_bs16_to_ep7/checkpoints/epoch=7-step=832.ckpt` | 15 | `[5, 4, 4]` | `nuclei_prediction`, `membrane_prediction` |

Both templates carry the `_prediction` suffix in `target_channel` deliberately —
see §2 for why (biahub#288). Older zebrafish configs say plain `nuclei`/`membrane`;
that still works, but only if nothing reads the store before the assemble-stage
rename.

`data.init_args.z_window_size` **must equal** `model.init_args.model_config.in_stack_depth`.
A mismatch does not raise — the model silently stalls in the inference path.

The VS config must contain **only** `model`, `data`, and `ckpt_path` at the top
level (plus the optional `sliding_window_step` / `device` /
`output_ome_zarr_version`). The old `viscy predict` blocks `trainer:` and
`return_predictions:` are rejected by `build_predict_parser` — older projects
still have a `predict.yml` carrying them, so strip those when copying. The
current pipeline flag is `--virtual_stain_config`.

`data.init_args.data_path` is injected per-position by biahub; leave it out.

**`track.yml` — schema.** `TrackingSettings` forbids extra keys, and the schema
changed. Current fields: `output_mode`, `z_slicing`, `target_channel`, `fov`,
`input_images`, `segmentation_method`, `cellpose_config`, `tracking_config`.
The **old** fields `mode`, `z_range`, and `focus_config` are now rejected:

```
mode                            -> output_mode
z_range + focus_config.z_window -> z_slicing.{method, window_size, focus_channel}
```

`focus_config`'s `NA_det` / `lambda_ill` / `pixel_size` have no equivalent —
focus detection now runs waveorder's `focus_from_transverse_band` internally on
`z_slicing.focus_channel`. Several configs under `/hpc/projects/tlg2_mantis/`
still carry the old schema; do not copy them without migrating.

Also: `target_channel` and `cellpose_config.input_channel` must match a channel
name in the **assembled** plate, because track reads `5-assemble`, not
`3-virtual-stain` — and they must match the **pre-rename** names, since track
runs inside the pipeline while the rename (§2) runs after it.

**`z_slicing.focus_channel` resolves against `input_images`, not the store.**
`apply_focus_slicing` (`biahub/track.py`) looks the focus channel up in the dict
of channels `input_images` actually loaded and raises if it is missing:

```
ValueError: focus_channel 'Phase3D' not in loaded channels ['nuclei_prediction'].
```

So `method: focus` requires the focus channel to *also* appear in
`input_images.channels`, even though it exists in the assembled plate. Observed
for real: a zebrafish template declaring `focus_channel: Phase3D` while loading
only `nuclei_prediction` failed all 8 positions × 5 retries — **after** flat-field,
deskew, reconstruct, virtual-stain and assemble had all succeeded, i.e. the
cheapest possible bug discovered at the most expensive possible moment. Both
shipped templates now use `method: all`. When `focus_channel` is unset the code
falls back to the first loaded channel, which is why `method: all` is the safe
default.

This is the argument for biahub#304 (validate every step's config up front): a
one-line config error in the last step burned a whole run's compute.

**`concatenate.yml` — `concat_data_paths` stay as `placeholder`.** The assemble
subworkflow injects the three real source paths via `--concat-data-paths`. Three
placeholder entries and three `channel_names` entries, one per source (deskew,
reconstruct, virtual-stain). Do not "fix" them to real paths.

---

## 4. Neuromast/zebrafish datasets are not tracked

**Tracking is an A549 step.** For the neuromast/zebrafish family the deliverable
is `5-assemble`; `4-track` is not part of the pipeline's purpose there, and its
parameters (cell diameter, min/max area, linking distance) are tuned for A549
cells, not neuromasts.

There is no way to express this today: `nextflow/mantis-v2.nf` errors without
`--track_config` and unconditionally wires `track_wf` after assemble. So a
zebrafish run still passes a track config, still produces a `4-track` store, and
that store is a by-product to discard.

What this means in practice:

- Pass `<BIAHUB>/nextflow/configs/zebrafish/track.yml` (a marked placeholder) so the run
  validates and starts.
- Say in the plan that `5-assemble` is the deliverable and `4-track` is
  discarded, so the user is not surprised by the extra step's wall time.
- Do not report tracking results for these datasets, and do not tune the track
  config to "make tracking work" unless the user asks for neuromast tracking
  explicitly.

Making steps optional is tracked in [biahub#306](https://github.com/czbiohub-sf/biahub/issues/306).
Once that lands, drop the placeholder and stop after assemble.

## 5. Track reads the assembled plate, not the intermediates

As wired today, `track_wf` takes `5-assemble/<DATASET>.zarr` for *both* its
inputs. Two consequences:

- Any Z/Y/X crop or `time_indices` subset in `concatenate.yml` is what tracking
  sees.
- Tracking runs strictly *after* assemble, not in parallel with it. The whole
  plate must assemble before any track task starts.

## 6. Assemble is a single job on one reserved node

`concatenate --cluster debug` iterates every position in-process, so `5-assemble`
is one large SLURM job rather than a fan-out — 16 TiB of I/O and tens of
thousands of write units for a typical plate. A late kill used to discard all of
it; `biahub concatenate --resume` (which the Nextflow task passes) makes it
recoverable. Give it room: it is the single longest step.

## 7. Shard geometry and `shards_ratio`

`shards_ratio` is a ratio over the **chunk** shape, not an absolute shard shape
(`shards = chunks * shards_ratio`, elementwise TCZYX). Changing the T entry
without recomputing ZYX silently changes the spatial extent too. For an assemble
output `[T, 6, 86, 1664, 1193]` with chunks `(1,1,16,256,256)`, the ZYX entries
that reproduce today's extent are `ceil(86/16)=6, ceil(1664/256)=7,
ceil(1193/256)=5`.

Larger T shards mean fewer, bigger files but a much larger RAM request and a much
worse blast radius when one tears — a shard spanning 10 timepoints loses all 10.
Leave the T ratio alone unless you have a measured reason.

## 8. Preemption is expected, not an error

Per-position work runs on the `preempted` partition. SLURM reclaims jobs there
routinely; they exit 143 and Nextflow resubmits (`errorStrategy` retries exit
130–145, `maxRetries = 5`). A run with dozens of retries in `trace.txt` is normal.
Only escalate when retries are exhausted, or when the same position fails
identically every time — that is the torn-shard signature, see `recovery.md`.

## 9. SLURM log files look inverted

`slurm_logs()` in `nextflow/modules/common.nf` deliberately crosses
`--output`/`--error` to undo Nextflow's fd swap. The net effect: **the task's
real output is in `nextflow/slurm_output/<step>/*_<jobid>.out` and the `.err`
file is empty.** Also, `.command.log` in the Nextflow work dir is empty, because
the `clusterOptions` override wins over Nextflow's own `-o`. Read the
`slurm_output` files, not the work dir logs.

## 10. Do not edit the biahub checkout during a live run

Editing files in `$BIAHUB_PROJECT` changes Nextflow task hashes and invalidates
`-resume`, so a restart recomputes everything. Get the branch right *before*
launching, then leave it alone.

## 11. `uv sync` before launching — but check for out-of-band packages first

Run `uv sync --project <BIAHUB>` before every run. It is the whole provisioning
step, and once it has run, activating the venv is all the pipeline needs. On a
healthy checkout it is a ~1s near-no-op.

**The one thing to check first.** `uv sync` removes anything not in `uv.lock`, by
design. If a checkout's `.venv` carries packages installed out of band, sync will
prune them. This is per-checkout, so measure the checkout you are about to use
rather than assuming:

```bash
uv sync --all-extras --dry-run 2>&1 | grep -E "^ *- "   # lists what would go
```

Read the list before syncing:

- **Empty** — nothing would be removed. Sync. This is the expected state, and it
  is what a checkout tracking `main` gives today.
- **Unused leaf packages only** — nothing in the pipeline imports them, and
  `.venv/bin/{biahub,viscy}` still work after the sync. Sync.
- **Anything load-bearing** — `cupy-cuda12x`, `tracksdata`, `stitch`, `dexp`,
  `dask-cuda`, `ilpy`, `pyscipopt` — **stop and raise it with the user.** One
  checkout had 27 such packages and syncing it broke a working environment. A
  broken `cupy` cascades: `iohub`, `cytoland` and `ultrack` then all fail to
  import with `AttributeError: module 'cupy' has no attribute 'ndarray'`.

Do not memorize a package list for the middle case — lock membership moves. The
`nd2`/`ome-types`/`xsdata` closure was outside the lock and prunable one week and
pulled *into* it the next by an `iohub` git pin. Run the `--dry-run` and read what
it actually says.

That earlier damage is why this section used to forbid `uv sync` outright. The
hazard was that checkout's out-of-band set, not `uv sync` itself.

On Lustre a sync can also fail *partway* — `failed to remove directory ...
__pycache__: Directory not empty (os error 39)`. If that happens, **stop**: a
half-completed uninstall can leave a package whose `dist-info` lost its `RECORD`,
which then shadows the real module. The repair is to delete the corrupt package
directory plus its `dist-info` so uv can lay it down fresh, then `uv pip install`
the exact pins (`uv pip install` does *not* prune).

Verify after syncing, before launching:

```bash
uv lock --check                                  # lockfile current
<BIAHUB>/.venv/bin/biahub nf --help              # CLI importable
```

Use the venv's own entry point, not `uv run` — `uv run` performs an implicit sync
on every invocation, which is exactly what the launch procedure resolves once up
front.

**Two launch forms both work**; verified that each exports `VIRTUAL_ENV` and a
`.venv/bin`-first `PATH` to child processes, which is what carries the
environment through `nextflow` → `sbatch` → compute node:

```bash
uv sync --project <BIAHUB> && source <BIAHUB>/.venv/bin/activate && nextflow run ...   # preferred
uv run --project <BIAHUB> nextflow run ...                                             # equivalent
```

Prefer the first. `uv run` re-syncs on every invocation, so a lockfile change
mid-run could shift the environment under a live pipeline, and it hides which env
was used from the run log. Explicit activation is the provenance record. Note
`nextflow` itself comes from `module load nextflow`, not the venv, so `uv run`
buys nothing here beyond the implicit sync.

## 12. Environment churn: fixed, and how to recognize it coming back

The pipeline used to prefix every task with `uv run --project <path> biahub` (a
`biahub_cmd()` helper, plus a `--extra stain` variant in `virtual_stain.nf`).
With `maxForks = 30` that meant up to 30 concurrent processes each asking uv to
re-materialize one shared `site-packages`. Every invocation reinstalled a package
and never converged:

```
Uninstalled 1 package in 54ms / Installed 1 package in 232ms
```

Two changes removed this:

1. **The wrapper is gone.** No `--biahub_project` parameter, no `biahub_cmd()`,
   no `stain_cmd()`. Tasks call `biahub`/`viscy` bare and inherit the activated
   environment (§ above, and the ENVIRONMENT CONTRACT comment in
   `nextflow/modules/common.nf`).
2. **The root cause of the churn was a stale `uv.lock` entry.** `uv.lock`
   recorded `viscy-transforms==0.0.0.post215.dev0+4b62365`, but the wheel built
   from that same pinned commit declares `0.1.0a0`. uv compared the two, decided
   the install was not fresh, and reinstalled it — forever. Diagnose this class
   of problem with:

   ```bash
   uv sync --dry-run -v 2>&1 | grep -i "does not match resolved version"
   ```

   The fix was `uv lock --refresh-package viscy-transforms`, a one-line lockfile
   change. Expect it to recur whenever a git-sourced dependency's declared
   version changes without the pinned rev changing.

A healthy checkout now shows no churn at all — `uv sync --dry-run` reports
"Would make no changes" and repeated runs install nothing. If you see
`Uninstalled 1 package` on every invocation again, it is a version mismatch like
the above, not something to work around with `--no-sync`.
Was tracked in [biahub#308](https://github.com/czbiohub-sf/biahub/issues/308).

## 13. A param passed on the command line is a String

Nextflow types params declared in a config `params { }` block, but a param
supplied as `--foo 1` on the command line arrives as the **String** `"1"`. Any
operator or directive that needs a number must coerce it.

This silently broke `--max_positions`, the flag the run script advertises for a
quick smoke test. `positions.take("1")` matches no `take` overload, so Nextflow
fell back to resolving `take` as a process and aborted the whole run at launch:

```
[ERROR] Missing process or function take([DataflowStream[?], 1])
```

The error names `take` and points at the `collect_positions(...)` call site, which
reads like a broken channel wiring rather than a type problem — the quotes are not
shown in the rendered arguments. The default path (`max_positions = 0`, from the
config, already an int) never reaches `take`, so every production run worked and
only the smoke-test path failed.

Both sites are coerced now (`(params.max_positions ?: 0) as int` in
`collect_positions`, and `queueSize` in the slurm profile). **When adding a
numeric param, coerce it at the point of use** and test it via the CLI, not just
via its config default — a config-default test cannot catch this.

**What a `--max_positions N` run produces.** Only the per-position `run_*` tasks
honour the limit. Every `init_*` step globs the whole input (`-i <store>/*/*/*`),
so it scaffolds the output plate for *all* positions. A one-position smoke test
therefore leaves a full-width plate in which only the first position holds data —
correct and expected, not a partial-write bug. Do not treat such a store as a
deliverable, and delete it before a real run so `-resume` cannot reuse it.
