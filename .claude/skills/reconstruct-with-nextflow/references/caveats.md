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
uv run --project /hpc/mydata/taylla.theodoro/repo/biahub python -c "
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
DATASET=<DATASET> uv run --project <BIAHUB> python rename_channels.py
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

- Pass `templates/configs/zebrafish/track.yml` (a marked placeholder) so the run
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

## 11. NEVER run `uv sync` on the biahub checkout — verify instead

**This is the one instruction in this skill that has caused damage.** An earlier
version said to run `uv lock && uv sync` before launching, on the theory that
otherwise 30 concurrent tasks would each resolve dependencies. Doing that broke a
working environment.

**Why.** The `.venv` on this checkout carries **27 packages that are not in
`uv.lock`** — installed out of band, and load-bearing: `cupy-cuda12x`,
`tracksdata`, `stitch`, `dexp`, `dask-cuda`, `dask-image`, `distributed`, `ilpy`,
`pyscipopt`, `polars`, `pims`, and their closure. `uv sync` treats anything absent
from the lockfile as extraneous and removes it. That is correct uv behaviour and
exactly the problem: **every** variant prunes them — `uv sync` would remove 33,
`uv sync --all-extras` still removes 27. Confirm before believing otherwise:

```bash
uv sync --all-extras --dry-run 2>&1 | grep -cE "^ *- "   # 27 = do not sync
```

It also fails *partway* on Lustre — `failed to remove directory ... __pycache__:
Directory not empty (os error 39)`, and a half-completed uninstall can leave a
package whose `dist-info` lost its `RECORD`, which then shadows the real module.
A broken `cupy` cascades: `iohub`, `cytoland` and `ultrack` all fail to import
with `AttributeError: module 'cupy' has no attribute 'ndarray'`.

**Do this instead — verify, never mutate:**

```bash
uv lock --check                     # lockfile current? (no error = yes)
time uv run --project <BIAHUB> biahub nf --help   # ~0.5s = env materialized
```

A fast `uv run` means there is no resolution work to do and nothing to sync. If
the env genuinely needs provisioning, that is a deliberate maintenance task with
the user in the loop — not a pre-launch step. Never run a bare `uv sync` here, and
if one fails with Lustre `ENOTEMPTY`, **stop**: the failure mode is a broken env,
not a no-op.

If it is already broken, the repair is to delete the corrupt package directory
plus its `dist-info` so uv can lay it down fresh, then `uv pip install` the exact
pins (`uv pip install` does *not* prune). Recovering the out-of-band set means
reinstalling `stitch` and `tracksdata`, whose dependency closure pulls most of the
other 25 back.

## 12. Per-task `uv run` mutates the shared venv

Every `uv run --project ...` invocation reinstalls one package — measured, 3 runs
in a row, never converging:

```
Uninstalled 1 package in 54ms / Installed 1 package in 232ms
```

With `maxForks = 30`, that is up to 30 concurrent processes writing the same
`site-packages`. `virtual_stain.nf` compounds it by using a different environment
spec (`--extra stain`) from every other step's plain `biahub_cmd()`, so successive
tasks ask uv for different environments against one shared venv.

`uv run --no-sync` eliminates the mutation entirely (verified: no churn across
repeated runs). The pipeline does not pass it today — tracked in [biahub#308](https://github.com/czbiohub-sf/biahub/issues/308). Until
that lands, expect this churn in a live run; it has not been observed to break a
run, but it is the reason not to add any *further* environment mutation on top.
