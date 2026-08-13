# Pipeline-specific caveats

Read this before writing configs or launching. Each item has bitten a real run.

---

## 1. Neuromast / zebrafish acquisitions have no HCS plate

The pipeline fans out over plate positions, so its input must be an HCS plate.
A549/cell-line acquisitions already are one (root `zarr.json` has
`attributes.ome.plate`). Neuromast/zebrafish/dynatrack acquisitions are flat
`bioformats2raw.layout: 3` stores with positions named `{R}Pos{C}` at the root.

The pinned iohub enumerates flat stores, so `list_positions` alone works — but
every `init_*` step globs `-i <store>/*/*/*`, track configs use `fov: "*/*/*"`,
and concatenate builds a plate, so a plate is still required. (New zebrafish
acquisitions will eventually be written as plates; until then 0-convert stands.)

**Fix:** build a plate at `<OUTPUT>/0-convert/<DATASET>.zarr` that *symlinks*
the raw arrays into `{R}Pos{C}` → `0/{R}/{C}` — no pixel copy. Use the
**build-hcs-plate** agent; it handles the mapping, group metadata, provenance
symlinks, and recovery of positions whose group `zarr.json` was left empty by
an interrupted instrument write. Canonical scripts to adapt:

- `/hpc/projects/tlg2_mantis/2026_06_25_dynatrack_48hpf/build_plate_48hpf.py`
  (includes the corrupt-metadata recovery fallback)
- `/hpc/projects/tlg2_mantis/2026_07_24_dynatrack/build_plate.py`

Detection check:

```bash
python3 -c "
import json; a=json.load(open('<STORE>/zarr.json'))['attributes']
print('HCS plate' if 'plate' in a.get('ome',{}) else 'FLAT — needs 0-convert')"
```

Verify before launching:

```bash
<BIAHUB>/.venv/bin/python -c "
from iohub.ngff import open_ome_zarr
with open_ome_zarr('<OUTPUT>/0-convert/<DATASET>.zarr', mode='r') as p:
    print(len(list(p.positions())), 'positions', p.channel_names)"
```

---

## 2. Channel renaming after assemble

The assembled plate inherits its sources' names (`BF - Oblique`, `Phase3D*`,
`nuclei`/`membrane`, camera strings like `mCherry EX561 EM600-37`). The
convention in [biahub#291](https://github.com/czbiohub-sf/biahub/issues/291) is
implemented by `templates/rename_channels.py` — first match wins, idempotent:

| incoming | renamed to |
|---|---|
| `BF - Oblique` | `BF` |
| `Phase3D*` | `Phase3D` |
| `nuclei` | `nuclei_prediction` |
| `membrane` | `membrane_prediction` |
| already canonical, or already `raw `-prefixed | left alone |
| anything else | `raw <original>` |

Run once after `5-assemble`, from the directory holding the store:

```bash
cd <OUTPUT>/5-assemble
DATASET=<DATASET> <BIAHUB>/.venv/bin/python rename_channels.py
```

`nuclei` → `nuclei_prediction` bridges [biahub#288](https://github.com/czbiohub-sf/biahub/issues/288):
`biahub virtual-stain` names outputs verbatim from `target_channel`, but the
rest of biahub keys off the `_prediction` suffix.

**⚠ Ordering.** `4-track` runs *inside* the pipeline and reads the assembled
plate, so **track sees the pre-rename names**. The shipped templates avoid the
trap by putting the suffix directly in the VS `target_channel`
(`nuclei_prediction`/`membrane_prediction`), making the rename a no-op for
those channels.

**Stopgap:** PRs [#260](https://github.com/czbiohub-sf/biahub/pull/260) and
[#250](https://github.com/czbiohub-sf/biahub/pull/250) carry a `rename-channels`
CLI/subworkflow. Neither has landed — check before running by hand, and drop
the manual step once one merges.

---

## 3. Config values that must be checked per dataset

**`deskew.yml` — `pixel_size_um` and `scan_step_um`.** `pixel_size_um` is the
lateral pixel size of the *input* store (no longer read from the zarr);
`scan_step_um` comes from the acquisition — read
`/hpc/instruments/cm.mantis/<DATASET>/config.yaml`, do not assume 0.150.

**`reconstruct.yml` — `input_channel_names`.** Must name the brightfield
channel exactly as it appears in the deskewed store (`["BF - Oblique"]`
today). A mismatch fails at `compute_transfer_function`, after flat-field and
deskew have already burned hours.

**`virtual_stain.yml` — `ckpt_path` and model geometry.** The model/data
fields must agree with the checkpoint:

| family | checkpoint | `in_stack_depth` / `z_window_size` | `stem_kernel_size` |
|---|---|---|---|
| neuromast / zebrafish | `/hpc/projects/comp.micro/virtual_staining/models/fcmae-3d/fit_v2/pretrain_end2end/lightning_logs/finetune_VS_end2end_v1_test6_prefetch2_nopersistwork_restart_2/checkpoints/epoch=64-step=24960.ckpt` | 21 | `[7, 4, 4]` |
| A549 | `/hpc/projects/organelle_phenotyping/models/VSCyto3D-A549-infection-finetune/4gpu_bf16_bs16_to_ep7/checkpoints/epoch=7-step=832.ckpt` | 15 | `[5, 4, 4]` |

`data.init_args.z_window_size` **must equal**
`model.init_args.model_config.in_stack_depth` — a mismatch does not raise, the
model silently stalls. Top level may contain **only** `model`, `data`,
`ckpt_path` (plus optional `sliding_window_step`/`device`/
`output_ome_zarr_version`); old `viscy predict` blocks `trainer:` and
`return_predictions:` are rejected. `data.init_args.data_path` is injected
per-position — leave it out. `target_channel` carries the `_prediction` suffix
deliberately (§2).

**`track.yml` — schema.** `TrackingSettings` forbids extra keys. Current
fields: `output_mode`, `z_slicing`, `target_channel`, `fov`, `input_images`,
`segmentation_method`, `cellpose_config`, `tracking_config`. Old configs under
`/hpc/projects/tlg2_mantis/` still carry rejected fields — migrate:

```
mode                            -> output_mode
z_range + focus_config.z_window -> z_slicing.{method, window_size, focus_channel}
```

(`focus_config`'s `NA_det`/`lambda_ill`/`pixel_size` have no equivalent.)
`target_channel` and `cellpose_config.input_channel` must match **pre-rename**
names in the **assembled** plate (§2, §5).

**`z_slicing.focus_channel` resolves against `input_images`, not the store.**
`apply_focus_slicing` raises if the focus channel is not among the loaded
channels, even when it exists in the plate — a template declaring
`focus_channel: Phase3D` while loading only `nuclei_prediction` failed all
positions after every other step had succeeded. Both shipped templates use
`method: all` (falls back to the first loaded channel), the safe default.

**`concatenate.yml` — `concat_data_paths` stay as `placeholder`.** The
assemble subworkflow injects the real source paths via `--concat-data-paths`.
Three placeholder entries and three `channel_names` entries (deskew,
reconstruct, virtual-stain). Do not "fix" them to real paths.

---

## 4. Neuromast/zebrafish datasets are not tracked

Tracking is an A549 step; for neuromast/zebrafish the deliverable is
`5-assemble`, and track's parameters are tuned for A549 cells. But
`mantis-v2.nf` errors without `--track_config` and unconditionally wires
`track_wf` after assemble (making steps optional is
[biahub#306](https://github.com/czbiohub-sf/biahub/issues/306)). So:

- Pass `<BIAHUB>/nextflow/configs/zebrafish/track.yml` (a marked placeholder)
  so the run validates and starts.
- Say in the plan that `4-track` is a discarded by-product.
- Do not report tracking results or tune the track config for these datasets
  unless the user asks for neuromast tracking explicitly.

## 5. Track reads the assembled plate, not the intermediates

`track_wf` takes `5-assemble/<DATASET>.zarr` for *both* inputs. So any Z/Y/X
crop or `time_indices` subset in `concatenate.yml` is what tracking sees, and
tracking starts only after the whole plate assembles.

## 6. Assemble is a single job on one reserved node

`concatenate --cluster debug` iterates every position in-process, so
`5-assemble` is one large SLURM job — the single longest step.
`biahub concatenate --resume` (passed by the Nextflow task) makes a late kill
recoverable per (t, c) unit.

## 7. Shard geometry and `shards_ratio`

`shards_ratio` is a ratio over the **chunk** shape
(`shards = chunks * shards_ratio`, elementwise TCZYX), not an absolute shard
shape — changing T without recomputing ZYX silently changes the spatial extent.
Larger T shards mean a larger RAM request and a worse blast radius when one
tears; leave the T ratio alone without a measured reason. Channels cannot be
sharded — keep C at 1. Auto-sizing is
[iohub#458](https://github.com/czbiohub-sf/iohub/issues/458); most of this
section goes away when it lands.

## 8. Preemption is expected, not an error

Per-position work runs on the `preempted` partition; SLURM reclaims jobs
routinely (exit 143) and Nextflow resubmits (retry on exit 130–145,
`maxRetries = 5`). Dozens of retries in `trace.txt` is normal. Escalate only
when retries are exhausted or the same position fails *identically* every time
— the torn-shard signature, see `recovery.md`.

## 9. SLURM log files look inverted

`slurm_logs()` in `nextflow/modules/common.nf` deliberately crosses
`--output`/`--error` to undo Nextflow's fd swap: **the task's real output is
in `nextflow/slurm_output/<step>/*_<jobid>.out`** and the `.err` file is
empty. `.command.log` in the work dir is also empty. Read the `slurm_output`
files.

## 10. Do not edit the biahub checkout during a live run

Editing files in `$BIAHUB_PROJECT` changes Nextflow task hashes and
invalidates `-resume`. Get the branch right before launching, then leave it
alone. (Editing this skill is fine; it is not part of the pipeline.)

## 11. `uv sync` before launching — but check for out-of-band packages first

`uv sync --project <BIAHUB>` is the whole provisioning step (~1s no-op on a
healthy checkout). But it removes anything not in `uv.lock`, so check the
checkout you are about to use first:

```bash
uv sync --all-extras --dry-run 2>&1 | grep -E "^ *- "   # lists what would go
```

- **Empty** — sync. The expected state.
- **Unused leaf packages only** (nothing the pipeline imports) — sync, then
  confirm `.venv/bin/{biahub,viscy}` still work.
- **Anything load-bearing** (GPU/tracking stacks like `cupy-cuda12x`,
  `tracksdata`, `dask-cuda`, …) — **stop and raise it with the user.** Syncing
  such a checkout has broken a working environment before. Don't memorize a
  package list — lock membership moves; read what the dry-run actually says.

On Lustre a sync can fail *partway*
(`failed to remove directory ... __pycache__: Directory not empty`). **Stop**:
a half-completed uninstall can leave a package whose `dist-info` lost its
`RECORD`. Repair by deleting the corrupt package directory plus its
`dist-info`, then `uv pip install` the exact pins (`uv pip install` does not
prune).

Verify after syncing: `uv lock --check` and `<BIAHUB>/.venv/bin/biahub nf --help`.
Use explicit activation (`source .venv/bin/activate`), not `uv run` — `uv run`
re-syncs on every invocation, which could shift the environment under a live
pipeline and hides which env was used.

## 12. Per-invocation venv churn means a lockfile version mismatch

A healthy checkout shows no churn — `uv sync --dry-run` reports "Would make no
changes". If every invocation prints `Uninstalled 1 package / Installed 1
package`, a git-sourced dependency's declared version diverged from `uv.lock`
(seen with `viscy-transforms`; was [biahub#308](https://github.com/czbiohub-sf/biahub/issues/308)).
Diagnose and fix:

```bash
uv sync --dry-run -v 2>&1 | grep -i "does not match resolved version"
uv lock --refresh-package <package>
```

Do not work around it with `--no-sync`.

## 13. `--max_positions N` smoke tests leave a full-width plate

Only the per-position `run_*` tasks honour `--max_positions`; every `init_*`
step globs the whole input, so a one-position smoke test scaffolds the output
plate for *all* positions with data in only the first — correct and expected.
Do not treat such a store as a deliverable, and delete it before a real run so
`-resume` cannot reuse it.

(Related, for pipeline developers: a param passed as `--foo 1` on the command
line arrives as a String; coerce numeric params at the point of use and test
via the CLI — see the comments in `nextflow/modules/common.nf`.)

## 14. Tracking is not bitwise reproducible

Two runs of `biahub track` on the same input, same config, same commit do **not**
produce identical label arrays. Measured on one position of an assembled A549
plate (T=10, 1600×1370), two back-to-back runs differed in **1.06% of voxels**
while agreeing on everything summary-level: 13 objects, same max label id, same
per-frame cellpose counts (`mean=7.5, min=0, max=12`), same shape and dtype.

The sources are all inside tracking: cellpose GPU inference is not deterministic,
ultrack's watershed hierarchy runs `n_workers=16` with `random_seed='frame'`, and
the linking ILP can return any of several equally-optimal solutions.

Consequences worth knowing:

- **Do not diff label arrays to decide whether a change was harmful.** A 1–2%
  voxel difference is the noise floor. Compare object counts, per-frame counts,
  track lengths, and shape/dtype instead.
- **A rerun is not a reproduction.** If a tracking result matters, keep the
  output; you cannot regenerate the identical one later.
- **This applies to tracking only.** Every earlier step is deterministic given the
  same input and config — including `virtual-stain`, despite it also running GPU
  inference with test-time augmentation. Measured: the same position predicted on
  two different GPU nodes, one of them with a different CUDA runtime on
  `LD_LIBRARY_PATH`, produced **bitwise identical** output over three full
  timepoints (0 differing voxels in 1,024,338,432). So a virtual-stain difference
  between two runs is a real difference, not noise — worth investigating rather
  than shrugging at.
