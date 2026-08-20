# Pipeline-specific caveats

Read this before writing configs or launching. Each item has bitten a real run.

---

## 1. Neuromast / zebrafish acquisitions have no HCS plate

The pipeline fans out over plate positions, so its input must be an HCS plate.
A549/cell-line acquisitions already are one (root `zarr.json` has
`attributes.ome.plate`). Neuromast/zebrafish/dynatrack acquisitions are flat
`bioformats2raw.layout: 3` stores with positions named `{R}Pos{C}` at the root.

iohub enumerates flat stores (`open_ome_zarr(..., layout="bf2raw")`), so
`list_positions` alone works — but every `init_*` step globs
`-i <store>/*/*/*`, track configs use `fov: "*/*/*"`,
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

Run once after assemble, from the directory holding the store:

```bash
cd <OUTPUT>/<N>-assemble        # 4-assemble on a standard run
DATASET=<DATASET> <BIAHUB>/.venv/bin/python rename_channels.py
```

`nuclei` → `nuclei_prediction` bridges [biahub#288](https://github.com/czbiohub-sf/biahub/issues/288):
`biahub virtual-stain` names outputs verbatim from `target_channel`, but the
rest of biahub keys off the `_prediction` suffix.

**⚠ Ordering.** tracking runs *inside* the pipeline and reads the assembled
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

Tracking is an A549 step; for neuromast/zebrafish the deliverable is the
assembled store, and track's parameters (cellpose `diameter`, `min_area`/
`max_area`, linking `max_distance`) are tuned for A549 cells and do not
transfer. So simply do not run it:

- Omit `--track_config` — drop `track` (and `qc_track`) from `STEPS` in the run
  script. A step runs only if its config is passed
  ([biahub#306](https://github.com/czbiohub-sf/biahub/issues/306), fixed).
- There is no tracking by-product to explain away any more, and the
  `zebrafish/track.yml` placeholder that existed only to satisfy the old
  hard requirement is deleted.
- Do not report tracking results or tune the track config for these datasets
  unless the user asks for neuromast tracking explicitly.
- QC still runs: a neuromast run does image QC of the assembled store
  (`--qc_config`). Only the tracking tab is absent.

## 5. Track reads the assembled plate, not the intermediates

`track_wf` takes the assembled `<DATASET>.zarr` for *both* inputs. So any Z/Y/X
crop or `time_indices` subset in `concatenate.yml` is what tracking sees, and
tracking starts only after the whole plate assembles.

## 6. Assemble is a single job on one reserved node

`concatenate --cluster debug` iterates every position in-process, so
Assemble is one large SLURM job — the single longest step.
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

**A T ratio above 1 needs [iohub#460](https://github.com/czbiohub-sf/iohub/pull/460),
which shipped in iohub 0.3.11** — the floor `pyproject.toml` already requires, so a
synced checkout has it.

Before that fix, `concatenate` aborted with a message that points nowhere near the
cause — `numpy ... Unable to allocate 651. TiB for an array with shape
(1048576, 86, 1664, 1193)`, whose first axis is not T, not a shard, not
anything. It is **not** evidence that the shard buffer is too large, and
unsetting `shards_ratio` "fixes" it for the wrong reason. What happens: a
shard-aligned batch writes a whole shard's worth of timepoints in one call,
iohub drops the blank ones (`Skipping t=1, c=0 due to all zeros or nans` in the
step's `.command.out`), and the surviving indices are left with a *gap* — which
a sharded write cannot express. Confirm it by grepping `.command.err` for
`DiscontiguousArrayError`; the array it prints is the diff of the surviving
indices (`[3 1]` for `[0, 3, 4]`, i.e. t=1 and t=2 were blank). If you see this
signature, check the installed iohub version rather than the shard geometry:

```bash
<BIAHUB>/.venv/bin/python -c "import iohub; print(iohub.__version__)"   # need >= 0.3.11
```

There is no git pin to check: `pyproject.toml` requires `iohub>=0.3.11` and
`uv.lock` resolves it from PyPI, so `uv sync` is enough. An out-of-band install
is the only way to end up below the floor — see §11.

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
alone. (Editing this skill is *mostly* fine — but the notification path now runs
through `biahub/utils/notify.py` and `nextflow/modules/notify.nf`, which ARE part
of the pipeline, so fixing a notification bug mid-run is a pipeline edit. The
notify task hashes are the only ones affected, and re-running six sub-second local
tasks costs nothing.)

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
- **Tracking's instability is structural; virtual-stain's is not.** Both run GPU
  inference, so the distinction matters:

  | | tracking | virtual-stain |
  |---|---|---|
  | repeat run, same environment | 1.06% of voxels differ | bitwise identical (0 of 1,024,338,432) |
  | different GPU node / different CUDA libraries installed | — | 99.87% of voxels differ, but `corr = 1.000000`, median abs diff `5.6e-06`, max abs diff `9.3e-04` on a ~39 range |

  Virtual-stain's differences are float32 last-bit rounding from GPU kernel
  selection — identical mean, std, min and max to four decimals. Tracking's are
  *structural*: different label ids and boundaries, from cellpose
  non-determinism, ultrack's `n_workers=16` watershed with `random_seed='frame'`,
  and ILP solution ties.

  So **judge virtual-stain on values, not voxel counts.** A bare "99% of voxels
  differ" means nothing here; `corr` and max abs diff are the measures that do. A
  real virtual-stain regression moves the distribution, not the last bit. Deskew,
  reconstruct and assemble are fully deterministic.
