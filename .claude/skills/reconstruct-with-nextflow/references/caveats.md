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

The assembled plate inherits raw instrument channel labels, which are camera/filter
strings rather than biological names. Downstream analysis expects the cleaned
names, so this runs **once, after `5-assemble` completes**, in place on the
assembled store.

Canonical script:
`/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_05_27_A549_SEC61B_TOMM20_G3BP1_ZIKV/1-preprocess/5-assemble/rename_channels.py`
(shipped here as `templates/rename_channels.py`).

Mapping it applies:

| from | to |
|---|---|
| `BF - Oblique` | `BF` |
| `mCherry EX561 EM600-37` | `raw mCherry EX561 EM600-37` |
| `GFP EX488 EM525-45` | `raw GFP EX488 EM525-45` |

The `raw ` prefix distinguishes the acquired fluorescence channels from the
virtual-stain predictions (`nuclei_prediction`, `membrane_prediction`), which are
left alone.

**The channel list is dataset-dependent — print the actual channel names first
and adapt the mapping.** A dataset with `Cy5 EX639 EM698-70` needs that entry
added. The script takes the store stem via the `DATASET` env var and appends
`.zarr`, so run it from the directory containing the store:

```bash
cd <OUTPUT>/5-assemble
DATASET=<DATASET> uv run --project <BIAHUB> python rename_channels.py
```

Renaming is in-place and not idempotent in the prefix direction — running it
twice yields `raw raw mCherry ...`. Check the current names before re-running.

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
| neuromast / zebrafish | `/hpc/projects/comp.micro/virtual_staining/models/fcmae-3d/fit_v2/pretrain_end2end/lightning_logs/finetune_VS_end2end_v1_test6_prefetch2_nopersistwork_restart_2/checkpoints/epoch=64-step=24960.ckpt` | 21 | `[7, 4, 4]` | `nuclei`, `membrane` |
| A549 | `/hpc/projects/organelle_phenotyping/models/VSCyto3D-A549-infection-finetune/4gpu_bf16_bs16_to_ep7/checkpoints/epoch=7-step=832.ckpt` | 15 | `[5, 4, 4]` | `nuclei_prediction`, `membrane_prediction` |

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
`3-virtual-stain`. If you rename channels (§2) before tracking, the names must
agree.

**`concatenate.yml` — `concat_data_paths` stay as `placeholder`.** The assemble
subworkflow injects the three real source paths via `--concat-data-paths`. Three
placeholder entries and three `channel_names` entries, one per source (deskew,
reconstruct, virtual-stain). Do not "fix" them to real paths.

---

## 4. Track reads the assembled plate, not the intermediates

As wired today, `track_wf` takes `5-assemble/<DATASET>.zarr` for *both* its
inputs. Two consequences:

- Any Z/Y/X crop or `time_indices` subset in `concatenate.yml` is what tracking
  sees.
- Tracking runs strictly *after* assemble, not in parallel with it. The whole
  plate must assemble before any track task starts.

## 5. Assemble is a single job on one reserved node

`concatenate --cluster debug` iterates every position in-process, so `5-assemble`
is one large SLURM job rather than a fan-out — 16 TiB of I/O and tens of
thousands of write units for a typical plate. A late kill used to discard all of
it; `biahub concatenate --resume` (which the Nextflow task passes) makes it
recoverable. Give it room: it is the single longest step.

## 6. Shard geometry and `shards_ratio`

`shards_ratio` is a ratio over the **chunk** shape, not an absolute shard shape
(`shards = chunks * shards_ratio`, elementwise TCZYX). Changing the T entry
without recomputing ZYX silently changes the spatial extent too. For an assemble
output `[T, 6, 86, 1664, 1193]` with chunks `(1,1,16,256,256)`, the ZYX entries
that reproduce today's extent are `ceil(86/16)=6, ceil(1664/256)=7,
ceil(1193/256)=5`.

Larger T shards mean fewer, bigger files but a much larger RAM request and a much
worse blast radius when one tears — a shard spanning 10 timepoints loses all 10.
Leave the T ratio alone unless you have a measured reason.

## 7. Preemption is expected, not an error

Per-position work runs on the `preempted` partition. SLURM reclaims jobs there
routinely; they exit 143 and Nextflow resubmits (`errorStrategy` retries exit
130–145, `maxRetries = 5`). A run with dozens of retries in `trace.txt` is normal.
Only escalate when retries are exhausted, or when the same position fails
identically every time — that is the torn-shard signature, see `recovery.md`.

## 8. SLURM log files look inverted

`slurm_logs()` in `nextflow/modules/common.nf` deliberately crosses
`--output`/`--error` to undo Nextflow's fd swap. The net effect: **the task's
real output is in `nextflow/slurm_output/<step>/*_<jobid>.out` and the `.err`
file is empty.** Also, `.command.log` in the Nextflow work dir is empty, because
the `clusterOptions` override wins over Nextflow's own `-o`. Read the
`slurm_output` files, not the work dir logs.

## 9. Do not edit the biahub checkout during a live run

Editing files in `$BIAHUB_PROJECT` changes Nextflow task hashes and invalidates
`-resume`, so a restart recomputes everything. Sync the branch and run
`uv lock && uv sync` *before* launching, then leave it alone. Each per-position
task is a `uv run --project`, so an unsynced lockfile also means every task
resolves dependencies concurrently at startup.
