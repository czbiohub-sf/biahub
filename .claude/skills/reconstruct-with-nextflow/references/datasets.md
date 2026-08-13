# Dataset conventions — where data lives and where it goes

## Raw acquisitions

Everything the mantis-v2 microscope writes lands in:

```
/hpc/instruments/cm.mantis/<DATASET_NAME>/
├── <DATASET_NAME>_1.ome.zarr    # the acquisition (index appended by Micro-Manager)
├── config.yaml                  # acquisition settings
├── logs/
├── pos_list.pos                 # stage positions
└── well_map.xlsx                # (multi-well acquisitions only)
```

**This tree is read-only for reconstruction. Never write into it.**

### Naming convention

`YYYY_MM_DD_<description>`, e.g. `2026_07_14_A549_MAP4_ZIKV`,
`2026_06_25_dynatrack_48hpf`, `2025_11_06_72hpf_cldnb-she-myo6b`.

The `.ome.zarr` inside is *usually* `<DATASET_NAME>_1.ome.zarr`, but not always.
Real deviations that exist today:

- Directory `2026_08_04_smart_fov_selection_test` holds
  `2026_08_04_test_fov_selection_{1..8}.ome.zarr`, plus `_epi_*`, `_prescan_10`,
  and `_fov_debug*` variants — the store stem differs from the directory *and*
  there are 30+ candidates.
- Directory `2026_07_24_dynatrack` holds both `2026_07_24_dynatrack_1.ome.zarr`
  and `2026_07_24_dynatrack_test_1.ome.zarr`, plus `checkin_t*_*.ome.zarr`
  snapshots written mid-acquisition (these are *not* the dataset).
- `2026_03_20_52hpf_cldnb-she-my6b` — typo'd gene name (`my6b` vs `myo6b`) in
  the directory itself.

**Whenever the store name does not equal `<DATASET_NAME>_1.ome.zarr`, tell the
user what you found and ask which store to use.** Do not pick by size or mtime
on your own.

`checkin_t<N>_<timestamp>.ome.zarr` stores are periodic snapshots from the
acquisition script, not reconstruction inputs. Ignore them.

## Output project roots

Reconstruction output goes to `/hpc/projects/...`, never next to the raw data.
Which root depends on the dataset family:

| family | cues in the name | output root | deliverable |
|---|---|---|---|
| zebrafish / neuromast / dynatrack | `dynatrack`, `hpf`, `dpf`, `neuromast`, `cldnb`, `she`, `myo6b`, `zebrafish` | `/hpc/projects/tlg2_mantis/<DATASET>` | `5-assemble` — **no tracking**, see `caveats.md` §4 |
| cell line / organelle / infection | `A549`, `HEK`, `iPSC`, organelle genes (`SEC61B`, `TOMM20`, `G3BP1`, `MAP4`, `CAAX`, `H2B`), viruses (`DENV`, `ZIKV`, `HSV1`) | `/hpc/projects/intracellular_dashboard/organelle_dynamics/<DATASET>` | `5-assemble` + `4-track` |
| instrument QC / calibration | `argolight`, `beads`, `psf`, `alignment`, `illumination`, `fluorescein`, `first_light`, `mantis_v2_*` | *no default* — ask. These are usually not reconstructed. | — |

Neighbouring roots that exist and are **not** the default target — do not use
them without being asked: `/hpc/projects/comp.micro/mantis`,
`/hpc/projects/comp.micro/zebrafish`, `/hpc/projects/organelle_phenotyping/datasets`,
`/hpc/projects/intracellular_dashboard/{viral-sensor,organelle_box}`.

Always state the chosen root in the plan and let the user correct it.

## Output directory layout

`--output` is the project root; each step writes a sibling subdirectory. The
layout is defined once in `nextflow/mantis-v2.nf` (`directory_layout()`):

```
<OUTPUT>/
├── 0-convert/<DATASET>.zarr        # only when the raw store has no HCS plate
├── 0-flatfield/<DATASET>.zarr
├── 1-deskew/<DATASET>.zarr
├── 2-reconstruct/<DATASET>.zarr
├── 3-virtual-stain/<DATASET>.zarr
├── 4-track/<DATASET>.zarr
├── 5-assemble/<DATASET>.zarr       # the deliverable
├── configs/                        # copied + edited per dataset
│   ├── flat_field.yml
│   ├── deskew.yml
│   ├── reconstruct.yml
│   ├── virtual_stain.yml           # older runs name this predict.yml
│   ├── concatenate.yml
│   └── track.yml
├── nextflow/
│   ├── work/                       # Nextflow work dir (default: <OUTPUT>/nextflow/work)
│   ├── slurm_output/<step>/%x_%j.{out,err}
│   ├── provenance.txt              # branch/commit/input per launch, appended
│   └── report.html  timeline.html  trace.txt  dag.html
├── .nextflow.log                   # the run record you read; rotates to .1, .2, ...
└── run_mantis_v2.sh                # the exact command used
```

The dataset stem comes from the *input* store name with `.ome.zarr`/`.zarr`
stripped (`dataset_name()` in `nextflow/modules/common.nf`). So if the input is
`0-convert/<DATASET>.zarr`, every step writes `<DATASET>.zarr`; if it is the raw
`<DATASET>_1.ome.zarr`, every step writes `<DATASET>_1.zarr`. Build the
`0-convert` plate under the clean name to keep the outputs clean.

Some older projects nest the step directories under a `1-preprocess/`
subdirectory (e.g. `2026_05_27_A549_SEC61B_TOMM20_G3BP1_ZIKV`). That is a legacy
layout — new runs put the step dirs directly under the project root.

## Reference runs worth copying from

| dataset | path | notes |
|---|---|---|
| `2026_07_14_A549_MAP4_ZIKV_rerun` | `/hpc/projects/intracellular_dashboard/organelle_dynamics/` | most recent A549 run; current `track.yml` schema; `HANDOFF_torn_shard_resume.md` documents the I/O recovery work |
| `2026_07_24_dynatrack` | `/hpc/projects/tlg2_mantis/` | most recent zebrafish run; has `build_plate.py` |
| `2026_06_25_dynatrack_48hpf` | `/hpc/projects/tlg2_mantis/` | canonical `build_plate_48hpf.py` with corrupt-metadata recovery |

Check the run actually succeeded (a populated `5-assemble/` and a clean tail in
`.nextflow.log`) before treating it as a reference.
