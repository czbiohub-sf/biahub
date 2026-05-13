# Standalone QC Pipeline

Run `imaging-qc` against one or more pre-existing zarr stores without running
the full mantis processing pipeline. Useful for:

- Re-running QC on already-processed data
- Running QC on a subset of stages
- Running QC on zarrs produced outside the mantis pipeline

## Quick start

```bash
# From your experiment directory
nextflow run /path/to/biahub/nextflow/qc-standalone.nf \
    -c /path/to/biahub/nextflow/nextflow.config \
    -profile slurm \
    --stages_manifest  stages.csv \
    --output_dir       /path/to/experiment \
    --qc_project       /path/to/imaging-qc-pipeline \
    --biahub_project   /path/to/biahub \
    --quarto_bin       /path/to/quarto/bin \
    -resume
```

## Stages manifest

A CSV file with header that declares which zarr stores to QC and their configs.
Each row is one QC stage. Stages run in parallel.

| Column | Required | Description |
|--------|----------|-------------|
| `zarr_path` | yes | Absolute path to the zarr store |
| `config_path` | yes | Absolute path to the `imaging-qc` YAML config for this stage |
| `stage_name` | yes | Label for this stage (used in summary output and file naming) |
| `assembly` | no | `true` if this is the assembly zarr that receives consolidated QC parquets from the other (step) zarrs. At most one row should be `true`. Default: `false` |

### Example: `stages.csv`

```csv
zarr_path,config_path,stage_name,assembly
/hpc/projects/.../2-reconstruct/dataset.zarr,/hpc/projects/.../configs/qc/qc_stage3_post_reconstruct.yaml,reconstruct,false
/hpc/projects/.../5-assemble/dataset.zarr,/hpc/projects/.../configs/qc/qc_stage5_post_assembly.yaml,assembly,true
```

### Example: single zarr, no assembly

When QC-ing a standalone zarr (not part of a multi-step pipeline), omit the
`assembly` column or set all rows to `false`. Consolidation is skipped.

```csv
zarr_path,config_path,stage_name,assembly
/hpc/projects/.../my_data.zarr,/path/to/qc_config.yaml,my_stage,false
```

## Launch script

### Example: `run_qc.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
module load uv
module load nextflow

BIAHUB_PROJECT="/home/aliu/repos/biahub"
QC_PROJECT="/home/aliu/repos/imaging-qc-pipeline"
QUARTO_BIN="/home/aliu/opt/quarto-1.7.23/bin"

DEV_DIR="/hpc/projects/intracellular_dashboard/refactor_biahub/phase2_dev/mantis_v2_dev_qc"
PIPELINE="${BIAHUB_PROJECT}/nextflow/qc-standalone.nf"
NF_CONFIG="${BIAHUB_PROJECT}/nextflow/nextflow.config"
WORK_DIR="${DEV_DIR}/work-qc"

nextflow run "${PIPELINE}" \
    -c "${NF_CONFIG}" \
    -profile slurm \
    --stages_manifest  "${DEV_DIR}/configs/qc/stages.csv" \
    --output_dir       "${DEV_DIR}" \
    --biahub_project   "${BIAHUB_PROJECT}" \
    --qc_project       "${QC_PROJECT}" \
    --quarto_bin       "${QUARTO_BIN}" \
    --work_dir         "${WORK_DIR}" \
    -resume \
    "$@"
```

### Example: dev run with limited positions

```bash
nextflow run "${PIPELINE}" \
    -c "${NF_CONFIG}" \
    -profile slurm \
    --stages_manifest  "${DEV_DIR}/configs/qc/stages.csv" \
    --output_dir       "${DEV_DIR}" \
    --biahub_project   "${BIAHUB_PROJECT}" \
    --qc_project       "${QC_PROJECT}" \
    --quarto_bin       "${QUARTO_BIN}" \
    --positions        "B/3/000000,B/3/000001" \
    --qc_report_static \
    -resume
```

## QC config linking

Each QC stage needs a YAML config for `imaging-qc`. These live alongside
your data, not in the biahub repo. A typical layout:

```
experiment_dir/
  configs/
    qc/
      stages.csv                          # manifest for qc-standalone
      qc_stage3_post_reconstruct.yaml     # imaging-qc config per stage
      qc_stage5_post_assembly.yaml
  2-reconstruct/
    dataset.zarr
  5-assemble/
    dataset.zarr
  run_qc.sh
  work-qc/                               # nextflow work dir
```

The QC YAML configs define metric groups, thresholds, and gating rules.
See the [imaging-qc-pipeline docs](https://github.com/czbiohub-sf/imaging-qc-pipeline)
for config format.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--stages_manifest` | CSV manifest (see above) | *required* |
| `--output_dir` | Parent directory for report output | *required* |
| `--positions` | Comma-separated position keys (e.g. `B/3/000000,B/3/000001`) | auto-discovered from first zarr |
| `--max_positions` | Limit to first N positions (0 = all) | `0` |
| `--qc_chunk_size` | Timepoints per distributed chunk job | `10` |
| `--qc_project` | Path to `imaging-qc-pipeline` for `uv run` | falls back to PyPI |
| `--biahub_project` | Path to biahub repo root for `uv run` | assumes `biahub` on PATH |
| `--quarto_bin` | Directory containing `quarto` binary | not set |
| `--qc_report_static` | Generate static PNG report (no Quarto) | `false` |
| `--qc_report_dir` | Report output directory | `<output_dir>/qc/report` |

## Architecture

```
          manifest.csv
               │
    ┌──────────▼──────────┐
    │  discover_positions │  (local)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  init_qc_fanout     │  (local, per stage)
    │  estimate_resources  │  (local, per metric group)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  PASS 1: chunked    │  (Slurm) timepoint batches as
    │  position metrics   │  distributed jobs
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  merge_qc_metrics   │  (Slurm) consolidate chunk parquets
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  PASS 2: dependent  │  (Slurm) metrics that read pass-1
    │  temporal metrics   │  results (e.g. bleach_rate)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  merge + gate       │  (Slurm) per stage
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  consolidate        │  (local) copy parquets to assembly
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  report             │  (Slurm)
    └──────────┴──────────┘
```

### Local vs Slurm

| Step | Executor | Why |
|------|----------|-----|
| Position discovery | local | Reads zarr metadata only |
| Fan-out + resource estimation | local | Lightweight CLI calls |
| Metric computation (pass 1 + 2) | Slurm | CPU-intensive, parallelized |
| Metric merge | Slurm | Reads/writes parquets |
| Gate evaluation | Slurm | Reads merged parquets |
| Consolidation | local | File copy only |
| Report generation | Slurm | Quarto rendering can be memory-heavy |

### Two-pass metric computation

Some metrics depend on the results of other metrics. For example,
`bleach_rate` requires `intensity_stats` to be computed and merged first.
The pipeline handles this automatically:

1. **Pass 1** runs all position-scoped, time-chunked metrics as distributed
   Slurm jobs (controlled by `--qc_chunk_size`).
2. `merge_qc_metrics` consolidates chunk parquets into per-position results.
3. **Pass 2** runs dependent/temporal metrics that read pass-1 outputs.

The fan-out logic (`init_qc_fanout`) determines which metrics go to which
pass based on whether they have time-chunk boundaries.
