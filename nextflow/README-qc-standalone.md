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

Python owns all dispatch logic via `plan-stage` (JSON). Nextflow owns
fan-out, barriers, and retries. Three waves execute sequentially with
`.count()` barriers between them.

```
          manifest.csv
               │
    ┌──────────▼──────────┐
    │  plan_stage          │  (local) Python emits plan.json v2
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Wave 0: run_step   │  (Slurm) position-scoped, chunked
    └──────────┬──────────┘
               │ .count() barrier
    ┌──────────▼──────────┐
    │  finalize_wave(0)   │  (Slurm) merge chunk parquets
    └──────────┬──────────┘
               │ .count() barrier
    ┌──────────▼──────────┐
    │  Wave 1: run_step   │  (Slurm) dependent metrics
    └──────────┬──────────┘
               │ .count() barrier
    ┌──────────▼──────────┐
    │  Wave 2: run_step   │  (Slurm) store-scoped metrics
    └──────────┬──────────┘
               │ .count() barrier
    ┌──────────▼──────────┐
    │  finalize_stage     │  (Slurm) aggregate + gates + summary
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  report             │  (Slurm)
    └──────────┴──────────┘
```

### Local vs Slurm

| Step | Executor | Why |
|------|----------|-----|
| plan_stage | local | Reads zarr metadata, emits JSON |
| run_step (waves 0-2) | Slurm | CPU-intensive, parallelized |
| finalize_wave | Slurm | Reads/writes parquets |
| finalize_stage | Slurm | Aggregation + gate evaluation |
| Report generation | Slurm | Quarto rendering can be memory-heavy |

### Three-wave metric computation

`plan-stage` classifies metrics into waves by scope:

1. **Wave 0** (position): position-scoped metrics, optionally chunked by
   timepoint (`--qc_chunk_size`). Each (position, chunk) is a Slurm job.
2. **finalize_wave(0)** merges chunk parquets into per-position results.
3. **Wave 1** (dependent): metrics that read wave-0 outputs (e.g. `bleach_rate`
   depends on `intensity_stats`).
4. **Wave 2** (store): store-scoped metrics (one job per zarr, no position
   fan-out).
5. **finalize_stage** aggregates all results, evaluates gates, writes summary.

Empty waves are no-ops — `.count()` emits 0, and the barrier propagates
without spawning processes.
