# Nextflow Pipelines

## Philosophy

Configs and launch scripts live **with the data**, not in the repo. Each experiment directory should contain:

```
experiment_dir/
  input.zarr
  configs/
    flat_field.yml
    deskew.yml
    reconstruct.yml
    ...
  run_pipeline.sh
  work/                  # nextflow work dir (gitignored, disposable)
```

The bash script pins all paths (input zarr, output dir, config dir, repo locations) and passes them to `nextflow run`. This keeps the pipeline definitions in the repo generic and reusable while the per-experiment choices are tracked alongside the data.

See the working example at:

```
/hpc/projects/intracellular_dashboard/refactor_biahub/phase2_dev/mantis_v2_dev/
  run_mantis_v2.sh
  configs/
    flat_field.yml
    deskew.yml
    reconstruct.yml
    predict.yml
    track.yml
    concatenate.yml
```

## Pipelines

### example-flatfield-deskew-reconstruct

Four-step pipeline: flat-field correction -> deskew -> reconstruct (compute transfer function + apply inverse TF) -> virtual stain (optional). Each step fans out per position within the plate zarr.

![Pipeline DAG](example-flatfield-deskew-reconstruct.png)

### mantis-v2-timelapse

Full mantis v2 pipeline: flat-field -> deskew -> reconstruct -> virtual stain + tracking (parallel) -> rename channels -> assembly (concatenate). Designed for timelapse plate-level data.

![Pipeline DAG](mantis-v2-timelapse.png)

**Entry points.** The pipeline supports `-entry` to restart from any stage, assuming prior outputs exist on disk:

| Entry | Assumes exists on disk | Runs |
|-------|------------------------|------|
| `full` (default) | input zarr | all stages |
| `from_deskew` | 0-flatfield | deskew → assembly |
| `from_reconstruct` | 0-flatfield, 1-deskew | reconstruct → assembly |
| `from_virtual_stain` | through 2-reconstruct | virtual stain → assembly |
| `from_tracking` | through 3-virtual-stain | tracking → assembly |
| `from_assembly` | through rename | assembly only |

Each entry point only validates the params it needs. For example, `from_assembly` only requires `--output_dir`, `--concatenate_config`, `--input_zarr`, and `--biahub_project`. QC runs only for the stages that actually execute.

Example — rerun just assembly + QC:

```bash
nextflow run "${PIPELINE}" \
    -c "${NF_CONFIG}" \
    -profile slurm \
    -entry from_assembly \
    --input_zarr         "${INPUT_ZARR}" \
    --output_dir         "${OUTPUT_DIR}" \
    --concatenate_config "${CONFIGS}/concatenate.yml" \
    --biahub_project     "${BIAHUB_PROJECT}" \
    --qc_config_dir      "${QC_CONFIGS}" \
    --qc_project         "${QC_PROJECT}" \
    --quarto_bin         "${QUARTO_BIN}" \
    --work_dir           "${WORK_DIR}" \
    -resume
```

### qc-standalone

Standalone QC pipeline for running `imaging-qc` against one or more pre-existing zarr stores without the full mantis pipeline. Supports arbitrary zarr+config pairs via a CSV manifest, two-pass metric computation with timepoint batching as distributed Slurm jobs, and a chained consolidation + report. See [README-qc-standalone.md](README-qc-standalone.md) for full usage and examples.

## Environment setup

On HPC, load the required modules first:

```bash
module load nextflow
module load uv
```

### biahub

Create the virtualenv (from the repo root):

```bash
cd /path/to/biahub
uv venv
uv sync
```

Nextflow processes use `uv run --project <path> biahub` to invoke the CLI, so the venv does not need to be activated at runtime. Pass `--biahub_project` to point to the repo root:

```bash
--biahub_project /path/to/biahub
```

Omit `--biahub_project` if `biahub` is already on `PATH` (e.g. in a container image).

### VisCy (virtual staining)

The virtual stain step requires a separate VisCy environment. We use an editable install from the [`dynacell-models`](https://github.com/mehta-lab/VisCy/pull/404) branch, which depends on iohub ≥0.3.3 (earlier versions crash when the plate zarr contains a `tables/` group at root).

```bash
# checkout the correct branch
cd /path/to/VisCy
git checkout dynacell-models

# create a thin wrapper project for the environment
mkdir -p /path/to/viscy-env
cat > /path/to/viscy-env/pyproject.toml << 'EOF'
[project]
name = "viscy-env"
version = "0.1.0"
description = "Standalone viscy environment for virtual staining inference"
requires-python = ">=3.12"

dependencies = [
    "viscy",
    "cytoland",
    "umap-learn",
]

[tool.uv.sources]
viscy = { path = "/path/to/VisCy", editable = true }
cytoland = { path = "/path/to/VisCy/applications/cytoland", editable = true }
viscy-data = { path = "/path/to/VisCy/packages/viscy-data", editable = true }
viscy-models = { path = "/path/to/VisCy/packages/viscy-models", editable = true }
viscy-transforms = { path = "/path/to/VisCy/packages/viscy-transforms", editable = true }
viscy-utils = { path = "/path/to/VisCy/packages/viscy-utils", editable = true }
EOF

cd /path/to/viscy-env
uv sync
```

Pass the env path to the pipeline:

```bash
--viscy_project /path/to/viscy-env
```

### Airtable Dataset Registry

The optional Airtable registry hook uses the existing VisCy Airtable CLI after tracking completes. It runs:

```bash
register /path/to/output_dir/5-assemble/<dataset>.zarr/*/*/*
write    /path/to/output_dir/5-assemble/<dataset>.zarr/*/*/*
```

against the VisCy monorepo's `applications/airtable` project. This requires:

```bash
export AIRTABLE_API_KEY=...
export AIRTABLE_BASE_ID=...
```

and a VisCy checkout with the Airtable app available:

```bash
--airtable_project /path/to/VisCy
--airtable_registry_after_tracking true
```

### imaging-qc-pipeline (QC stages + Quarto)

The QC processes in `modules/qc.nf` shell out to the `imaging-qc` CLI — they do **not** import Nextflow subworkflows or Python code from `imaging-qc-pipeline`. All orchestration (fan-out, chunking, merging) is defined here in biahub; `imaging-qc-pipeline` is a black-box CLI dependency. This means changes to the internal Nextflow or Python orchestration in `imaging-qc-pipeline` (DAG resolution, dispatch, subworkflows) do not affect this pipeline as long as the CLI contract is stable.

The QC stages require a separate `imaging-qc-pipeline` environment and Quarto (for interactive reports). **Pin to the `v0.3.1` tag** — later versions may change the CLI contract. Follow steps 1–3 of the [imaging-qc-pipeline getting started guide](https://github.com/czbiohub-sf/imaging-qc-pipeline/tree/v0.3.1#getting-started-hpc-path-brunoreef), checking out the tag before installing:

```bash
cd /path/to/imaging-qc-pipeline
git checkout v0.3.1
```

Then pass the project path and Quarto binary location to the pipeline:

```bash
--qc_project /path/to/imaging-qc-pipeline
--quarto_bin /path/to/quarto/bin
```

### Run location

Run `nextflow` from your experiment directory so that `.nextflow.log`, `.nextflow/`, and `work/` land there rather than in the repo.

## Usage

Write a bash launch script in your experiment directory. Example (`run_mantis_v2.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail
module load uv
module load nextflow

BIAHUB_PROJECT="/path/to/biahub"
VISCY_PROJECT="/path/to/viscy-env"
QC_PROJECT="/path/to/imaging-qc-pipeline"
QUARTO_BIN="/path/to/quarto/bin"
PIPELINE="${BIAHUB_PROJECT}/nextflow/mantis-v2-timelapse.nf"
NF_CONFIG="${BIAHUB_PROJECT}/nextflow/nextflow.config"

RUN_DIR="/path/to/experiment"
INPUT_ZARR="/path/to/input.zarr"
OUTPUT_DIR="${RUN_DIR}"
CONFIGS="${RUN_DIR}/configs"
QC_CONFIGS="${CONFIGS}/qc"
WORK_DIR="${RUN_DIR}/work"

nextflow run "${PIPELINE}" \
    -c "${NF_CONFIG}" \
    -profile slurm \
    --input_zarr         "${INPUT_ZARR}" \
    --output_dir         "${OUTPUT_DIR}" \
    --flat_field_config  "${CONFIGS}/flat_field.yml" \
    --deskew_config      "${CONFIGS}/deskew.yml" \
    --reconstruct_config "${CONFIGS}/reconstruct.yml" \
    --predict_config     "${CONFIGS}/predict.yml" \
    --track_config       "${CONFIGS}/track.yml" \
    --concatenate_config "${CONFIGS}/concatenate.yml" \
    --rename_suffix      "_recon" \
    --biahub_project     "${BIAHUB_PROJECT}" \
    --viscy_project      "${VISCY_PROJECT}" \
    --qc_config_dir      "${QC_CONFIGS}" \
    --qc_project         "${QC_PROJECT}" \
    --quarto_bin         "${QUARTO_BIN}" \
    --work_dir           "${WORK_DIR}" \
    -resume \
    "$@"
```

Use `-profile local` instead of `-profile slurm` for local execution.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `--input_zarr` | Path to input plate-level OME-Zarr store |
| `--output_dir` | Parent directory for all intermediate and final zarrs |
| `--flat_field_config` | YAML config for `FlatFieldCorrectionSettings` |
| `--deskew_config` | YAML config for `DeskewSettings` |
| `--reconstruct_config` | YAML config for waveorder `ReconstructionSettings` |
| `--predict_config` | YAML config for VisCy virtual stain prediction (optional; enables virtual stain step) |
| `--track_config` | YAML config for tracking (optional) |
| `--concatenate_config` | YAML config for concatenation (optional) |
| `--rename_suffix` | Suffix for channel renaming (optional) |
| `--biahub_project` | Path to biahub repo root for `uv run` (optional; see [Environment setup](#environment-setup)) |
| `--viscy_project` | Path to viscy-env wrapper project for `uv run` (optional; see [VisCy setup](#viscy-virtual-staining)) |
| `--airtable_project` | Path to the VisCy monorepo root containing `applications/airtable` (optional; required for Airtable registry integration) |
| `--max_positions` | Limit fan-out to first N positions (default: 0 = all positions) |
| `--work_dir` | Nextflow work directory for intermediate files (default: `work/` in current directory) |
| `--qc_config_dir` | Directory containing per-stage QC YAML configs (optional; enables QC stages) |
| `--qc_project` | Path to `imaging-qc-pipeline` repo root for `uv run` (optional; falls back to PyPI install) |
| `--qc_report_dir` | Directory for the final QC report (default: `<output_dir>/qc/report`) |
| `--qc_report_static` | Generate static PNG-only report instead of interactive Quarto/Plotly (default: `false`) |
| `--quarto_bin` | Path to directory containing the `quarto` binary (required for interactive reports on Slurm, where `~/.bashrc` is not sourced) |
| `--airtable_registry_after_tracking` | After tracking finishes, run VisCy Airtable `register` + `write` on the assembled zarr positions (default: `false`) |
| `--airtable_registry_dataset` | Optional Airtable dataset name override passed to the registration CLI (default: zarr stem) |
| `--airtable_registry_dry_run` | Log Airtable registry actions without writing to Airtable or zarr metadata (default: `false`) |

## Output

The dataset name is derived from the input zarr basename (e.g. `experiment.zarr` -> `experiment`).

```
output_dir/
  0-flatfield/
    <dataset_name>.zarr
  1-deskew/
    <dataset_name>.zarr
  2-reconstruct/
    transfer_function_<dataset_name>.zarr
    <dataset_name>.zarr
```

## Profiles

| Profile | Executor | Notes |
|---------|----------|-------|
| `local` | Local | Pass `--biahub_project` to use `uv run` (see [Environment setup](#environment-setup)) |
| `slurm` | SLURM | Submits to `cpu` queue; deskew uses `gpu` queue with `--gres=gpu:1` |

## Cleanup

After a run completes, remove Nextflow work directories, cache, and logs:

```bash
bash nextflow/cleanup.sh           # clean current directory
bash nextflow/cleanup.sh /path/to  # clean a specific directory
```

This does **not** remove your output zarrs — only Nextflow's internal files (`work/`, `.nextflow/`, logs, reports).

## CLI commands

The pipeline invokes `biahub nf` subcommands. Run `biahub nf --help` for the full list. Each command is a single unit of work; Nextflow handles distribution and scheduling.
