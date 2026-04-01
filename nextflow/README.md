# Nextflow Pipelines

## example-flatfield-deskew-reconstruct

Three-step pipeline: flat-field correction → deskew → reconstruct (compute transfer function + apply inverse TF). Each step fans out per position within the plate zarr.

![Pipeline DAG](example-flatfield-deskew-reconstruct.png)

### Usage

```bash
nextflow run nextflow/example-flatfield-deskew-reconstruct.nf \
    -profile local \
    --venv_path /path/to/biahub/.venv \
    --input_zarr /path/to/input.zarr \
    --output_dir /path/to/output \
    --flat_field_config /path/to/flat_field.yml \
    --deskew_config /path/to/deskew.yml \
    --reconstruct_config /path/to/reconstruct.yml
```

Use `-profile slurm` instead of `-profile local` to submit jobs to SLURM.

Add `-resume` to restart from where a previous run left off.

### Environment setup

Nextflow processes run in a clean shell, so the `biahub` CLI must be made available explicitly. Pass `--venv_path` to activate a virtualenv before each process:

```bash
--venv_path /path/to/biahub/.venv
```

Omit `--venv_path` if `biahub` is already on `PATH` (e.g. in a container image).

If PR #202 (`biahub-uv`) is merged, an alternative is to replace `beforeScript` activation with `uv run` invocations in the process scripts, letting uv resolve the environment from the project directory without explicit venv activation.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--input_zarr` | Path to input plate-level OME-Zarr store |
| `--output_dir` | Parent directory for all intermediate and final zarrs |
| `--flat_field_config` | YAML config for `FlatFieldCorrectionSettings` |
| `--deskew_config` | YAML config for `DeskewSettings` |
| `--reconstruct_config` | YAML config for waveorder `ReconstructionSettings` |
| `--num_processes` | Intra-position parallelism for reconstruction (default: 1) |
| `--venv_path` | Path to virtualenv containing `biahub` (optional; see [Environment setup](#environment-setup)) |

### Output

The dataset name is derived from the input zarr basename (e.g. `experiment.zarr` → `experiment`).

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

### Profiles

| Profile | Executor | Notes |
|---------|----------|-------|
| `local` | Local | Pass `--venv_path` to activate your venv (see [Environment setup](#environment-setup)) |
| `slurm` | SLURM | Submits to `cpu` queue; deskew uses `gpu` queue with `--gres=gpu:1` |

### Nextflow reports

After a run completes, reports are written to `nextflow/output/`:
- `dag.html` — pipeline DAG
- `report.html` — execution report
- `trace.txt` — per-task trace
- `timeline.html` — timeline visualization

### Cleanup

After a run completes, remove Nextflow work directories, cache, and logs:

```bash
bash nextflow/cleanup.sh
```

This does **not** remove your output zarrs — only Nextflow's internal files (`work/`, `.nextflow/`, logs, reports).

### CLI commands

The pipeline invokes these `biahub nf` subcommands:

```
biahub nf list-positions -i <plate.zarr>
biahub nf init-flat-field -i <input.zarr> -o <output.zarr> -c <config.yml>
biahub nf run-flat-field -i <input.zarr> -o <output.zarr> -p <position> -c <config.yml>
biahub nf init-deskew -i <input.zarr> -o <output.zarr> -c <config.yml>
biahub nf run-deskew -i <input.zarr> -o <output.zarr> -p <position> -c <config.yml>
biahub nf init-reconstruct -i <input.zarr> -o <output.zarr> -t <tf.zarr> -c <config.yml>
biahub nf run-apply-inv-tf -i <input.zarr> -o <output.zarr> -t <tf.zarr> -p <position> -c <config.yml>
```

Each command is a single unit of work (no SLURM/submitit). Nextflow handles distribution and scheduling.
