# CLAUDE.md

## Project Overview

**biahub** is a microscopy image analysis pipeline for intracellular dynamics research. The main development focus is the `scripts/dynacell/` package — a preprocessing pipeline for organelle dynamics datasets (label-free + light-sheet OME-Zarr data).

## Environment

- **HPC cluster** (SLURM-based). Jobs are submitted via `submitit`.
- **Conda env**: `biahub` — activate with `conda activate biahub`
- **Python path**: Always set `PYTHONPATH=/hpc/mydata/taylla.theodoro/repo/biahub_dev/scripts:$PYTHONPATH` when running scripts
- **Data root**: `/hpc/projects/intracellular_dashboard/organelle_dynamics/`
- **Python**: 3.11

## Key Packages

- `iohub` — OME-Zarr I/O (`open_ome_zarr`, `create_empty_plate`)
- `waveorder` — `focus_from_transverse_band` for z-focus detection
- `submitit` — SLURM job submission
- `numpy`, `pandas`, `matplotlib`, `scipy`

## Repository Structure

```
biahub/                  # Main installable package
scripts/
  dynacell/              # Preprocessing pipeline package
    __init__.py          # Public exports
    geometry.py          # Overlap mask, inscribed bbox, circular mask
    stage1.py            # Per-FOV metadata: z_focus, bbox, drop list, QC metrics
    stage2.py            # Crop FOVs into unified plate
    run.py               # Pipeline orchestration (run_all_fovs, run_from_config)
    qc.py                # QC compute functions (beads reg, laplacian, entropy, etc.)
    qc_report.py         # HTML report generator (standard + annotated)
    plotting.py          # All plot functions (per-FOV and all-FOVs summary)
    init_annotations.py  # Pre-run annotations CSV initializer
    manual_crop.py       # Manual crop utilities
  configs/               # YAML dataset configs
  debug_*.py             # Debug/development scripts (thin wrappers)
```

## Pipeline Architecture

### Two-stage processing
- **Stage 1**: Per-FOV metadata computation
  - Phase 1: Core metadata (z_focus, bbox, overlap mask, drop list) — one SLURM job per FOV
  - Phase 2: Per-(FOV, metric) QC jobs (laplacian, entropy, hf_ratio, frc, etc.)
- **Stage 2**: Crop all qualified FOVs into a unified output plate

### Key design decisions
- **z_focus outlier detection**: Two modes — `z_window` (absolute band: mean ± window) or `n_std` (statistical: mean ± n*sigma)
- **Per-metric configurable thresholds**: `qc_thresholds` dict from YAML flows through the pipeline
- **FOV qualification**: FOVs with drops > `max_drops` (default 5) are disqualified from stage 2

### Config-driven runs
Datasets are configured via YAML files in `scripts/configs/`. Run with:
```python
from dynacell.run import run_from_config
run_from_config("configs/<dataset>.yaml")
```

## Workflow Preferences

- **Commit after every logical unit of work** — don't wait until the end
- **Run pipelines in background** when the user asks to run them
- **Cancel previous SLURM jobs** before re-running (use `scancel`)
- When debugging pipeline issues, **check the actual data** (zarr contents, CSV outputs, per-FOV plots) rather than guessing
- **Compare with reference datasets** (e.g., `2024_11_07_A549_SEC61_DENV`) when investigating anomalies
- Prefer **editing existing files** over creating new ones
- Keep `debug_dynacell_preprocessing.py` as a **thin wrapper** — all logic belongs in `dynacell/run.py`

## Common Tasks

### Running the pipeline
```bash
PYTHONPATH=.../scripts:$PYTHONPATH python -c "from dynacell.run import run_from_config; run_from_config('configs/<dataset>.yaml')"
```

### Cancelling SLURM jobs
```bash
scancel -u taylla.theodoro  # Cancel all user jobs
```

### Regenerating the report (without re-running pipeline)
```python
from dynacell.qc_report import generate_dataset_report
generate_dataset_report(run_dir, overlay_channels=[...])
```

### Checking run outputs
- Run logs: `<run_dir>/run_log.yaml`
- Global summary: `<run_dir>/global_summary.csv`
- Per-FOV plots: `<run_dir>/per_fov_analysis/<FOV_NAME>/`
- Drop lists: `<run_dir>/per_fov_analysis/<FOV_NAME>/drop_list.csv`
- Report: `<run_dir>/dataset_report.html`
