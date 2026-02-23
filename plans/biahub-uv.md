# Plan: Convert biahub to uv Workspaces with analysis-templates Subpackage

## Session Summary (for resuming)

**Plan file**: `biahub/plans/biahub-uv.md`

**Key paths**:
- biahub repo: `/hpc/mydata/$USER/code/biahub`
- Widefield dataset (Hummingbird): `/hpc/projects/intracellular_dashboard/virtual_stain_ft_infected/2026_01_29_A549_H2B_CAAX_DAPI_DENV_ZIKV/`
  - 0-convert/convert.zarr: Raw converted data (134 FOVs, channels: DAPI, TXR, BF, shape: 9×3×126×2048×2048)
  - 1-reconstruct/: Reconstructed phase + deconvolved fluorescence
  - 2-concatenate/: Concatenated output
  - 0-convert_zarrv3/, 1-reconstruct_zarrv3/, 2-register_zarrv3/: Zarr v3 variants
- Light-sheet + label-free dataset (Mantis): `/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/`
  - 0-convert, 1-preprocess, 2-assemble, 3-visualization, 4-phenotyping

**Environment for test data creation**: `biautils`
- Activate with module load anaconda; module load comp_micro; conda activate biautils.

**Related PRs**:
- PR #200 (feature/merge-mantis-analysis-template): Do NOT merge, create new PR instead
- VisCy issue #353: Reference for uv workspaces pattern
- infected_vs PR #1 (located at /hpc/mydata/$USER/code/infected_vs): Example of attempt at processing data from another microscope, called hummingbird.

---

## Overview

Convert biahub to a uv workspace monorepo with two subpackages:
1. **biahub**: Core processing modules (unchanged package name, Click CLI)
2. **analysis-templates**: Hydra-based analysis workflows with microscope-specific configs

## Key Decisions

- **Package name**: Keep `biahub` unchanged; new package is `analysis-templates`
- **CLI strategy**: Click CLI stays with biahub; Hydra CLI for pipeline orchestration
- **Microscope naming**: Use modality-based names (e.g., `widefield+label-free`, `light-sheet+label-free`) instead of hardware names (e.g., hummingbird, mantis)
- **Pipeline scope**: Start with 4-stage widefield+label-free workflow (convert → reconstruct → register → virtual stain)
- **Data layout**: Must match existing bash-based template output structure
- **PR #200**: Do NOT merge; create new PR with the workspace approach

## Reference Datasets (Ground Truth)

**Widefield (Hummingbird)**: `/hpc/projects/intracellular_dashboard/virtual_stain_ft_infected/2026_01_29_A549_H2B_CAAX_DAPI_DENV_ZIKV/`
- 0-convert/ (and 0-convert_zarrv3/)
- 1-reconstruct/ (and 1-reconstruct_zarrv3/)
- 2-concatenate/
- 2-register_zarrv3/

**Light-sheet + label-free (Mantis)**: `/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/`
- 0-convert, 1-preprocess, 2-assemble, 3-visualization, 4-phenotyping

## Prerequisite: Create Test Dataset

Use a subset of the reference dataset above for fast iteration:

```bash
conda activate biautils

python << 'EOF'
from iohub import open_ome_zarr
from pathlib import Path
import numpy as np

# Source: full dataset with 134 FOVs
src_path = Path("/hpc/projects/intracellular_dashboard/virtual_stain_ft_infected/2026_01_29_A549_H2B_CAAX_DAPI_DENV_ZIKV/0-convert/convert.zarr")
dst_path = Path("/hpc/projects/intracellular_dashboard/refactor_biahub/test_widefield.zarr")

dst_path.parent.mkdir(parents=True, exist_ok=True)

src = open_ome_zarr(src_path, mode="r")
positions = list(src.positions())

# Copy first 2 FOVs for a minimal test dataset
channel_names = list(positions[0][1].channel_names)
dst = open_ome_zarr(dst_path, layout="hcs", mode="w", channel_names=channel_names)

for name, pos in positions[:2]:
    parts = name.split("/")
    dst_pos = dst.create_position(*parts)
    dst_pos["0"] = np.array(pos["0"])
    print(f"  Copied {name}: shape={pos['0'].shape}")

print(f"\nCreated test dataset at {dst_path}")
print(f"  Channels: {channel_names}")

src.close()
EOF
```

**Test data location**: `/hpc/projects/intracellular_dashboard/refactor_biahub/test_widefield.zarr`
- First 2 FOVs from 2026_01_29 convert.zarr (channels: DAPI, TXR, BF; shape: 9×3×126×2048×2048 each)

## Proposed Directory Structure

```
biahub/                              # Repository root
├── pyproject.toml                   # Workspace root configuration
├── uv.lock                          # Generated lockfile
│
├── packages/
│   ├── biahub/                      # Core processing (SAME package name)
│   │   ├── pyproject.toml
│   │   └── src/biahub/              # NO rename - imports stay as `biahub.X`
│   │       ├── __init__.py
│   │       ├── cli/                 # Click CLI (biahub command)
│   │       │   ├── main.py
│   │       │   └── ...
│   │       ├── deskew.py
│   │       ├── register.py
│   │       ├── reconstruct.py
│   │       ├── settings.py
│   │       └── ...
│   │
│   └── analysis-templates/          # Hydra-based workflows
│       ├── pyproject.toml
│       └── src/analysis_templates/
│           ├── __init__.py
│           ├── cli.py               # Hydra entry point
│           ├── pipeline.py          # Orchestration
│           ├── steps/               # Pipeline step wrappers
│           │   ├── __init__.py
│           │   ├── convert.py
│           │   ├── reconstruct.py
│           │   ├── register.py
│           │   └── virtual_stain.py
│           └── conf/                # Hydra configs
│               ├── config.yaml      # Main config
│               ├── microscope/      # Modality-based presets
│               │   ├── widefield+label-free.yaml    # Hummingbird
│               │   └── light-sheet+label-free.yaml  # Mantis
│               ├── step/            # Per-step configs
│               │   ├── reconstruct.yaml
│               │   ├── register.yaml
│               │   └── virtual_stain.yaml
│               └── slurm/           # SLURM presets
│                   ├── default.yaml
│                   └── gpu.yaml
│
├── docs/
├── tests/
└── settings/                        # Legacy YAML configs
```

## pyproject.toml Configuration

### Root pyproject.toml (workspace)
```toml
[project]
name = "biahub-workspace"
requires-python = ">=3.11"

[tool.uv.workspace]
members = ["packages/*"]
```

### packages/biahub/pyproject.toml
```toml
[project]
name = "biahub"  # UNCHANGED
dependencies = [
  "iohub[tensorstore]>=0.3.0a5", "torch", "monai", "waveorder",
  "antspyx", "submitit", "pydantic>2", "click", ...
]

[project.scripts]
biahub = "biahub.cli.main:cli"

[project.optional-dependencies]
segment = ["cellpose"]
track = ["ultrack>=0.7.0rc2"]
visualization = ["napari", "PyQt6", "napari-animation"]
```

### packages/analysis-templates/pyproject.toml
```toml
[project]
name = "analysis-templates"
dependencies = [
  "biahub",
  "hydra-core>=1.3",
  "hydra-submitit-launcher>=1.2",
  "omegaconf>=2.3",
]

[project.scripts]
biahub-pipeline = "analysis_templates.cli:main"

[tool.uv.sources]
biahub = { workspace = true }
```

## Hydra Configuration Design

### Main config (conf/config.yaml)
```yaml
defaults:
  - microscope: widefield+label-free
  - step/reconstruct: default
  - step/register: default
  - step/virtual_stain: default
  - slurm: default
  - _self_

# Required parameters
dataset: ???
input_path: ???
output_path: ???

# Pipeline control
dry_run: false
steps:
  convert: true
  reconstruct: true
  register: true
  virtual_stain: true
```

### Microscope config (conf/microscope/widefield+label-free.yaml)
```yaml
# Widefield label-free optical parameters (e.g., Hummingbird)
reconstruct:
  wavelength_illumination: 0.450
  yx_pixel_size: 0.1031
  z_pixel_size: 0.2
  numerical_aperture_detection: 1.47
  numerical_aperture_illumination: 0.52

register:
  target_channel: Phase3D
  source_channel: "GFP EX488 EM525-45"
  estimation_method: beads
```

### Usage Examples
```bash
# Run 4-stage widefield+label-free pipeline
biahub-pipeline microscope=widefield+label-free \
  dataset=test_experiment \
  input_path=/data/raw \
  output_path=/data/processed

# Dry run to see what would execute
biahub-pipeline dataset=test ... dry_run=true

# Skip virtual staining step
biahub-pipeline dataset=test ... steps.virtual_stain=false

# Submit to SLURM with GPU
biahub-pipeline dataset=test ... slurm=gpu
```

## Implementation Steps

### Phase 1: Test Data Setup
1. Create test zarr at `/hpc/projects/intracellular_dashboard/refactor_biahub/test_widefield.zarr`
2. Subset of 2 FOVs from 2026_01_29 convert.zarr (channels: DAPI, TXR, BF)

### Phase 2: Workspace Scaffolding
1. Create `packages/` directory structure
2. Create root `pyproject.toml` with `[tool.uv.workspace]`
3. Move `biahub/` to `packages/biahub/src/biahub/` (NO rename)
4. NO import changes needed - imports stay as `biahub.X`
5. Create `packages/biahub/pyproject.toml`
6. Verify `uv sync` and `biahub --help` work

### Phase 3: analysis-templates Package
1. Create `packages/analysis-templates/` structure
2. Implement Hydra config schema matching ground truth scripts
3. Create step classes that wrap biahub functions:
   - `ConvertStep` → calls iohub convert
   - `ReconstructStep` → calls `biahub reconstruct`
   - `RegisterStep` → calls `biahub estimate-registration` + `biahub register`
   - `VirtualStainStep` → calls `viscy preprocess` + `viscy predict`
4. Implement `biahub-pipeline` CLI entry point
5. Create `widefield+label-free` microscope preset

### Phase 4: Validation
1. Run pipeline on test data with Hydra CLI
2. Compare output to ground truth (processed by bash scripts)
3. Verify data layout matches existing structure

## Critical Files to Modify

1. **pyproject.toml** (root) - Convert to workspace configuration
2. **biahub/** → **packages/biahub/src/biahub/** - Move (no rename)
3. **New**: `packages/biahub/pyproject.toml` - Package-specific config
4. **New**: `packages/analysis-templates/` - Create entire package

## Reference Files from Ground Truth

| Stage | Ground Truth Script | biahub Command |
|-------|---------------------|----------------|
| convert | `0-convert/` scripts | `iohub convert` |
| reconstruct | `1-reconstruct/*.yaml` | `biahub reconstruct` |
| register | `2-register_zarrv3/register_*.sh` | `biahub estimate-registration` + `biahub register` |
| virtual_stain | (not yet in this dataset) | `viscy preprocess` + `viscy predict` |

## Verification

```bash
# 1. Test workspace sync
uv sync

# 2. Test backwards-compatible Click CLI
biahub --help
biahub reconstruct --help

# 3. Test new Hydra CLI
biahub-pipeline --help
biahub-pipeline --cfg job  # Show resolved config

# 4. Run on test data (dry run)
biahub-pipeline microscope=widefield+label-free \
  dataset=test_widefield \
  input_path=/hpc/projects/intracellular_dashboard/refactor_biahub/test_widefield.zarr \
  output_path=/hpc/projects/intracellular_dashboard/refactor_biahub/output \
  dry_run=true

# 5. Run on test data (actual)
biahub-pipeline microscope=widefield+label-free \
  dataset=test_widefield \
  input_path=/hpc/projects/intracellular_dashboard/refactor_biahub/test_widefield.zarr \
  output_path=/hpc/projects/intracellular_dashboard/refactor_biahub/output

# 6. Compare output structure to ground truth
diff -r output/ ground_truth_output/
```
