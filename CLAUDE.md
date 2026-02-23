# Claude Code Instructions for biahub

This file contains project-specific instructions and environment setup for working on the biahub repository.

## Environment Setup

### Conda Environment Discovery

**Custom conda environments location**: `/hpc/mydata/$USER/envs/`

**List all available conda environments**:
```bash
conda env list --prefix /hpc/mydata/$USER/envs/
```

### Neuroglancer Environment

For neuroglancer visualization tasks:

```bash
conda activate /hpc/mydata/$USER/envs/neuroglancer_iohub
```

**Tools available**:
- `neuroglancer_view.py` - Neuroglancer viewer CLI for OME-Zarr datasets

### Biautils Environment

For iohub and general bioimage analysis tasks:

```bash
module load anaconda
module load comp_micro
conda activate biautils
```

**Tools available**:
- `iohub` - OME-Zarr I/O operations
- `biahub` CLI - Core processing modules

## Repository Structure

### Current Structure (Single Package)
```
biahub/
├── pyproject.toml          # setuptools-based configuration
├── biahub/                 # Main package
│   ├── cli/                # Click-based CLI
│   ├── *.py               # Processing modules
│   └── vendor/            # Vendored dependencies
├── settings/              # Example YAML configurations
├── tests/                 # Test suite
└── docs/                  # Documentation
```

### Planned Structure (UV Workspaces - See plan file)
```
biahub/
├── pyproject.toml          # Workspace root
├── packages/
│   ├── biahub/            # Core processing (unchanged name)
│   └── analysis-templates/ # Hydra-based workflows
└── ...
```

**See plan**: `/hpc/mydata/$USER/code/biahub/plans/biahub-uv.md`

## Important Paths

### Project Repositories
- **biahub**: `/hpc/mydata/$USER/code/biahub` (this repository)
- **infected_vs**: `/hpc/mydata/$USER/code/infected_vs` (hummingbird microscope processing)

### Reference Data for Development

**Widefield microscope (Hummingbird)**:
- `/hpc/projects/intracellular_dashboard/virtual_stain_ft_infected/2026_01_29_A549_H2B_CAAX_DAPI_DENV_ZIKV/`
- Stages: 0-convert, 1-reconstruct, 2-concatenate (+ zarrv3 variants)

**Light-sheet + label-free microscope (Mantis)**:
- `/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/`
- Stages: 0-convert, 1-preprocess, 2-assemble, 3-visualization, 4-phenotyping

## Development Workflow

### Current Branch
Check with `git status` and `git branch`

### Running biahub CLI
```bash
# In current environment
biahub --help
biahub <command> --help

# Common commands
biahub reconstruct
biahub register
biahub stabilize
biahub concatenate
```

### Testing
```bash
pytest tests/
pytest tests/test_concatenate.py -v
```

## Migration to UV Workspaces (In Progress)

**Status**: Planning phase
**Plan file**: `plans/biahub-uv.md`

**Key decisions**:
- Keep `biahub` package name unchanged
- New `analysis-templates` package for Hydra-based workflows
- Start with 4-stage Dragonfly workflow (convert → reconstruct → register → virtual stain)
- Data layout must match existing bash-based template output structure

**Related PRs**:
- PR #200 (feature/merge-mantis-analysis-template): Do NOT merge
- VisCy issue #353: Reference for UV workspaces pattern

## Code Style & Conventions

### Import Organization
- Standard library first
- Third-party packages second
- Local imports last
- Prefer absolute imports: `from biahub.X import Y`

### CLI Development
- Use Click for command-line interfaces
- Lazy loading for command discovery (see `biahub/cli/main.py`)
- Pydantic for configuration validation

### Configuration
- Pydantic v2 models in `biahub/settings.py`
- YAML configuration files in `settings/` directory
- Extra fields forbidden to catch typos: `extra="forbid"`

## Common Tasks

### Viewing OME-Zarr Data

Two options for visualizing zarr stores during analysis or development:

**Option 1: `neuroglancer_view.py` (preferred — faster)**
```bash
conda activate /hpc/mydata/$USER/envs/neuroglancer_iohub
neuroglancer_view.py /path/to/data.zarr --position A/1/0
```

**Option 2: `ndimg_view.py` (from nd-embedding-atlas — interactive browser UI)**
```bash
conda activate /hpc/mydata/$USER/envs/idetik_iohub
python /hpc/mydata/$USER/code/nd-embedding-atlas/scripts/ndimg_view.py /path/to/data.zarr --position A/1/0
```
- Launches a web dashboard at `http://localhost:5055` with a table of FOVs and WebGL viewer
- Supports multiple zarr stores for side-by-side comparison
- Filter channels with `--channels DAPI,TXR`

### Creating Test Data (iohub)
```bash
module load anaconda
module load comp_micro
conda activate biautils

python << 'EOF'
from iohub import open_ome_zarr
# ... see plans/biahub-uv.md for test data creation script
EOF
```

### Checking Dataset Metadata
```bash
conda activate biautils
python -c "from iohub import open_ome_zarr; d = open_ome_zarr('/path/to/data.zarr'); print(d.channel_names, d.data.shape)"
```

## Notes for Claude

- When working with HPC paths, always use absolute paths
- Check for existing conda environments before suggesting new ones
- The user prefers Click over argparse for CLI development
- The user works with OME-Zarr format via iohub library
- Always check `plans/biahub-uv.md` for current migration status
- Test data locations are important for validation
