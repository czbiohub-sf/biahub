# Claude Code Instructions for biahub

## Environment

- **Conda envs**: `/hpc/mydata/$USER/envs/`
- **biautils** (iohub, biahub CLI): `module load anaconda; module load comp_micro; conda activate biautils`
- **neuroglancer**: `conda activate /hpc/mydata/$USER/envs/neuroglancer_iohub`

## Repository Structure

Single package, managed with uv (migration in progress from setuptools).

```
biahub/
├── pyproject.toml       # uv-managed project
├── biahub/              # Main package
│   ├── cli/             # Click CLI + Hydra pipeline CLI
│   ├── *.py             # Processing modules
│   └── vendor/          # Vendored dependencies
├── settings/            # Example YAML configs
├── tests/
└── plans/               # Migration plans
```

## Reference Data

- **Widefield (Hummingbird)**: `/hpc/projects/intracellular_dashboard/virtual_stain_ft_infected/2026_01_29_A549_H2B_CAAX_DAPI_DENV_ZIKV/`
- **Light-sheet (Mantis)**: `/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/`
- **Test data**: `/hpc/projects/intracellular_dashboard/refactor_biahub/`

## Plans

- `plans/biahub-uv.md` — Migrate packaging from setuptools to uv
- `plans/biahub-cli-refactor.md` — Add Hydra pipeline CLI and config templates

## Code Conventions

- Click for CLI, Pydantic v2 for settings (`extra="forbid"`)
- Absolute imports: `from biahub.X import Y`
- OME-Zarr format via iohub library
- Use absolute HPC paths

## Viewing Data

```bash
# Neuroglancer (preferred)
neuroglancer_view.py /path/to/data.zarr --position A/1/0

# Browser dashboard (nd-embedding-atlas)
python /hpc/mydata/$USER/code/nd-embedding-atlas/scripts/ndimg_view.py /path/to/data.zarr --position A/1/0
```

## Related PRs

- PR #200 (feature/merge-mantis-analysis-template): Do NOT merge
