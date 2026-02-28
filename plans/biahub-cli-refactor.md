# Plan: Add Hydra pipeline CLI and config templates to biahub

## Goal

Add a Hydra-based pipeline CLI (`biahub-pipeline`) that orchestrates multi-step processing workflows. Config templates define microscope-specific parameters and step sequences. Everything lives inside the existing `biahub` package.

## Key Decisions

- **Single package**: Hydra CLI and configs live inside `biahub/`, not a separate package
- **Microscope naming**: Configs will be named after the microscope and first objective to specify the type of experiment, which specifies good defaults (e.g., `mantis_63x_1.35`, `hummingbird_20x_0.55`, `dragonfly_20x_0.55`, `dragonfly_63x_1.47`, `hummingbird_63x_1.47`). Each microscope config template will contain a comment section that specifies NA, wavelength, channels.
- **First pipeline**: 4-stage widefield+label-free (convert → reconstruct → register → virtual stain)
- **Logging**: Save resolved YAML configs alongside output data

## Directory Structure (additions)

```
biahub/
├── cli/
│   ├── main.py              # Existing Click CLI
│   └── pipeline.py          # New Hydra entry point
├── conf/                    # Hydra config templates
│   ├── config.yaml          # Main defaults
│   ├── microscope/
│   │   ├── mantis_63x_1.35.yaml
│   │   └── hummingbird_63x_1.47.yaml
│   ├── step/
│   │   ├── reconstruct.yaml
│   │   ├── register.yaml
│   │   └── virtual_stain.yaml
│   └── slurm/
│       ├── default.yaml
│       └── gpu.yaml
└── pipeline/                # Step orchestration
    ├── __init__.py
    ├── runner.py            # Pipeline executor
    └── steps.py             # Step wrappers (convert, reconstruct, register, etc.)
```

## Entry Point

Add to `pyproject.toml`:
```toml
[project.scripts]
biahub-pipeline = "biahub.cli.pipeline:main"
```

## Usage

```bash
# Run full pipeline
biahub-pipeline microscope=hummingbird_63x_1.47 \
  input_path=/data/raw output_path=/data/processed

# Dry run
biahub-pipeline ... dry_run=true

# Skip a step
biahub-pipeline ... steps.virtual_stain=false

# Submit to SLURM with GPU
biahub-pipeline ... slurm=gpu
```

## Pipeline Steps

| Step | Wraps |
|------|-------|
| convert | `iohub convert` |
| reconstruct | `biahub reconstruct` |
| register | `biahub estimate-registration` + `biahub register` |
| virtual_stain | `viscy preprocess` + `viscy predict` |

## Dependencies to Add

```
hydra-core>=1.3
hydra-submitit-launcher>=1.2
omegaconf>=2.3
```

## Reference Data

- **Ground truth**: `/hpc/projects/intracellular_dashboard/virtual_stain_ft_infected/2026_01_29_A549_H2B_CAAX_DAPI_DENV_ZIKV/`
- **Test data**: `/hpc/projects/intracellular_dashboard/refactor_biahub/test_widefield.zarr`
  - 2 FOVs from the widefield dataset (channels: DAPI, TXR, BF)

## Validation

1. Run pipeline on test data, compare output structure to ground truth
2. Ensure existing Click CLI (`biahub --help`) still works unchanged
