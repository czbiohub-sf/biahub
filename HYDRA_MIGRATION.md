# Hydra Migration Guide

This document describes the new Hydra-based configuration system for biahub.

## Overview

The Hydra migration introduces powerful configuration management features:

- **Config Composition**: Mix and match config files
- **CLI Overrides**: Override any config value from command line
- **Config Groups**: Easily switch between presets (manual/beads/ants registration)
- **Structured Logging**: Better logging with automatic output organization
- **Multirun**: Run parameter sweeps for experiments
- **Reproducibility**: Automatic logging of all configs used

## Installation

The Hydra dependencies are now included in biahub. To install:

```bash
pip install -e ".[dev]"
```

## Usage

### Basic Command

```bash
# Run with default config
biahub-hydra

# See the generated config
biahub-hydra --cfg job
```

### Config Override from CLI

One of Hydra's most powerful features - override any nested config value:

```bash
# Override single values
biahub-hydra registration.target_channel_name=Phase3D

# Override nested values
biahub-hydra registration.affine_transform_settings.transform_type=affine

# Override multiple values
biahub-hydra \
  registration.target_channel_name=Phase3D \
  registration.source_channel_name=GFP \
  deskew.ls_angle_deg=30.0
```

### Using Config Groups

Config groups let you switch between preset configurations:

```bash
# Use beads-based registration instead of manual
biahub-hydra registration=beads

# Use ANTs registration
biahub-hydra registration=ants

# Combine with overrides
biahub-hydra registration=beads registration.verbose=true
```

### Running Workflows

Execute multi-step workflows defined in config files:

```bash
# Run complete lightsheet pipeline
biahub-hydra workflow=lightsheet_pipeline \
  data.input_path=./raw.zarr \
  data.output_path=./processed.zarr

# Run lattice lightsheet workflow
biahub-hydra workflow=lattice_lightsheet \
  data.input_path=./data.zarr

# Run stabilization and tracking
biahub-hydra workflow=timelapse_stabilization \
  data.input_path=./timelapse.zarr
```

### Parameter Sweeps (Multirun)

Run the same pipeline with different parameters:

```bash
# Sweep over light sheet angles
biahub-hydra --multirun \
  deskew.ls_angle_deg=28,30,32 \
  data.input_path=./data.zarr

# Sweep over multiple parameters
biahub-hydra --multirun \
  deskew.ls_angle_deg=28,30,32 \
  deskew.average_n_slices=3,5,7

# Each run gets its own output directory
# Results are organized under: multirun/YYYY-MM-DD/HH-MM-SS/0,1,2...
```

## Config Structure

### Directory Layout

```
conf/
├── config.yaml                 # Main config
├── registration/               # Registration presets
│   ├── manual.yaml
│   ├── beads.yaml
│   └── ants.yaml
├── deskew/                     # Deskew presets
│   └── default.yaml
├── workflow/                   # Workflow templates
│   ├── lightsheet_pipeline.yaml
│   ├── lattice_lightsheet.yaml
│   └── timelapse_stabilization.yaml
└── hydra/                      # Hydra settings
    └── job_logging/
        └── colorlog.yaml
```

### Creating Custom Configs

Create a new config file in your project:

```yaml
# my_experiment.yaml
defaults:
  - /registration: beads
  - /deskew: default

# Override specific values
deskew:
  ls_angle_deg: 32.0
  pixel_size_um: 0.108

registration:
  target_channel_name: MyPhaseChannel
  source_channel_name: MyFluorChannel

data:
  input_path: /path/to/data.zarr
  output_path: /path/to/output.zarr
```

Then run:

```bash
biahub-hydra --config-path=/path/to/configs --config-name=my_experiment
```

## Output Organization

Hydra automatically organizes outputs:

```
outputs/
└── 2026-02-17/
    └── 14-30-45/              # Timestamp of run
        ├── .hydra/            # Config snapshots
        │   ├── config.yaml    # Full resolved config
        │   ├── overrides.yaml # CLI overrides used
        │   └── hydra.yaml     # Hydra settings
        ├── biahub-hydra.log   # Execution log
        └── [your outputs]     # Your data outputs
```

For multirun:

```
multirun/
└── 2026-02-17/
    └── 14-35-20/
        ├── 0/                 # First parameter combination
        ├── 1/                 # Second parameter combination
        └── 2/                 # Third parameter combination
```

## Migration from Click CLI

The original Click-based CLI (`biahub`) remains fully functional. The new Hydra CLI (`biahub-hydra`) is complementary:

**Use Click CLI (`biahub`):**
- Quick single operations
- When you prefer explicit flags
- Existing scripts and workflows

**Use Hydra CLI (`biahub-hydra`):**
- Complex configurations
- Parameter sweeps
- Reproducible experiments
- Workflow orchestration
- Config composition

## Examples

### Example 1: Registration with Custom Settings

```bash
biahub-hydra registration=beads \
  registration.target_channel_name=Phase3D \
  registration.source_channel_name=GFP \
  registration.beads_match_settings.algorithm=hungarian \
  registration.beads_match_settings.cost_threshold=0.15 \
  data.input_path=./data.zarr \
  data.output_path=./registered.zarr
```

### Example 2: Deskew Parameter Sweep

```bash
biahub-hydra --multirun \
  deskew.ls_angle_deg=28,29,30,31,32 \
  deskew.pixel_size_um=0.104,0.108,0.115 \
  data.input_path=./raw.zarr
```

### Example 3: Custom Workflow

```bash
# Create custom workflow config
cat > my_workflow.yaml <<EOF
defaults:
  - /registration: beads
  - /deskew: default

workflow:
  name: my_custom_pipeline
  steps:
    - deskew
    - estimate_registration
    - register

deskew:
  ls_angle_deg: 30.0
  pixel_size_um: 0.115

registration:
  estimation_method: beads
EOF

# Run it
biahub-hydra --config-path=. --config-name=my_workflow \
  data.input_path=./data.zarr
```

## Advanced Features

### Config Interpolation

Use variables and references in configs:

```yaml
data:
  base_path: /data/experiment1
  input_path: ${data.base_path}/raw.zarr
  output_path: ${data.base_path}/processed.zarr

registration:
  verbose: ${verbose}  # References global verbose setting
```

### Conditional Defaults

Choose configs based on conditions:

```yaml
defaults:
  - registration: ${oc.env:REGISTRATION_METHOD,manual}  # From env var
  - deskew: ${deskew_preset,default}                   # From CLI
```

### Package Directives

Control where configs are merged:

```yaml
# @package _global_  - Merge at root level
# @package data      - Merge under 'data' key
# @package _group_   - Merge at group level
```

## Troubleshooting

### View Final Config

```bash
# See what config will be used
biahub-hydra --cfg job

# See the config structure
biahub-hydra --cfg hydra
```

### Validate Config

```bash
# Hydra validates on startup
# Add --cfg job to see resolved config without running
biahub-hydra --cfg job your_overrides_here
```

### Debug Mode

```bash
# Get more detailed output
biahub-hydra --cfg job --help
biahub-hydra hydra.verbose=true
```

## Future Enhancements

- [ ] Complete all command implementations (currently proof-of-concept)
- [ ] Add Hydra Compose API for programmatic config management
- [ ] Integration with MLflow/Weights&Biases for experiment tracking
- [ ] Parallel execution plugins (Ray, Dask)
- [ ] Config validation with Pydantic + Hydra integration
- [ ] More workflow templates for common use cases

## Resources

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Structured Configs Tutorial](https://hydra.cc/docs/tutorials/structured_config/intro/)
