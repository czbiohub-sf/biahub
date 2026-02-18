# Hydra Migration Summary

## Overview

This branch implements Hydra-based configuration management for biahub, enabling:
- Composable configurations
- CLI overrides for any parameter
- Structured logging
- Parameter sweeps for experiments
- Workflow orchestration

## What Was Changed

### 1. Dependencies Added (`pyproject.toml`)
- `hydra-core>=1.3.0` - Core Hydra framework
- `hydra-colorlog>=1.2.0` - Colorful logging
- `omegaconf>=2.3.0` - Configuration management

### 2. New Config Directory (`conf/`)
```
conf/
├── config.yaml                     # Main config
├── registration/                   # Registration methods
│   ├── manual.yaml
│   ├── beads.yaml
│   └── ants.yaml
├── deskew/
│   └── default.yaml
├── workflow/                       # Workflow templates
│   ├── lightsheet_pipeline.yaml
│   ├── lattice_lightsheet.yaml
│   └── timelapse_stabilization.yaml
└── hydra/
    └── job_logging/
        └── colorlog.yaml           # Logging config
```

### 3. New Python Modules
- `biahub/hydra_configs.py` - Structured config dataclasses
- `biahub/cli_hydra.py` - Hydra-based CLI entry point

### 4. New Entry Point
- `biahub-hydra` - New command for Hydra-based interface
- Original `biahub` command remains unchanged

### 5. Documentation
- `HYDRA_MIGRATION.md` - Complete user guide
- `MIGRATION_SUMMARY.md` - This file
- `examples/hydra_example.py` - Programmatic usage examples

## Key Features

### Config Composition
```bash
# Mix and match config files
biahub-hydra registration=beads deskew=default
```

### CLI Overrides
```bash
# Override any nested value from CLI
biahub-hydra registration.target_channel_name=Phase3D \
             deskew.ls_angle_deg=30.0
```

### Workflows
```bash
# Run multi-step workflows
biahub-hydra workflow=lightsheet_pipeline \
             data.input_path=./data.zarr
```

### Parameter Sweeps
```bash
# Run with multiple parameter values
biahub-hydra --multirun \
             deskew.ls_angle_deg=28,30,32 \
             deskew.average_n_slices=3,5,7
```

### Automatic Logging
- All runs logged to timestamped directories
- Full config snapshots saved
- Structured, colorful console output

## Backward Compatibility

✅ **Full backward compatibility maintained:**
- Original Click CLI (`biahub`) unchanged
- All existing code paths work as before
- Existing tests continue to pass
- Existing YAML configs still valid

The Hydra system is **additive** - it provides a new interface alongside the existing one.

## Implementation Status

### ✅ Completed
- [x] Hydra dependencies added
- [x] Config directory structure created
- [x] Structured config dataclasses
- [x] Hydra CLI entry point
- [x] Logging configuration
- [x] Workflow templates
- [x] Documentation
- [x] Examples

### 🚧 In Progress / Future Work
- [ ] Full integration with existing Click commands
- [ ] Complete all command implementations
- [ ] Pydantic + Hydra validation integration
- [ ] Test suite for Hydra configs
- [ ] Migration guide for existing configs
- [ ] More workflow templates
- [ ] Integration with experiment tracking tools

## Testing

### Manual Testing
```bash
# 1. Install with new dependencies
cd /path/to/biahub
pip install -e ".[dev]"

# 2. Test basic config
biahub-hydra --cfg job

# 3. Test config override
biahub-hydra registration=beads --cfg job

# 4. Test workflow
biahub-hydra workflow=lightsheet_pipeline --cfg job

# 5. Test examples
python examples/hydra_example.py
```

### Unit Tests (TODO)
- Config validation tests
- CLI tests with different overrides
- Workflow execution tests
- Config composition tests

## Migration Path for Users

### Phase 1: Parallel Usage
- Keep using `biahub` for existing workflows
- Try `biahub-hydra` for new experiments
- Gradually adopt Hydra features

### Phase 2: Config Migration
- Convert custom YAML configs to Hydra format
- Create workflow templates for common pipelines
- Set up parameter sweep experiments

### Phase 3: Full Adoption
- Migrate scripts to use `biahub-hydra`
- Leverage config composition
- Integrate with experiment tracking

## Benefits for Biahub

### For Users
- **Easier configuration**: Less YAML boilerplate
- **Better reproducibility**: Automatic config logging
- **Parameter exploration**: Built-in sweep support
- **Clearer workflows**: Named workflow templates

### For Developers
- **Type safety**: Structured configs with validation
- **Modularity**: Config groups for different components
- **Testing**: Easier to test different configurations
- **Extensibility**: Easy to add new config options

### For Research
- **Experiment tracking**: All configs automatically logged
- **Parameter optimization**: Multirun for hyperparameter search
- **Reproducibility**: Exact config snapshots saved
- **Collaboration**: Shareable workflow templates

## Example Use Cases

### 1. Parameter Optimization
```bash
# Find optimal deskew parameters
biahub-hydra --multirun \
  workflow=lightsheet_pipeline \
  deskew.ls_angle_deg=28,29,30,31,32 \
  deskew.average_n_slices=1,3,5,7
```

### 2. Method Comparison
```bash
# Compare registration methods
biahub-hydra --multirun \
  registration=manual,beads,ants \
  data.input_path=./test_data.zarr
```

### 3. Batch Processing
```bash
# Process multiple datasets with same config
for dataset in dataset1 dataset2 dataset3; do
  biahub-hydra workflow=lightsheet_pipeline \
    data.input_path=./data/${dataset}.zarr \
    data.output_path=./results/${dataset}.zarr
done
```

### 4. Config Templates
```bash
# Share configs as templates
biahub-hydra \
  --config-path=/shared/configs \
  --config-name=lab_standard_lightsheet \
  data.input_path=./my_data.zarr
```

## Architecture Decisions

### Why Hydra + Existing Code?
- **Non-disruptive**: Adds features without breaking existing code
- **Gradual adoption**: Users can migrate at their own pace
- **Best of both**: Combine Click's explicit flags with Hydra's composition

### Why Dataclasses + Pydantic?
- **Validation**: Keep Pydantic's powerful validation
- **Type hints**: Hydra works well with dataclasses
- **Flexibility**: Can convert between formats as needed

### Why Separate Entry Point?
- **Clarity**: Clear distinction between interfaces
- **Safety**: No risk of breaking existing workflows
- **Flexibility**: Can deprecate old CLI later if desired

## Next Steps

1. **Testing**: Add comprehensive tests for Hydra configs
2. **Integration**: Connect Hydra configs to actual command implementations
3. **Documentation**: Add more examples and tutorials
4. **Workflows**: Create more workflow templates for common use cases
5. **Feedback**: Get user feedback on the new system
6. **Migration**: Help users migrate existing configs

## Resources

- [Hydra Documentation](https://hydra.cc/)
- [Structured Configs Guide](https://hydra.cc/docs/tutorials/structured_config/intro/)
- [Config Groups Tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/)
- [Multi-run Guide](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)

## Questions?

For questions or feedback on the Hydra migration:
- Open an issue on GitHub
- Discuss in team meetings
- Check `HYDRA_MIGRATION.md` for detailed usage guide
