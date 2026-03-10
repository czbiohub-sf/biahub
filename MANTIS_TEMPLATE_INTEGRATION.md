# Mantis Analysis Template Integration

## Overview

This document describes the integration of the `mantis-analysis-template` repository into biahub as a reusable analysis pipeline template.

## What Was Integrated

### Source Repository
- **Original**: https://github.com/czbiohub-sf/mantis-analysis-template
- **Destination**: `biahub/templates/mantis/`

### Complete Template Structure

```
templates/mantis/
├── README.md                    # Complete user guide from original repo
├── run_pipeline.py              # Main pipeline orchestration
├── check_logs.py                # SLURM log checker
├── .gitignore                   # Git ignore patterns
│
├── 0-convert/                   # Data conversion stage
│   ├── link_datasets.py         # Create symlinks to raw data
│   ├── rename_wells.sh          # Rename well positions
│   ├── convert_DRAQ5_PSF_FLUOR.sh
│   └── trim_timepoints.py       # Trim timepoints if needed
│
├── 1-preprocess/                # Preprocessing pipelines
│   ├── label-free/              # Label-free imaging pipeline
│   │   ├── 0-reconstruct/       # Phase reconstruction
│   │   │   ├── phase_config.yaml
│   │   │   └── reconstruct.sh
│   │   ├── 1-virtual-stain/     # Virtual staining with VisCy
│   │   │   ├── predict.yml
│   │   │   ├── preprocess.sh
│   │   │   ├── predict.sh
│   │   │   └── combine.sh
│   │   ├── 2-stabilize/         # XYZ stabilization
│   │   │   ├── estimate-stabilization-xyz.yml
│   │   │   ├── estimate_stabilization.sh
│   │   │   ├── stabilize_phase.sh
│   │   │   └── stabilize_virtual_stain.sh
│   │   └── 3-track/             # Cell tracking
│   │       ├── tracking_settings.yml
│   │       ├── track.sh
│   │       └── visualizing_tracking_results.py
│   │
│   └── light-sheet/             # Light-sheet imaging pipelines
│       ├── raw/                 # Raw fluorescence
│       │   ├── 0-deskew/        # Deskewing
│       │   │   ├── deskew_settings.yml
│       │   │   └── deskew.sh
│       │   ├── 1-register/      # Registration
│       │   │   ├── estimate-registration-{manual,beads,ants}.yml
│       │   │   ├── estimate-registration.sh
│       │   │   └── register_timelapse.py
│       │   └── 2-stabilize/     # Z-stabilization
│       │       ├── estimate-stabilization-z.yml
│       │       ├── estimate_stabilization.sh
│       │       └── stabilize.sh
│       │
│       └── deconvolved/         # Deconvolved fluorescence
│           ├── 0-decon/         # Deconvolution
│           │   ├── estimate_psf.yml
│           │   ├── decon.yml
│           │   └── deconvolve.sh
│           ├── 1-deskew/        # Deskewing
│           │   └── deskew.sh
│           └── 2-stabilize-register/  # Combined registration & stabilization
│               ├── combine_stabilization_matrices.py
│               └── stabilize.sh
│
├── 2-assemble/                  # Data assembly
│   ├── find_crop.py             # Estimate crop regions
│   ├── estimate_crop.sh
│   ├── concatenate.yml          # Concatenation config
│   └── assemble.sh              # Assemble all channels
│
└── 3-visualization/             # Visualization & QC
    ├── README.md
    ├── visualization_videos.py
    ├── visualization_tracks.py
    ├── napari_zarr_to_mp4.py
    ├── grid_mov.py              # Create movie grids
    └── grid_per_well.py         # Per-well visualizations
```

## Integration Approach

### Why Templates?

Rather than merging mantis-analysis-template's code into biahub's core, we:

1. **Preserve Modularity**: Keep analysis pipelines separate from core library
2. **Enable Customization**: Users can copy and modify templates for their needs
3. **Support Multiple Workflows**: Easy to add more templates (lattice, confocal, etc.)
4. **Maintain Clarity**: Clear separation between library (biahub) and applications (templates)

### Benefits

**For Users:**
- **Quick Start**: Copy template to begin new analysis project
- **Best Practices**: Proven workflows from experienced users
- **Reproducibility**: Complete pipeline with configs and scripts
- **Documentation**: Comprehensive guides included

**For Biahub:**
- **Examples**: Real-world usage examples
- **Testing**: Templates serve as integration tests
- **Community**: Easier for users to contribute workflows
- **Flexibility**: Can version templates independently

## Usage Patterns

### Pattern 1: Direct Copy
```bash
# Copy template for new project
cp -r biahub/templates/mantis ~/projects/my-experiment-2026
cd ~/projects/my-experiment-2026

# Customize for your data
export DATASET=exp_2026_02_17
# Edit configs as needed...

# Run pipeline
python run_pipeline.py
```

### Pattern 2: Reference Template
```bash
# Keep template as reference
# Create custom project structure
# Copy specific scripts/configs as needed
```

### Pattern 3: Version Control
```bash
# Template as starting point with git tracking
git clone /path/to/biahub/templates/mantis my-analysis
cd my-analysis
git init
git add .
git commit -m "Initial from mantis template"
# Track modifications...
```

## Relationship to Biahub

### Template Uses Biahub Commands

The mantis template orchestrates biahub CLI commands:

```python
# From run_pipeline.py
command = (
    f"biahub reconstruct "
    f"-i {input_path} "
    f"-c phase_config.yaml "
    f"-o {output_path}"
)
```

All configurations use biahub's settings classes (from `biahub.settings`).

### Biahub Provides Core Functionality

- **Data I/O**: iohub for OME-Zarr handling
- **Processing**: deskew, registration, stabilization, etc.
- **Configuration**: Pydantic models for validation
- **SLURM**: Job submission and monitoring

### Template Adds Workflow Logic

- **Orchestration**: Multi-step pipeline coordination
- **Environment Management**: Conda environment switching
- **Error Handling**: Log checking and recovery
- **Visualization**: Data QC and presentation

## Future Enhancements

### Planned Features

1. **Template CLI Command**
   ```bash
   biahub init-template mantis --output ./my-project --dataset exp123
   ```

2. **Template Versioning**
   - Tag templates with version numbers
   - Allow users to specify template version

3. **Template Registry**
   ```bash
   biahub list-templates
   biahub template-info mantis
   ```

4. **Cookiecutter Integration**
   - Use cookiecutter for template initialization
   - Parameterize template creation

5. **More Templates**
   - Lattice light-sheet template
   - Confocal time-lapse template
   - High-content screening template

## Migration from Original Repo

### For Existing Users

If you're using the original `mantis-analysis-template` repo:

**Option 1: Continue As-Is**
- Keep using your cloned template repo
- No changes needed
- Original repo may continue to exist

**Option 2: Switch to Biahub Template**
```bash
# Update biahub
cd /path/to/biahub
git pull origin main
pip install -e .[dev]

# Use new template location
cp -r /path/to/biahub/templates/mantis ./new-project
```

**Option 3: Merge Changes**
```bash
# Compare your modifications with new template
diff -r your-project /path/to/biahub/templates/mantis/
# Selectively merge improvements
```

### Updating Templates

To get template updates:

```bash
# Update biahub
cd /path/to/biahub
git pull

# Check for template changes
cd templates/mantis
git log -- .

# Copy specific updated files to your project
cp templates/mantis/run_pipeline.py ~/my-project/
```

## Contributing to Templates

### Adding Features to Mantis Template

1. Fork biahub repository
2. Modify `templates/mantis/`
3. Test thoroughly
4. Update documentation
5. Submit PR

### Creating New Templates

1. Create `templates/your-template/`
2. Follow mantis structure as example
3. Include comprehensive README
4. Add to `templates/README.md`
5. Submit PR

## Technical Details

### File Sizes
- Total files: 84
- Total size: ~200 KB
- Mostly small scripts and configs

### Dependencies
Templates rely on existing biahub dependencies plus:
- For mantis: viscy, waveorder, ultrack (optional)

### Compatibility
- Works with existing biahub installations
- No changes to biahub core code required
- Backward compatible

## Questions & Support

### Common Questions

**Q: Should I modify the template in place?**
A: No, copy it to your project directory first.

**Q: Can I version control my modified template?**
A: Yes! Initialize git in your project directory.

**Q: How do I get template updates?**
A: Update biahub, then copy specific updated files to your project.

**Q: Can I contribute my custom workflow?**
A: Yes! Create a new template and submit a PR.

### Getting Help

- Template issues: Check template's README
- Biahub issues: https://github.com/czbiohub-sf/biahub/issues
- Command-specific help: `biahub <command> --help`

## Summary

This integration:
- ✅ Preserves mantis-analysis-template functionality
- ✅ Makes templates easily accessible to biahub users
- ✅ Enables template ecosystem growth
- ✅ Maintains clear separation of concerns
- ✅ Supports customization and version control
- ✅ Backward compatible with existing workflows

The mantis template is now part of biahub, ready to help users quickly set up robust analysis pipelines for their Mantis microscopy data.
