# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`biahub` is a bioimage analysis hub for high-throughput data reconstruction on HPC clusters with Slurm workload management. It accelerates processing of mult`imodal microscopy datasets stored as OME-ZARR files.

## Key Development Commands

### Usage on HPC nodes login-01, login-01, gpu-*, cpu-*
Use pre-built biautils environment.

```bash
module load comp_micro
conda activate biautils
```

### Fresh installation
```bash
# Create conda environment and install
conda create -n biahub python==3.11
conda activate biahub
pip install -e ".[dev]"
```

### Code Quality and Testing
```bash
# Format code
make format              # Run black and isort formatters
make check-format        # Check formatting without changing files

# Linting and validation
make lint                # Run flake8 linting
make pre-commit          # Run all pre-commit hooks

# Testing  
make test                # Run pytest with warnings disabled
python -m pytest . --disable-pytest-warnings
```

### Package Management
```bash
make setup-develop       # Install package in development mode
make uninstall          # Uninstall the package
```

## Architecture Overview

### Command Line Interface
- **Entry point**: `biahub.cli.main:cli` - Main CLI command with lazy-loaded subcommands
- **CLI structure**: Uses Click with `LazyCommand` pattern for performance - commands are only imported when invoked
- **Available commands**: 20+ processing commands including deskew, register, stitch, deconvolve, stabilize, track, segment, etc.

### Core Processing Modules
Each processing operation has its own module with standardized structure:
- `biahub/<operation>.py` - Core processing function
- CLI wrapper at `biahub/<operation>/<operation>_cli.py`
- Configuration handled via Pydantic settings classes in `biahub/settings.py`

### Key Processing Categories
1. **Image Correction**: deskew, stabilize, register, flip
2. **Reconstruction**: deconvolve (with PSF), reconstruct (phase/birefringence)
3. **Multi-image Operations**: concatenate, stitch  
4. **Analysis**: segment, track, characterize-psf
5. **Calibration**: estimate-* commands for parameter optimization

### Configuration System
- **Settings**: Comprehensive Pydantic models in `biahub/settings.py` with validation
- **Config files**: YAML configuration files in `settings/` directory with examples
- **Validation**: Strong typing and validation for all parameters including transforms, slices, file paths

### Data Flow Architecture
- **Input/Output**: OME-ZARR datasets enable efficient HPC parallelization
- **Processing patterns**: Most operations work on `/*/*/* `position patterns for parallel processing
- **Slurm integration**: Commands can launch Slurm jobs or run locally with `-l` flag

### Key Dependencies
- Contains vendored packages in `biahub/vendor/` directory
- External stitching library: `stitch @ git+https://github.com/ahillsley/stitching@jen`
- Uses iohub for zarr i/o: `https://github.com/czbiohub-sf/iohub/`
- Uses PyTorch for GPU acceleration.

## Code Formatting Standards
- **Black**: Line length 95, skip string normalization, Python 3.10+ target
- **isort**: Black-compatible profile, line length 95
- **Exclusions**: scripts/, examples/, notebooks/, ignore/ directories excluded from formatting

## Testing Structure
- Tests located in `tests/` directory with CLI-specific tests in `test_cli/`
- Uses pytest with importlib import mode
- Includes configuration validation tests and CLI integration tests

## Development Notes
- Write docstrings in numpy style.
- Prefer PyTorch for GPU acceleration over other libraries such as cupy.
- Python 3.11+ required
- GPU support available (CUDA) for relevant operations
- Heavy use of scientific Python stack: numpy, scipy, torch, matplotlib
- Microscopy-specific libraries: napari, iohub, waveorder, antspyx
