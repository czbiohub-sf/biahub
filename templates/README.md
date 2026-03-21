# Biahub Analysis Templates

This directory contains complete analysis pipeline templates for common microscopy workflows. These templates provide ready-to-use project structures with scripts, configurations, and documentation for processing and analyzing imaging data.

## Available Templates

### 1. Mantis Analysis Template (`mantis/`)

Complete pipeline for processing 4D microscopy datasets from the Mantis microscope, including label-free and fluorescence imaging.

**Features:**
- Zarr conversion and data management
- Label-free phase reconstruction
- Virtual staining with VisCy
- Light-sheet deskewing and deconvolution
- Multi-channel registration
- XYZ stabilization
- Cell tracking with ultrack
- Visualization and quality control

**Best for:** Standard Mantis microscope workflows with label-free and fluorescence imaging

See [`mantis/README.md`](mantis/README.md) for details.

### 2. Dragonfly Template (`dragonfly/`)

Pipeline variant optimized for Dragonfly microscope data analysis.

**Features:**
- Similar structure to Mantis template
- Dragonfly-specific configurations
- Optimized parameters for Dragonfly imaging

**Best for:** Data acquired with Dragonfly spinning disk confocal systems

See [`dragonfly/README.md`](dragonfly/README.md) for details.

### 3. Zebrafish Physics Pipeline (`zebraphysics-mantis-pipeline/`)

Specialized pipeline for zebrafish physics experiments using Mantis microscopy.

**Features:**
- Extended drift analysis tools
- Physics-specific data processing
- Customized tracking parameters
- Additional SLURM batch configurations

**Best for:** Zebrafish developmental biology and physics experiments

See [`zebraphysics-mantis-pipeline/README.md`](zebraphysics-mantis-pipeline/README.md) for details.

## Using Templates

### Option 1: Manual Copy

Clone the template directory to start a new project:

```bash
# From your project directory
cp -r /path/to/biahub/templates/mantis ./my-dataset-analysis
cd my-dataset-analysis

# Set your dataset name
export DATASET=my_dataset_name

# Follow the template README for next steps
```

### Option 2: CLI Command (Coming Soon)

```bash
# Initialize a new project from template
biahub init-template mantis --output ./my-dataset-analysis --dataset my_dataset_name
```

## Template Structure

Each template typically includes:

- **0-convert/**: Data conversion and setup scripts
- **1-preprocess/**: Preprocessing pipelines (reconstruction, registration, stabilization)
- **2-assemble/**: Assembly and concatenation
- **3-visualization/**: Visualization and QC scripts
- **run_pipeline.py**: Main pipeline orchestration
- **check_logs.py**: Log checking utilities
- **README.md**: Detailed documentation
- **Config files**: YAML configurations for each step

## Creating Custom Templates

To contribute a new template:

1. Create a new directory: `templates/your-template-name/`
2. Structure it with clear numbered stages
3. Include comprehensive README with:
   - Overview and workflow diagram
   - Setup instructions
   - Usage examples
   - Configuration details
4. Add YAML configs for biahub commands
5. Include example/test data if possible
6. Submit a PR!

## Environment Setup

Most templates require multiple conda environments:

```bash
# VisCy for virtual staining
conda create -n viscy python=3.11
conda activate viscy
pip install viscy
conda deactivate

# Waveorder for phase reconstruction
conda create -n waveorder python=3.11
conda activate waveorder
pip install waveorder
conda deactivate

# Biahub for preprocessing
conda create -n biahub python=3.11
conda activate biahub
pip install -e /path/to/biahub[dev]
conda deactivate
```

See individual template READMEs for specific requirements.

## Best Practices

When using templates:

1. **Version Control**: Initialize git in your project directory
   ```bash
   cd my-dataset-analysis
   git init
   git add .
   git commit -m "Initial commit from mantis template"
   ```

2. **Configuration**: Update YAML configs with your dataset parameters
3. **Documentation**: Keep notes on modifications and results
4. **Testing**: Run on a small subset first before full dataset
5. **Logs**: Check SLURM logs regularly with `check_logs.py`

## Support

- Report issues: https://github.com/czbiohub-sf/biahub/issues
- Documentation: https://github.com/czbiohub-sf/biahub
- Template-specific questions: See individual template READMEs

## Contributing

We welcome contributions of new templates and improvements to existing ones! See CONTRIBUTING.md for guidelines.
