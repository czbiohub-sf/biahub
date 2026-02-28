# Plan: Add Hydra pipeline CLI and config templates to biahub

## Goal

Add a Hydra-based pipeline CLI (`biahub-pipeline`) that orchestrates multi-step processing workflows. Config templates define microscope-specific parameters and step sequences. Everything lives inside the existing `biahub` package. Develop both microscope pipelines in parallel.

## Key Decisions

- **Single package**: Hydra CLI and configs live inside `biahub/`, not a separate package
- **Backward compatible**: Existing Click CLI and shell scripts continue to work
- **Microscope naming**: `{microscope}_{objective}_{NA}` — each config contains optical parameters, channel names, and default step settings
- **Two pipelines in parallel**:
  - `mantis_63x_1.35` — light-sheet + label-free (complex, 3 branches)
  - `hummingbird_63x_1.47` — widefield label-free (simpler, linear)
- **Logging**: Save resolved YAML configs alongside output data
- **Validate parameters**: In current biahub/settings.py, we have a collection of pydantic dataclasses which validate the yaml config files in the current workflow. When we transition to hydra we should preserve this config validation behavior - it's useful to know that the job will fail as soon as the CLI is called, rather than after dispatching SLURM jobs.

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
│   │   ├── deskew.yaml
│   │   ├── register.yaml
│   │   ├── stabilize.yaml
│   │   ├── deconvolve.yaml
│   │   ├── concatenate.yaml
│   │   └── virtual_stain.yaml
│   └── slurm/
│       ├── default.yaml
│       └── gpu.yaml
└── pipeline/                # Step orchestration
    ├── __init__.py
    ├── runner.py            # Pipeline executor
    └── steps.py             # Step wrappers
```

## Entry Point

Add to `pyproject.toml`:
```toml
[project.scripts]
biahub-pipeline = "biahub.cli.pipeline:main"
```

## Usage

```bash
# Mantis pipeline (light-sheet + label-free)
biahub-pipeline microscope=mantis_63x_1.35 \
  input_labelfree=/data/labelfree.zarr \
  input_lightsheet=/data/lightsheet.zarr \
  output_path=/data/processed

# Hummingbird pipeline (widefield label-free)
biahub-pipeline microscope=hummingbird_63x_1.47 \
  input_path=/data/convert.zarr \
  output_path=/data/processed

# Common options
biahub-pipeline ... dry_run=true
biahub-pipeline ... steps.virtual_stain=false
biahub-pipeline ... slurm=gpu
```

---

## Pipeline 1: Hummingbird 63x/1.47 (widefield label-free)

Simple linear pipeline: convert → reconstruct (3 channels) → concatenate.

### Steps

| Step | Wraps | Key Params |
|------|-------|------------|
| convert | `iohub convert` | TIFF → OME-Zarr |
| reconstruct_phase | `biahub reconstruct` | BF → Phase3D |
| reconstruct_dapi | `biahub reconstruct` | DAPI → DAPI_Density3D |
| reconstruct_txr | `biahub reconstruct` | TXR → TXR_Density3D |
| concatenate | `biahub concatenate` | merge phase + DAPI + TXR into single zarr |

### Flow

```
Raw TIFF → [convert] → convert.zarr (3ch: DAPI, TXR, BF)
                            ↓
              [reconstruct] × 3 channels
              ├─→ phase.zarr (Phase3D)
              ├─→ DAPI.zarr (DAPI_Density3D)
              └─→ TXR.zarr (TXR_Density3D)
                            ↓
                    [concatenate]
                            ↓
              output.zarr (Phase3D, DAPI_Density3D, TXR_Density3D)
```

### Microscope Config: hummingbird_63x_1.47

```yaml
# Hummingbird widefield microscope
# Objective: 63x/1.47 NA (oil)
# Modalities: widefield label-free (BF, DAPI, TXR)
# Medium: oil (n=1.515)

reconstruct:
  phase:
    input_channel_names: ["BF"]
    reconstruction_dimension: 3
    transfer_function:
      wavelength_illumination: 0.53
      yx_pixel_size: 0.103
      z_pixel_size: 0.2
      z_padding: 5
      index_of_refraction_media: 1.515
      numerical_aperture_detection: 1.47
      numerical_aperture_illumination: 0.4
    apply_inverse:
      reconstruction_algorithm: Tikhonov
      regularization_strength: 0.001
      TV_rho_strength: 0.001
      TV_iterations: 1

  fluorescence:
    DAPI:
      input_channel_names: ["DAPI"]
      wavelength_emission: 0.454
      numerical_aperture_detection: 1.44
      regularization_strength: 0.01
    TXR:
      input_channel_names: ["TXR"]
      wavelength_emission: 0.615
      numerical_aperture_detection: 1.44
      regularization_strength: 0.01

concatenate:
  channel_names: [Phase3D, DAPI_Density3D, TXR_Density3D]
```

---

## Pipeline 2: Mantis 63x/1.35 (light-sheet + label-free)

Three parallel branches processed then assembled.

### Label-free branch

| Step | Wraps | Key Params |
|------|-------|------------|
| reconstruct | `biahub reconstruct` | BF → Phase3D |
| virtual_stain | `viscy preprocess` + `viscy predict` | model checkpoint, z_window=15 |
| stabilize_phase | `biahub estimate-stabilization` + `biahub stabilize` | method=focus-finding, channel=Phase3D |
| stabilize_vs | `biahub stabilize` | reuse phase stabilization transforms |

### Light-sheet raw branch

| Step | Wraps | Key Params |
|------|-------|------------|
| deskew | `biahub deskew` | ls_angle=30°, scan_step=0.150µm, avg_n_slices=3 |
| register | `biahub estimate-registration` + `biahub register` | method=beads, bead_fov=C/1/000000 |
| stabilize_ls | `biahub estimate-stabilization` + `biahub stabilize` | Z then XY stabilization |

### Light-sheet deconvolved branch

| Step | Wraps | Key Params |
|------|-------|------------|
| estimate_psf | `biahub estimate-psf` | from bead FOV (PSF.zarr) |
| deconvolve | `biahub deconvolve` | Richardson-Lucy with estimated PSF |
| deskew_decon | `biahub deskew` | same deskew params as raw |
| stabilize_decon | `biahub stabilize` | combined registration + stabilization matrices |

### Assembly

| Step | Wraps | Key Params |
|------|-------|------------|
| assemble | `biahub concatenate` | merge all branches into single zarr |

### Flow

```
labelfree.zarr (BF)              lightsheet.zarr (GFP, mCherry)
       ↓                                ↓                    ↓
  [reconstruct]                   [deskew raw]          [deconvolve]
       ↓                                ↓                    ↓
  [virtual_stain]                 [register]            [deskew decon]
       ↓                                ↓                    ↓
  [stabilize phase+vs]           [stabilize raw]     [stabilize decon]
       ↓                                ↓                    ↓
       └────────────────┬────────────────┘                    │
                        ↓                                     │
                   [assemble] ←───────────────────────────────┘
                        ↓
                  output.zarr (Phase3D, nuclei, membrane, GFP, mCherry, ...)
```

### Microscope Config: mantis_63x_1.35

```yaml
# Mantis light-sheet + label-free microscope
# Objective: 63x/1.35 NA (oil)
# Modalities: label-free (BF) + light-sheet fluorescence (GFP, mCherry)

reconstruct:
  input_channel_names: ["BF"]
  reconstruction_dimension: 3
  phase:
    transfer_function:
      wavelength_illumination: 0.450
      yx_pixel_size: 0.1494
      z_pixel_size: 0.174
      z_padding: 5
      index_of_refraction_media: 1.4
      numerical_aperture_detection: 1.35
      numerical_aperture_illumination: 0.52
      invert_phase_contrast: false
    apply_inverse:
      reconstruction_algorithm: Tikhonov
      regularization_strength: 0.01
      TV_rho_strength: 0.001
      TV_iterations: 1

deskew:
  pixel_size_um: 0.116
  ls_angle_deg: 30.0
  scan_step_um: 0.150
  average_n_slices: 3
  keep_overhang: false

register:
  target_channel_name: Phase3D
  source_channel_name: "mCherry EX561 EM600-37"
  estimation_method: beads
  bead_fov: "C/1/000000"

stabilize:
  estimation_channel: Phase3D
  method: focus-finding
```

---

## Test Data

### Mantis test data

**Source**: `/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_DENV/`
**Output**: `/hpc/projects/intracellular_dashboard/refactor_biahub/test_mantis/`

3 FOVs, first 5 timepoints:
- **Cells** — B/2/000000, B/2/000001
  - Label-free: [125, 1, 105, 1496, 1496], channel: BF, px=0.1494µm, z=0.174µm
  - Light-sheet: [125, 2, 1168, 300, 2048], channels: GFP EX488 EM525-45, mCherry EX561 EM600-37
- **Beads** — C/1/000000 (PSF estimation + registration calibration)

```python
from iohub import open_ome_zarr
from pathlib import Path
import numpy as np

src_base = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/"
                "2024_11_07_A549_SEC61_DENV/0-convert/"
                "2024_11_07_A549_SEC61_DENV_symlink")
dst_base = Path("/hpc/projects/intracellular_dashboard/refactor_biahub/test_mantis")
dst_base.mkdir(parents=True, exist_ok=True)

T_MAX = 5

for modality in ["labelfree_1", "lightsheet_1"]:
    src_path = src_base / f"2024_11_07_A549_SEC61_DENV_{modality}.zarr"
    dst_path = dst_base / f"test_{modality}.zarr"

    src = open_ome_zarr(src_path, mode="r")
    fovs = [("B", "2", "000000"), ("B", "2", "000001"), ("C", "1", "000000")]
    channel_names = list(src["B/2/000000"].channel_names)
    dst = open_ome_zarr(dst_path, layout="hcs", mode="w", channel_names=channel_names)

    for row, col, fov in fovs:
        src_pos = src[f"{row}/{col}/{fov}"]
        dst_pos = dst.create_position(row, col, fov)
        dst_pos["0"] = np.array(src_pos["0"][:T_MAX])
        print(f"  Copied {modality} {row}/{col}/{fov}: {dst_pos['0'].shape}")

    src.close()
```

### Hummingbird test data

**Source**: `/hpc/projects/intracellular_dashboard/virtual_stain_ft_infected/2026_01_29_A549_H2B_CAAX_DAPI_DENV_ZIKV/0-convert/convert.zarr`
**Output**: `/hpc/projects/intracellular_dashboard/refactor_biahub/test_hummingbird/`

3 FOVs, all 9 timepoints (already small):
- **Cells** — 0/2/000000, 0/2/000001
  - Shape: [9, 3, 126, 2048, 2048], channels: DAPI, TXR, BF, px=0.103µm, z=0.2µm
- **Different well** — 0/3/000000

```python
from iohub import open_ome_zarr
from pathlib import Path
import numpy as np

src_path = Path("/hpc/projects/intracellular_dashboard/virtual_stain_ft_infected/"
                "2026_01_29_A549_H2B_CAAX_DAPI_DENV_ZIKV/0-convert/convert.zarr")
dst_path = Path("/hpc/projects/intracellular_dashboard/refactor_biahub/"
                "test_hummingbird/convert.zarr")
dst_path.parent.mkdir(parents=True, exist_ok=True)

src = open_ome_zarr(src_path, mode="r")
fovs = [("0", "2", "000000"), ("0", "2", "000001"), ("0", "3", "000000")]
channel_names = list(src["0/2/000000"].channel_names)
dst = open_ome_zarr(dst_path, layout="hcs", mode="w", channel_names=channel_names)

for row, col, fov in fovs:
    src_pos = src[f"{row}/{col}/{fov}"]
    dst_pos = dst.create_position(row, col, fov)
    dst_pos["0"] = np.array(src_pos["0"])
    print(f"  Copied {row}/{col}/{fov}: {dst_pos['0'].shape}")

src.close()
```

### Ground truth outputs

**Mantis**: `/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_DENV/`
- `1-preprocess/label-free/` — reconstruct, virtual stain, stabilize
- `1-preprocess/light-sheet/raw/` — deskew, register, stabilize
- `1-preprocess/light-sheet/deconvolved/` — deconvolve, deskew, stabilize
- `2-assemble/` — final concatenated output

**Hummingbird**: `/hpc/projects/intracellular_dashboard/virtual_stain_ft_infected/2026_01_29_A549_H2B_CAAX_DAPI_DENV_ZIKV/`
- `1-reconstruct/` — phase.zarr, DAPI.zarr, TXR.zarr
- `2-concatenate/` — final merged output

### Reference templates

- [mantis-analysis-template](https://github.com/czbiohub-sf/mantis-analysis-template/) — mantis pipeline scripts and configs
- `/hpc/mydata/shalin.mehta/code/infected_vs/` — Hydra-based hummingbird pipeline (reference for Hydra patterns)

## Dependencies to Add

```
hydra-core>=1.3
hydra-submitit-launcher>=1.2
omegaconf>=2.3
```

## Validation

1. Run both pipelines on test data
2. Compare output structure and data to ground truth
3. Ensure existing Click CLI (`biahub --help`) still works unchanged
4. Verify SLURM job submission works on HPC
