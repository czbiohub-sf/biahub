# biahub

[![Docs][docs-badge]][docs-url]

<!-- Badges -->
[docs-badge]: https://github.com/czbiohub-sf/biahub/actions/workflows/docs.yml/badge.svg
<!-- URLs -->
[docs-url]: https://czbiohub-sf.github.io/biahub

<!-- The sections below are the single source of truth for docs/index.md, which
     includes them verbatim via pymdownx.snippets. Keep links inside the snippet
     regions absolute so they resolve both on GitHub and on the docs site. -->
<!-- --8<-- [start:intro] -->
Bio-image analysis hub supporting high-throughput data reconstruction on HPC clusters with [Slurm](https://slurm.schedmd.com/documentation.html) workload management.

`biahub` was originally developed to reconstruct data acquired on the [mantis](https://doi.org/10.1093/pnasnexus/pgae323) microscope using the [shrimPy](https://github.com/czbiohub-sf/shrimPy) acquisition engine, and has since been extended to process diverse multimodal datasets. `biahub` reconstruction workflows rely on OME-ZARR datasets (for example, as created with [iohub](https://github.com/czbiohub-sf/iohub)) which enable efficient parallelization across compute nodes.

<!-- --8<-- [end:intro] -->

![FOV reconstruction](docs/figures/dynacell_fig2.png)

<!-- --8<-- [start:body] -->
## Install

`biahub` uses [uv](https://docs.astral.sh/uv/) for environment and dependency management. Install `uv` first:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository and create the environment. `uv` reads `pyproject.toml` and `uv.lock`, downloads a suitable Python (>=3.12) if needed, and creates a `.venv` in the repository:

```sh
git clone https://github.com/czbiohub-sf/biahub.git
cd biahub
uv sync
```

Run commands with `uv run`, which always resolves against the locked environment:

```sh
uv run biahub --help
```

Alternatively, activate the environment once and call `biahub` directly:

```sh
source .venv/bin/activate  # Windows: .venv\Scripts\activate
biahub --help
```

### Optional dependencies

Heavier or task-specific dependencies live in extras. Add them with `--extra`:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `segment` | `cellpose` | `biahub segment` |
| `track` | `ultrack` | `biahub track` |
| `stain` | `cytoland` (VisCy) | `biahub virtual-stain` |
| `gui` | `napari`, `PyQt6` | interactive/visual commands |
| `all` | all of the above | everything |

```sh
uv sync --extra all
# or, per command:
uv run --extra stain biahub virtual-stain --help
```

### Development install

```sh
uv sync --group dev   # same as `make setup-develop`
pre-commit install
```

See the [contributing guide](https://github.com/czbiohub-sf/biahub/blob/main/CONTRIBUTING.md) for the full development workflow.

## Command line interface

Reconstruction is driven by the `biahub` CLI. The commands share a small set of conventions:

| Option | Meaning |
| --- | --- |
| `-i, --input-position-dirpaths` | input position(s), typically a glob such as `./data.zarr/*/*/*` |
| `-o, --output-dirpath` | output zarr store or YAML file |
| `-c, --config-filepath` | YAML settings file (see [`settings/`](https://github.com/czbiohub-sf/biahub/tree/main/settings) for examples) |

Execution is controlled by a few more flags: `-l, --local` runs the work on the current machine instead of submitting Slurm jobs, `--cluster {slurm,local,debug}` selects the backend explicitly (`debug` runs in-process in the foreground), `--init` only creates the output store and exits, and `-m, --monitor` follows the submitted jobs until they finish.

Some steps include an `estimate-*` command that writes a YAML of parameters you can inspect and edit, followed by an apply command that consumes that YAML. Run `biahub <command> --help` for the full option list, or browse the [CLI reference](https://czbiohub-sf.github.io/biahub/cli/).

### Available commands

| Command | Description |
| --- | --- |
| `estimate-bleaching` | Estimate photobleaching from raw data |
| `characterize-psf` | Characterize a point spread function and write a report |
| | |
| `flat-field` | Apply flat-field correction to selected channels |
| `flip` | Flip images in a dataset |
| `pyramid` | Create multiscale pyramid levels |
| `process-with-config` | Apply arbitrary YAML-defined functions to a dataset |
| | |
| `estimate-deskew` | Estimate deskewing parameters |
| `deskew` | Deskew positions across T and C |
| | |
| `estimate-psf` | Estimate a PSF from beads |
| `deconvolve` | Deconvolve across T and C using a PSF |
| | |
| `compute-tf` | Compute a transfer function from a PSF |
| `apply-inv-tf` | Apply an inverse transfer function to a dataset |
| `reconstruct` | Reconstruct phase/birefringence in one step |
| | |
| `estimate-registration` | Estimate the affine transform between arms or timepoints |
| `optimize-registration` | Refine a transform via match filtering |
| `register` | Apply an affine transform to positions |
| `estimate-crop` | Estimate the crop region for dual-channel alignment |
| | |
| `estimate-stabilization` | Estimate XYZ translation matrices |
| `stabilize` | Apply stabilization transforms |
| | |
| `estimate-stitch` | Estimate stitching parameters for positions |
| `stitch` | Stitch positions within wells |
| `concatenate` | Concatenate datasets channel-wise, with optional cropping |
| | |
| `segment` | Segment positions with a pretrained model or pipeline |
| `virtual-stain` | Run VisCy/cytoland virtual staining |
| `track` | Track objects in 2D/3D time-lapse data |
| | |
| `nf list-positions` | List position keys of a plate zarr (used for Nextflow fan-out) |

### Example: raw data to registered volumes

```sh
# CONVERT TO ZARR
iohub convert -i ./acq_name/acq_name_labelfree_1 -o ./labelfree.zarr
iohub convert -i ./acq_name/acq_name_lightsheet_1 -o ./lightsheet.zarr

# DECONVOLVE FLUORESCENCE
biahub characterize-psf -i ./beads.zarr -c ./characterize.yml -o ./report/  # optional
biahub estimate-psf     -i ./beads.zarr -c ./psf.yml -o ./psf.zarr
biahub deconvolve       -i ./lightsheet.zarr -c ./deconvolve.yml \
                        --psf-dirpath ./psf.zarr -o ./lightsheet_deconvolved.zarr

# DESKEW FLUORESCENCE
biahub estimate-deskew -i ./lightsheet.zarr/0/0/0 -o ./deskew.yml
biahub deskew          -i ./lightsheet.zarr/*/*/* -c ./deskew.yml -o ./lightsheet_deskewed.zarr

# RECONSTRUCT PHASE/BIREFRINGENCE
biahub reconstruct -i ./labelfree.zarr/*/*/* -c ./recon.yml -o ./labelfree_reconstructed.zarr

# STABILIZE
biahub estimate-stabilization -i ./labelfree.zarr/*/*/* -o ./stabilization.yml \
                              --stabilize-xy --stabilize-z
biahub stabilize              -i ./labelfree.zarr/*/*/* -c ./stabilization.yml \
                              -o ./labelfree_stabilized.zarr

# REGISTER
biahub estimate-registration -s ./labelfree_reconstructed.zarr/0/0/0 \
                             -t ./lightsheet_deskewed.zarr/0/0/0 -o ./register.yml
biahub optimize-registration -s ./labelfree_reconstructed.zarr/0/0/0 \
                             -t ./lightsheet_deskewed.zarr/0/0/0 \
                             -c ./register.yml -o ./register_optimized.yml
biahub register              -s ./labelfree_reconstructed.zarr/*/*/* \
                             -t ./lightsheet_deskewed.zarr/*/*/* \
                             -c ./register_optimized.yml -o ./registered.zarr

# CONCATENATE CHANNELS
biahub concatenate -c ./concatenate.yml -o ./concatenated.zarr

# STITCH
biahub estimate-stitch -i ./acq_name.zarr/*/*/* -o ./stitch.yml
biahub stitch          -i ./acq_name.zarr/*/*/* -c ./stitch.yml -o ./stitched.zarr
```

## Nextflow workflows

The [`nextflow/`](https://github.com/czbiohub-sf/biahub/tree/main/nextflow) directory contains [Nextflow](https://www.nextflow.io/) pipelines that run a whole reconstruction end to end for established workflows, fanning each step out over positions and submitting the work to Slurm.

```
nextflow/
├── nextflow.config   # executor profiles, resources, retry policy, reports
├── mantis-v2.nf      # mantis-v2 pipeline: the step order and directory layout
└── modules/          # one path-agnostic subworkflow per step
```

Each step module runs the same CLI you would run by hand, in two phases: `biahub <step> --init` creates the output store and reports the CPU/memory/time a single position needs, then Nextflow fans out one `biahub <step> --cluster debug` task per position with those resources. `--cluster debug` makes the CLI do the work in-process, so Nextflow — not `submitit` — owns job submission.

### Running a pipeline

Install [Nextflow](https://www.nextflow.io/docs/latest/install.html) (requires Java 17+), then launch from the `nextflow/` directory so that `nextflow.config` is picked up automatically:

```sh
cd nextflow
nextflow run mantis-v2.nf \
    -profile slurm \
    --input /path/to/raw.zarr \
    --output /path/to/output \
    --flat_field_config    ./configs/flat_field.yml \
    --deskew_config        ./configs/deskew.yml \
    --reconstruct_config   ./configs/reconstruct.yml \
    --virtual_stain_config ./configs/virtual_stain.yml \
    --concatenate_config   ./configs/concatenate.yml \
    --track_config         ./configs/track.yml \
    --biahub_project /path/to/biahub
```

| Parameter | Description |
| --- | --- |
| `--input` | raw source dataset (the mantis-v2 pipeline expects a plate zarr) |
| `--output` | directory that receives every step's output, plus run reports |
| `--*_config` | YAML settings for each step, the same files the CLI takes via `-c` |
| `--biahub_project` | path to a `biahub` checkout; tasks then run as `uv run --project <path> biahub ...`. Omit it to use whatever `biahub` is on `PATH` in each task |
| `--max_positions` | process only the first N positions (`0` = all); useful for smoke tests |
| `--max_workers` | Slurm queue size, i.e. the cap on concurrently submitted jobs (default 100) |

Profiles select *where* work runs:

- `-profile local` — everything on the current machine. Good for debugging and single-node runs.
- `-profile slurm` — per-position work goes to the `preempted` partition, virtual-stain prediction to the `gpu` partition, and lightweight init steps stay local. Preempted, timed-out, and OOM-killed tasks are retried automatically (up to 5 times, with escalating time/memory); genuine errors fail fast instead of burning retries.

Nextflow's own flags still apply — most usefully `-resume` to reuse completed tasks after a failure or a config tweak, and `-w /fast/scratch` to move the work directory off the output filesystem.

### Outputs

Each step writes `<dataset>.zarr` into its own numbered subdirectory of `--output`, so intermediates stay inspectable:

```
output/
├── 0-flatfield/     1-deskew/     2-reconstruct/
├── 3-virtual-stain/ 4-track/      5-assemble/
└── nextflow/
    ├── report.html  timeline.html  dag.html  trace.txt
    ├── slurm_output/<step>/        # per-task Slurm logs
    └── work/                       # Nextflow work directory
```

The pipeline runs flat-field → deskew → reconstruct → virtual-stain → assemble → track; `5-assemble` concatenates the deskew, reconstruct, and virtual-stain channels into one plate, which tracking then consumes.

### Adapting a pipeline

The step modules are path-agnostic: each subworkflow is handed an input zarr, an output zarr, and a config, and knows nothing about where it sits in the pipeline. The pipeline file owns the directory layout (the `directory_layout()` map) and the order of steps, so reordering steps, dropping one, or pointing a step at a different upstream store is an edit in `mantis-v2.nf` alone. Copy it as a starting point for a new instrument or dataset.

## Contributing

We would appreciate bug reports and code contributions if you use this package. If you would like to contribute to this package, please read the [contributing guide](https://github.com/czbiohub-sf/biahub/blob/main/CONTRIBUTING.md).
<!-- --8<-- [end:body] -->
