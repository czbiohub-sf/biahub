## Summary

Add a Nextflow-based orchestration layer for biahub image processing pipelines, along with new processing modules and CLI infrastructure.

### Nextflow pipeline (`nextflow/`)

- **`example-flatfield-deskew-reconstruct.nf`**: Four-step pipeline (flat-field correction → deskew → reconstruction → virtual stain) that fans out per position within a plate zarr. Supports `local` and `slurm` profiles, dynamic resource estimation from data shape, and `-resume` for incremental reruns.
- **Named subworkflows for composability**: Each processing step (`flat_field_wf`, `deskew_wf`, `reconstruct_wf`, `virtual_stain_wf`) is a self-contained named subworkflow with explicit `take`/`emit` interfaces. This makes it straightforward to compose new pipelines by reordering, dropping, or adding steps — e.g. a deskew-only pipeline, or inserting a new denoising step between flat-field and deskew, without duplicating process definitions. The `virtual_stain_wf` step is already conditionally included via `params.predict_config` as an example of this pattern.
- **`biahub nf` CLI subcommands** (`biahub/cli/nf.py`): Single-unit-of-work commands (`list-positions`, `init-*`, `run-*`, `compute-transfer-function`, `run-apply-inv-tf`) designed for Nextflow to schedule — no embedded SLURM/submitit logic. New steps only need a matching `init-*`/`run-*` CLI command and a subworkflow definition.
- Uses `uv run --project` for portable CLI invocation without requiring an activated venv.

### Other changes

- Dynamic per-step resource estimation (CPU/memory) via `waveorder.estimate_resources`, surfaced as `RESOURCES:` stdout lines for Nextflow to parse.
- Flat-field correction: switched from `multiprocessing.Pool` to `ThreadPoolExecutor` for timepoint-level parallelism within a position. `mp.Pool` forks the full process, which OOMs on large zarr datasets; threads share the parent's address space and are sufficient here since the actual compute is in NumPy (GIL-released) and zarr I/O.

## Test plan

Tested end-to-end via `scripts/test-nextflow.sh` on SLURM.
