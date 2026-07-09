# Canonical biahub CLI structure

Derived from the reference commands: `biahub/deskew.py`, `biahub/flat_field_correction.py`,
`biahub/virtual_stain.py`, `biahub/apply_inverse_transfer_function.py` (processing) and
`biahub/reconstruct.py` (composition).

## Function layers (applies to every command)

Every module is layered; the pairing of layers is the core contract.

| Layer | Signature | Required? | Examples |
|---|---|---|---|
| 1. Array compute | `(array, params) -> array`, pure, no I/O | **Only** for compute methods *internal to biahub* | `deskew_zyx`, `fast_deskew_zyx`, `flat_field_correction` |
| 2. API function | `<verb>(paths, config, ...) -> None`, operates on zarr stores | **Every** command | `deskew`, `flat_field`, `virtual_stain`, `apply_inverse_transfer_function` |
| 3. CLI function | `<verb>_cli`, thin `@click.command` wrapper, delegates to layer 2 | **Every** command | `deskew_cli`, `apply_inverse_transfer_function_cli` |

Layer 1 is **exempt** when the numerical work lives in an external library — do not invent
an array-in/array-out function in that case:

- `apply_inverse_transfer_function` → `waveorder`'s
  `apply_inverse_transfer_function_single_position`.
- `virtual_stain` → cytoland's `AugmentedPredictionVSUNet.predict_sliding_windows`.

Layer 2 is **never** exempt: a command that puts its logic directly in `<verb>_cli` (no
separate API function) is a defect. `reconstruct.py` currently does this — flag it.

## Command categories

- **Processing** — reads a plate, creates an output plate, fans positions out to submitit
  (one job per position). Held to the *full* template below. Examples: deskew, flat-field,
  virtual-stain, apply-inv-tf.
- **Composition** — chains other biahub/waveorder CLIs; no plate creation or submitit of
  its own. Example: reconstruct. Held to the "Composition CLI" section, but still owes a
  layer-2 API function separate from its `_cli`.
- **Estimate / utility** — produces a config or a small artifact. Planned scope; layers +
  name triad apply, full processing template does not yet.

## Processing module anatomy (top to bottom)

1. **Imports**, isort-grouped (stdlib / third-party / iohub / biahub), `lines-between-types = 1`.
   Optional module-level setup (e.g. deskew's `torch.multiprocessing.set_start_method`).
2. **Pure compute function(s)** — operate on `np.ndarray` / `torch.Tensor`, numpydoc
   docstrings. e.g. `flat_field_correction`, `deskew_zyx`.
3. **`_czyx_*` wrapper** — top-level (picklable) CZYX adapter passed to
   `process_single_position`. Only when the command uses `process_single_position`.
   (virtual-stain instead defines a bespoke `virtual_stain_position`, which is an
   intentional exception because inference is GPU-bound and serial over time.)
4. **`_init_output_plate(input_position_dirpaths, output_dirpath, settings) ->
   (shape, channel_names)`** — creates the output plate via `create_empty_plate`
   (idempotent), returns the input `(T, C, Z, Y, X)` and channel names.
5. **Other private `_helpers`** as needed.
6. **Orchestrator function `<verb>(...)`** — the SLURM fan-out logic (body order below).
7. **`<verb>_cli`** — `@click.command` + decorator stack, delegates to the orchestrator.
8. **`if __name__ == "__main__": <verb>_cli()`**

## The name triad

One stem flows through the whole module:

| Artifact | Form | Example |
|---|---|---|
| module file | `<stem>.py` | `deskew.py` |
| orchestrator fn | `<stem>` | `deskew` |
| CLI fn | `<stem>_cli` | `deskew_cli` |
| `@click.command("...")` | `<stem>` kebab | `"deskew"` |
| `main.py` COMMANDS name | `<stem>` kebab | `"deskew"` |

`flat_field_correction.py` **violates** this and must be flagged: file/`_cli` use
`flat_field_correction` while the API function and command use `flat_field`/`flat-field`.
The fix converges the whole module on one stem — `flat_field`:

| Artifact | Currently | Should be |
|---|---|---|
| module file | `flat_field_correction.py` | `flat_field.py` |
| API fn | `flat_field` | `flat_field` (already correct) |
| CLI fn | `flat_field_correction_cli` | `flat_field_cli` |
| command | `flat-field` | `flat-field` (already correct) |
| `main.py` import_path | `biahub.flat_field_correction.flat_field_correction_cli` | `biahub.flat_field.flat_field_cli` |

This is `Auto-fix: needs review` — it renames a module and its `_cli`, so it touches
`main.py`, imports elsewhere, and the test file `tests/test_cli/test_flat_field_cli.py`.

## Standard signature

Both the orchestrator and `<verb>_cli` take:

```python
def <verb>(
    input_position_dirpaths: list[Path],   # callback returns list[Path], NOT list[str]
    config_filepath: Path,
    output_dirpath: Path,                   # callback returns Path, NOT str
    sbatch_filepath: Path | None = None,
    cluster: str = "slurm",
    monitor: bool = True,                   # orchestrator default True; _cli default False
    init_only: bool = False,
):
```

### Type hints must match `biahub/cli/parsing.py` (authoritative)

Every option decorator in `parsing.py` produces a specific runtime type — via its
`callback` or its `click.Path`/`click.Choice`/`is_flag`/`int` type. The `_cli` parameter
annotation must be exactly that type. This is a hard rule, not a preference. There is no
`path_type=` on any `click.Path`, so an un-callbacked `click.Path` yields a **`str`**, not
a `Path`.

| Option decorator | Mechanism | Runtime type → annotate as |
|---|---|---|
| `input_position_dirpaths` | `callback=_validate_and_process_paths` | `list[Path]` |
| `source_position_dirpaths` | `callback=_validate_and_process_paths` | `list[Path]` |
| `target_position_dirpaths` | `callback=_validate_and_process_paths` | `list[Path]` |
| `config_filepaths` | `callback=_validate_and_process_config_paths` | `list[Path]` |
| `config_filepath` | `callback=_str_to_path` | `Path` |
| `output_dirpath` | `callback=_str_to_path` | `Path` |
| `output_filepath` | `callback=_str_to_path` | `Path` |
| `sbatch_filepath` (+ `_preprocess`/`_predict`) | `click.Path`, **no callback**, `default=None` | `str \| None` |
| `cluster` | `click.Choice`, `default="slurm"` | `str` |
| `init_only` | `is_flag`, dest `init_only` | `bool` |
| `monitor` | `is_flag` | `bool` |
| `local` | `is_flag` | `bool` |
| `num_processes` | `type=int`, `default=1` | `int` |
| ad-hoc `click.Path()` (e.g. apply-inv-tf's `--transfer-function-dirpath`) | `click.Path`, no callback | `str \| None` |

**Two-layer typing rule.** The `_cli` (layer 3) annotates the *raw* type the option
produces (above). If the `_cli` then converts a value before delegating (e.g.
`Path(transfer_function_dirpath)`), the **API function** (layer 2) annotates the
*converted* type. Where no conversion happens, both layers use the same type. So:

- `sbatch_filepath` is never converted → **`str | None` in both layers.** Annotating it
  `Path | None` (as `apply_inverse_transfer_function` does) is a mismatch — flag it.
- `output_dirpath`/`config_filepath` arrive as `Path` from the callback → `Path` in both
  layers (a defensive `output_dirpath = Path(output_dirpath)` inside the API fn is fine).
- `--transfer-function-dirpath` arrives as `str | None`, is converted with `Path(...)` in
  the `_cli`, and is `Path` in the API function — the correct pattern to mirror.

Prefer `X | None` over a bare `= None` with a non-optional annotation.

## Orchestrator body order

```python
output_dirpath = Path(output_dirpath)
slurm_out_path = output_dirpath.parent / "slurm_output"

settings = yaml_to_model(config_filepath, <Verb>Settings)
input_shape, channel_names = _init_output_plate(input_position_dirpaths, output_dirpath, settings)

num_cpus, gb_ram_per_cpu = estimate_resources(shape=input_shape, ram_multiplier=8, max_num_cpus=16)
mem_gb = num_cpus * gb_ram_per_cpu
time_minutes = 60
echo_resources(num_cpus, mem_gb, time_minutes)

if init_only:
    click.echo(f"Initialized {output_dirpath} ({len(input_position_dirpaths)} positions)")
    return

output_position_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)

<verb>_args = {
    ...,
    # Provenance — important; required on every command. The ONLY known exception
    # is apply-inv-tf, because waveorder writes the equivalent metadata for it.
    # Using an external library is not itself an excuse (virtual_stain uses cytoland
    # and still owes this block). Flag it whenever it is missing.
    "extra_metadata": {"biahub-<verb>": settings.model_dump()},
}

slurm_args = {
    "slurm_job_name": "<verb>",
    "slurm_mem": f"{mem_gb}G",
    "slurm_cpus_per_task": num_cpus,
    "slurm_array_parallelism": 100,   # process up to N positions at a time  <- keep the comment
    "slurm_time": time_minutes,
    "slurm_partition": "preempted",   # preferred; flag any other value
}
if sbatch_filepath:
    slurm_args.update(sbatch_to_submitit(sbatch_filepath))

resolved_cluster = get_submitit_cluster(cluster=cluster)
click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
executor.update_parameters(**slurm_args)

click.echo("Submitting jobs...")
jobs = []
with submitit.helpers.clean_env(), executor.batch():
    for input_position_path, output_position_path in zip(
        input_position_dirpaths, output_position_paths, strict=True
    ):
        jobs.append(executor.submit(process_single_position, _czyx_<verb>, ...))

job_ids = [job.job_id for job in jobs]
slurm_out_path.mkdir(exist_ok=True)                          # canonical form (no parents=True)
log_path = slurm_out_path / "submitit_jobs_ids.log"          # wrapping in Path(...) is also fine
with log_path.open("w") as log_file:
    log_file.write("\n".join(job_ids))

# submitit's DebugExecutor is lazy — run each in the foreground and stream progress.
if resolved_cluster == "debug":
    for job, path in zip(jobs, input_position_dirpaths, strict=True):
        job.wait()
        click.echo(f"<Verb> complete: {path}")
    return

if monitor:
    monitor_jobs(jobs, input_position_dirpaths)
```

## CLI decorator stack

```python
@click.command("<verb>")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
def <verb>_cli(...):
    """One-line summary.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub <verb> -i ./input.zarr/*/*/* -c ./<verb>_params.yml -o ./output.zarr

    \b
    Initialize the output plate only (e.g. before running per-position Nextflow workers):
    >>> biahub <verb> --init -i ./input.zarr/*/*/* -c ./<verb>_params.yml -o ./output.zarr

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub <verb> --cluster debug -i ./input.zarr/A/1/0 -c ./<verb>_params.yml -o ./output.zarr
    """  # noqa: D301
    <verb>(
        input_position_dirpaths=input_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
    )
```

The `_cli` delegates with **keyword** arguments and does no logic of its own.

## Registration (biahub/cli/main.py)

Add an entry to `COMMANDS`:

```python
{"name": "<verb>", "import_path": "biahub.<module>.<verb>_cli", "help": "..."},
```

## Composition CLI (reconstruct)

A composition command has no plate creation and no submitit batch of its own — it just
chains sub-CLIs. It is still layered: it needs a **layer-2 API function** (`reconstruct`)
that takes paths + config and calls the sub-steps, plus a thin `reconstruct_cli` that
delegates to it. `reconstruct.py` currently inlines everything in `reconstruct_cli` with
no API function — **flag this** (a caller cannot invoke `reconstruct` programmatically).

Hold a composition command to:

- the name triad and the layer-2/layer-3 split (API function + delegating `_cli`);
- correct type hints matching the option callbacks;
- the `@click.command` decorator stack it needs (may omit `init_only` if it delegates);
- a docstring whose runnable example(s) use the same `\b` block + trailing `# noqa: D301`
  formatting as the processing commands (the *examples* may differ — a composition command
  need not show `--init`/`--cluster debug` — but the formatting must match).

Do **not** demand a layer-1 array function, `_init_output_plate`, `estimate_resources`,
the debug loop, or the three-example `\b` block from a composition command — but do flag
misleading signatures (e.g. `monitor: bool = True` on a `_cli` when the `monitor()`
option default is `False`).
