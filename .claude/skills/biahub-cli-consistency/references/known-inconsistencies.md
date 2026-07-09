# Known open defects in the reference commands

These are real deviations present in `deskew.py`, `flat_field_correction.py`,
`virtual_stain.py`, `apply_inverse_transfer_function.py`, and `reconstruct.py` at the time
this skill was written. **All of them should be fixed** — they are also the *calibration
set*: a module under review very likely has issues from the same categories. Line numbers
drift as files change — treat them as pointers, re-locate before reporting.

The point is not just to fix these commands — it is to recognize the **class** of each
issue when auditing any CLI.

Type-hint correctness is **not** listed here: it is fully governed by the authoritative
*Option → runtime type* table in `canonical-cli.md`. Any annotation that disagrees with
that table is a defect to fix, full stop. (Note: calling `Path()` on a value that is
already a `Path` — e.g. `output_dirpath = Path(output_dirpath)` or wrapping a `/`-joined
path — is fine as a safety idiom and is **not** a finding.)

| # | Category | Where | Deviation | Fix |
|---|---|---|---|---|
| 1 | idiom | virtual_stain `slurm_out_path.mkdir(parents=True, exist_ok=True)` | Diverges from deskew/flat_field/apply-inv-tf | Use `slurm_out_path.mkdir(exist_ok=True)` (drop `parents=True`) — the canonical form. `Auto-fix: safe`. |
| 2 | idiom | flat_field `num_workers=int(slurm_args["slurm_cpus_per_task"])` vs deskew no cast | Inconsistent `num_workers` coercion | `num_cpus` is already `int` from `estimate_resources`; drop the cast (deskew form) everywhere. `Auto-fix: safe`. |
| 3 | comments | flat_field `slurm_array_parallelism: 100` has no comment; deskew/virtual_stain do | Missing the "process up to N positions at a time" comment | Add the explanatory inline comment on the parallelism value. `Auto-fix: safe`. |
| 4 | provenance | virtual_stain writes no `extra_metadata` | Output plate lacks the `{"biahub-<verb>": ...}` provenance block the others record | Record a provenance block. virtual-stain uses a VisCy jsonargparse config, not a biahub `settings` model, so dump the equivalent config rather than skipping it. `Auto-fix: needs review`. |
| 5 | name triad | `flat_field_correction.py` / `flat_field_correction_cli` vs API fn `flat_field` / command `flat-field` | Module + `_cli` stem disagree with the API-fn/command stem | Converge the whole module on `flat_field` (see canonical-cli.md → "The name triad" rename table). `Auto-fix: needs review` — renames module + `_cli`, touches main.py + test file. |
| 6 | layered functions | `reconstruct.py` inlines all logic in `reconstruct_cli`; there is no `reconstruct` API function | Every command owes a layer-2 API function `(paths, config, ...)` separate from the `_cli` | Extract `reconstruct(input_position_dirpaths, config_filepath, output_dirpath, ...)`; make `reconstruct_cli` a thin wrapper that delegates to it. `Auto-fix: needs review`. |
| 7 | signature | `reconstruct_cli(... monitor: bool = True)` vs the `monitor()` option default `False` | Misleading dead default (click passes the actual value) | Set the `_cli` signature default to `monitor: bool = False` to match the option. `Auto-fix: safe`. |
| 8 | docstring accuracy | deskew `_get_transform_matrix` documents a `keep_overhang` parameter it does not accept | numpydoc Parameters drifted from the signature | Make the Parameters block match the actual signature. ruff's D-rules won't catch a documented-but-absent param. `Auto-fix: safe`. |
| 9 | slurm args | `apply_inverse_transfer_function` sets no `slurm_array_parallelism` key; the other processing commands all do | Missing the per-position parallelism cap in `slurm_args` | Add `slurm_array_parallelism` with an explanatory comment. `Auto-fix: needs review` (confirm the intended cap). |
| 10 | docstring/CLI | `reconstruct_cli` uses a single inline example, no `\b` block, no `# noqa: D301` | Docstring example formatting diverges from the processing commands | Wrap the runnable example(s) in a `\b` block and add the trailing `# noqa: D301`, matching the other commands' style. `Auto-fix: safe`. |
| 11 | slurm args | flat_field `"slurm_partition": "cpu"`, apply-inv-tf `"cpu"`, virtual_stain `"gpu"` (only deskew uses `"preempted"`) | Partition is not the preferred `"preempted"` | Set `"slurm_partition": "preempted"`. `Auto-fix: needs review` — a GPU workload (virtual_stain) may require a GPU-capable partition, so confirm a preempted variant exists before switching it. |

## How to use this table

- When you find an issue in a target module, name its **category** so the fixer agent can
  batch related fixes.
- Idiom/comment/signature/docstring fixes (#1–3, #7, #8, #10) are `Auto-fix: safe`.
- Provenance (#4), the name triad (#5), the missing API function (#6), the slurm-args
  cap (#9), and the partition (#11) are `Auto-fix: needs review` — they change behavior or
  ripple across files (`main.py`, filenames, tests, callers).
- Do not report a deviation that ruff already enforces (import order, formatting, missing
  docstrings). Check `pyproject.toml [tool.ruff]` if unsure.
