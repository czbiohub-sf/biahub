---
name: biahub-cli-consistency
description: >-
  Audit a biahub CLI module (biahub/*.py) for style and structural consistency
  against the canonical command template, and emit a structured findings report
  that a separate fixer agent can apply. Use when adding a new `biahub <command>`,
  reviewing a CLI PR, or when asked to check that CLIs "follow the same
  structure/style". Reviews and reports only — it does NOT edit files.
---

# biahub CLI consistency review

biahub commands are single-file modules under `biahub/` that expose a `@click.command`
and are registered in `biahub/cli/main.py`. The processing commands share a strict
template — see `flat_field_correction.py`, `deskew.py`, `virtual_stain.py`, and
`apply_inverse_transfer_function.py`. This skill finds where a module drifts from that
template and reports it for a fixer agent.

## The layered-function rule (most important)

Every CLI module is layered. When reviewing, first classify each function into a layer,
then check the layer's rule:

1. **Array-in / array-out compute functions** — pure `(array, params) -> array`, no I/O.
   e.g. `deskew_zyx`, `fast_deskew_zyx`, `flat_field_correction`. **Required only for
   reconstruction/compute methods internal to biahub.** A CLI whose heavy lifting comes
   from an external library is *exempt* — do NOT invent one. Exempt examples:
   `apply_inverse_transfer_function` (delegates to `waveorder`'s
   `apply_inverse_transfer_function_single_position`) and `virtual_stain`
   (delegates to cytoland's `AugmentedPredictionVSUNet`).
2. **API function** — `<verb>(paths_to_data, config, ...) -> None`, operating on zarr
   stores given paths + config. **Required for every CLI.** This is the orchestrator:
   `deskew`, `flat_field`, `virtual_stain`, `apply_inverse_transfer_function`. It must be
   a real function separate from the `_cli`, importable and callable programmatically.
3. **CLI function** — `<verb>_cli`, the thin `@click.command` wrapper that only parses
   options and delegates to the API function by keyword.

The pairing is the contract: `deskew_cli`↔`deskew`,
`apply_inverse_transfer_function_cli`↔`apply_inverse_transfer_function`. A `_cli` that
inlines its logic instead of delegating to an API function is a defect (see
`reconstruct.py`).

The two layers must also share one stem (the **name triad** below). `flat_field` breaks
this — its module/`_cli` are `flat_field_correction`/`flat_field_correction_cli` while the
API function/command are `flat_field`/`flat-field`. **Flag this as a defect**; do not
treat `flat_field_correction` as the canonical name. The whole module should converge on
the single stem `flat_field` (file `flat_field.py`, functions `flat_field`/
`flat_field_cli`, command `flat-field`).

## Scope

- **In scope now:** processing and composition commands (the four references + peers like
  `register`, `stabilize`, `stitch`, `concatenate`, `segment`, `track`, `reconstruct`).
- **Planned, not yet enforced:** `estimate-*` and utility commands. The layered-function
  rule and the name triad already apply to them, but do not yet fail them against the full
  processing template — note them as "future scope" if asked to review one.

Ruff (`make lint`, config in `pyproject.toml`: rules `F,E,W,I,N,UP,B,D`, numpy
docstring convention, isort `lines-between-types=1`) already enforces the *mechanical*
layer — import order, formatting, docstring presence. **Do not re-report anything ruff
catches.** This skill covers the *structural/semantic* layer ruff cannot see: module
anatomy, the name triad, type hints that must match the option callbacks, orchestrator
body ordering, the CLI docstring example block, provenance metadata, and registration.

## What "consistent" means here

Read these before reviewing — they are the source of truth:

- `references/canonical-cli.md` — the annotated module template, orchestrator body
  order, decorator stack, and docstring rules.
- `references/known-inconsistencies.md` — real deviations that exist *among the four
  reference commands today*. Use them to calibrate: the target module almost certainly
  has issues from the same categories.

The canonical processing CLIs (`flat_field`, `deskew`, `virtual_stain`,
`apply_inverse_transfer_function`) are the gold standard. `reconstruct.py` is a thinner
"composition" command (it chains `compute-tf` + `apply-inv-tf`); judge composition
commands against the "Composition CLI" section of the template — but note that even
composition commands must expose an API function separate from the `_cli`
(`reconstruct.py` currently does not, and that is a flagged defect, not an exemption).

## Procedure

1. **Load the canon.** Read both reference files. If reviewing something other than the
   four examples, also skim one reference command (`deskew.py`) end-to-end so line-level
   comparisons are grounded.
2. **Identify the command category** of each target module: *processing* (fans positions
   out to submitit), *composition* (chains other CLIs), or *estimate/utility* (planned
   scope — see "Scope" above). Only processing commands are held to the full template;
   the layered-function rule and name triad apply to all categories.
3. **Cross-check type hints against `biahub/cli/parsing.py`.** Every `_cli` parameter
   annotation must equal the runtime type the corresponding option decorator produces.
   Use the authoritative *Option → runtime type* table in `references/canonical-cli.md`
   (grounded in the callbacks/types in `parsing.py`); re-derive from `parsing.py` if an
   option isn't in the table. Key traps: un-callbacked `click.Path` yields **`str`** (no
   `path_type` is set anywhere), so `sbatch_filepath` is `str | None`, not `Path | None`;
   callbacked path options yield `Path`/`list[Path]`. Apply the two-layer rule: the `_cli`
   annotates the raw option type; the API function annotates the post-conversion type.
   This is the single most common real defect.
4. **Cross-check registration.** Confirm the command is in the `COMMANDS` list in
   `biahub/cli/main.py` with a `name` (kebab-case), `import_path` ending in `<verb>_cli`,
   and a `help` string.
5. **Walk the checklist below** against each target module.
6. **Emit the findings report** in the format below. Do not edit any files.

## Checklist

For each target module, check and record deviations in these categories:

- **Name triad** — module filename stem, orchestrator function, and `<verb>_cli`
  function should share one stem; the `@click.command("...")` string and `main.py` name
  are that stem in kebab-case. (`flat_field_correction.py` breaks this.)
- **Module anatomy / order** — pure compute fn(s) → `_czyx_*` picklable wrapper (if using
  `process_single_position`) → `_init_output_plate` → other `_helpers` → orchestrator →
  `<verb>_cli` → `if __name__ == "__main__":`.
- **Signatures & type hints** — orchestrator and `_cli` share the standard signature
  `(input_position_dirpaths, config_filepath, output_dirpath, sbatch_filepath=None,
  cluster="slurm", monitor=..., init_only=False)`; **every annotation matches the
  authoritative Option → runtime type table** (`references/canonical-cli.md`), applying
  the two-layer rule (raw type in `_cli`, converted type in the API fn); prefer `X | None`
  over a bare `= None`; `_cli` uses `monitor: bool = False`.
- **Orchestrator body order** — matches the ordered steps in the template (Path coerce →
  slurm_out_path → settings → init plate → resources+echo → `init_only` return →
  output paths → args dict → slurm_args dict → sbatch override → executor → batched
  submit → jobs-id log → debug foreground loop → monitor).
- **Idioms / dedup** — `slurm_out_path.mkdir(exist_ok=True)` (no `parents=True`);
  consistent `num_workers` handling (no `int(...)` cast — `num_cpus` is already `int`).
  Wrapping an already-`Path` value in `Path(...)` is an accepted safety idiom, **not** a
  finding.
- **Provenance** — this metadata block is important: output metadata must carry
  `extra_metadata={"biahub-<verb>": settings.model_dump()}`. **Flag it whenever it is
  missing.** The *only* current exception is `apply_inverse_transfer_function`, because it
  is known that `waveorder` writes the equivalent metadata for it. Do not generalize this:
  using an external library is not itself an excuse — any other command that omits the
  block (e.g. `virtual_stain`, which uses cytoland but does not write it) is still flagged
  unless you have verified that command writes the equivalent provenance.
- **CLI docstring** — one-line summary + `\b` block with the three canonical examples
  (SLURM fan-out / `--init` / `--cluster debug`) + trailing `# noqa: D301`.
- **Docstring accuracy** — numpydoc Parameters match the actual signature (no documented-
  but-absent params, no undocumented params); this is drift ruff won't flag.
- **SLURM args** — `slurm_args` dict has the standard keys; `slurm_partition` is
  `"preempted"` (the preferred partition) — **flag any other value** (e.g. `"cpu"`,
  `"gpu"`); parallelism value carries an explanatory comment.

## Findings report format

Emit one section per file. Each finding is a numbered entry:

```
### biahub/<module>.py  (category: processing)

1. [signature] biahub/<module>.py:264 — severity: structural
   Issue:   `input_position_dirpaths: list[str]` but the option callback
            `_validate_and_process_paths` returns `list[Path]`.
   Canon:   Type hints match the callback's runtime type (see deskew.py, reconstruct.py).
   Fix:     Change annotation to `list[Path]` on both the orchestrator and `_cli`.
   Auto-fix: safe.

2. [idiom] biahub/<module>.py:388 — severity: cosmetic
   Issue:   `log_path = Path(slurm_out_path / "...")` wraps an already-`Path` value.
   Canon:   `log_path = slurm_out_path / "submitit_jobs_ids.log"` (flat_field_correction.py:220).
   Fix:     Drop the redundant `Path(...)`.
   Auto-fix: safe.
```

Rules for the report:

- Anchor every finding to `file:line` and name the **category** and **severity**
  (`structural` = shape/contract differs; `cosmetic` = idiom/wording).
- State the **canonical rule** and cite the reference command that exemplifies it.
- Give a **concrete fix**, and mark `Auto-fix: safe` (mechanical, no judgment) vs
  `Auto-fix: needs review` (behavior or intent may change — e.g. adding `init_only` to a
  command that never had it, or changing a `slurm_partition`).
- End with a short **handoff line**: which findings a fixer agent can apply blind, and the
  reminder to run `make format && make lint && make test` afterward.
- If the module is fully consistent, say so explicitly and list what was checked.

## Handoff to a fixer agent

This skill only reports. To resolve findings, dispatch a separate agent (e.g. via the
Agent tool) with the findings report as its task, instructing it to apply only the
`Auto-fix: safe` items unless told otherwise, then run `make format && make lint && make
test`. Keep the reviewer and the fixer separate so the audit stays auditable.
