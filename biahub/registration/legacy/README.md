# Legacy registration and stabilization

Everything biahub keeps for the retired registration / stabilization commands and configs
lives in this folder, so it can be removed in one place once nobody depends on it. The
rest of biahub reads only the unified models (`EstimateTransformSettings`,
`TransformSettings`) and runs only `estimate-transform` / `apply-transform`.

| File | What it keeps |
|---|---|
| `convert_settings.py` | The retired config schemas (defined here and nowhere else) and `biahub convert-settings`, which turns an old config into a new one |
| `aliases.py` | The old command names, as hidden aliases of the new commands |

## Old commands

| Before | After |
|---|---|
| `estimate-registration -s ... -t ... -c old.yml -o registration_settings.yml` | `estimate-transform -m ... -r ... -c new.yml -o transforms.yml` |
| `estimate-stabilization -i ... -c old.yml -o dir` | `estimate-transform -m ... -c new.yml -o transforms.yml` (no `-r`) |
| `register -s ... -t ... -c ...` | `apply-transform -m ... -r ... -c transforms.yml` |
| `stabilize -i ... -c ...` | `apply-transform -m ... -c transforms.yml` |
| `optimize-registration` | `estimate-transform` with method `ants` or `manual` |

The first four still run. Each alias is hidden from `biahub --help`, prints a
`DeprecationWarning`, converts the config on the fly, runs the new command, and prints the
new command it ran (`instead, next time: biahub apply-transform ...`) so a script can be
moved over by copy and paste. `optimize-registration` has no direct equivalent: it only
says what replaces it and exits with an error.

The aliases read the same positions and channels the old commands did:

- `estimate-registration` and beads stabilization estimate on the first position given.
- The other stabilization methods estimate every position given except the config's
  `skip_beads_fov` (matched as a substring of the path).
- `register` moves only `source_channel_names`.
- `estimate-stabilization` writes `<dir>/transforms.yml`, and an old
  `stabilize -c "<dir>/xyz_stabilization_settings/*.yml"` finds it.

## Converting a config

```
biahub convert-settings -c old.yml -o new.yml
biahub convert-settings -c "xyz_stabilization_settings/*.yml" -o transforms.yml  # per-position files fold into one
```

`convert-settings` is hidden from `--help`. The unified loaders refuse a retired config
(any of the keys in `biahub.settings._LEGACY_KEYS`) and point at this command.

- An `estimate-registration` / `estimate-stabilization` config becomes an
  `EstimateTransformSettings` for `estimate-transform`; a `register` / `stabilize` config
  becomes a `TransformSettings` for `apply-transform`.
- `approx_transform` becomes `transform.seed` (`seed_direction: inverse`, the direction the
  old pipeline stored), and `compute_approx_transform` becomes `transform.seed_from_shapes`.
- `use_prev_t_transform` becomes `transform.seed_from: previous_timepoint` for beads, the
  only method that read it; the other methods estimated each timepoint independently
  whatever the flag said.
- `estimate-registration` always read the source channel from the source store and the
  target from the target store, so its config becomes a `reference.frame: cross` estimate,
  even when both channels have the same name.
- Focus-finding stabilization had separate crops for the z focus search and the xy
  stack-reg; the unified method has one, so the conversion keeps the one the
  `stabilization_type` needs and prints which crop it dropped.
- Phase-cross-correlation's `t_reference` becomes `reference.frame`.
- ANTs preparation fields become `preprocess` blocks: `sobel_filter` -> `sobel` (and a
  `sobel` score block, as the old score compared edges), `ref_mask_radius` ->
  `mask_circle` on the reference, `clip` -> `clip_percentiles` (its fixed mantis limits
  are replaced, with a note); `crop` is dropped with a note (ANTs no longer warps the
  moving volume first, so there is no empty border to crop).

What has no equivalent is dropped, with a note printed at conversion time:

- `eval_transform_settings`: the engine flags and repairs outlying timepoints during the
  estimate instead of smoothing the series afterwards.
- `average_across_wells` and `skip_beads_fov`: per-plate orchestration, not part of the
  estimate (the `estimate-stabilization` alias still leaves the beads FOV out).
- In `register` / `stabilize` configs, `time_indices`, `keep_overhang`, `interpolation`
  and `output_ome_zarr_version`: these are now `apply-transform` options.

## What behaves differently from the old pipeline

- **Outliers** are flagged against the run's own score distribution and repaired
  (re-estimated from good neighbours), instead of being smoothed out afterwards.
- **ANTs starts from the seed.** The old pipeline let ANTs first align the volumes'
  intensity centres of mass, which replaced the seed's translation, so a converted ANTs
  config can give a different (seeded) result than the old run.
- **A failed timepoint** is recorded as `status: unreliable` with `filled_from` and a
  `note` in the transforms file, rather than silently holding the previous transform.
- **The transforms file** holds one matrix per timepoint with its score and status: one
  list shared by every position (registration on the beads FOV) or one list per position
  (stabilization). How it is applied (timepoints, canvas, interpolation, channels) is set
  by `apply-transform` options, not by the file.

## Removing the legacy code

When nobody runs the old commands or keeps old configs any more:

1. Delete this folder.
2. Remove the `convert-settings` entry and the deprecated-alias loop from
   `biahub/cli/main.py`.
3. Remove `_LEGACY_KEYS` and its check from `biahub/settings.py` (`_load_unified`).
4. Delete `tests/test_convert_settings.py` and `tests/test_legacy_aliases.py`, and update
   the scripts that import from `biahub.registration.legacy`
   (`scripts/debug_beads_registration.py`, `debug_beads_stabilization.py`,
   `debug_registration_sobel.py`) and the conversion note in
   `.claude/skills/register-with-engine/references/transforms-file.md`.
