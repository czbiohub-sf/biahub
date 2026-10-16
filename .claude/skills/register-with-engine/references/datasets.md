# Where the stores are, and the beads wells

## Layouts

| Project family | Moving (light-sheet, deskewed) | Reference (phase) | Beads well |
|---|---|---|---|
| tlg2 mantis (`/hpc/projects/tlg2_mantis/<DATASET>`) | `1-preprocess/light-sheet/raw/0-deskew/<DATASET>.zarr` | `1-preprocess/label-free/0-reconstruct/<DATASET>.zarr` | `0/8/000` or `0/8/000000` |
| A549 organelle dynamics, mantis v1 (`/hpc/projects/intracellular_dashboard/organelle_dynamics/<DATASET>`) | `1-preprocess/light-sheet/raw/0-deskew/<DATASET>.zarr` | `1-preprocess/label-free/0-reconstruct/<DATASET>.zarr` | `C/1/000000` |
| mantis v2 (`0-flatfield` ... `4-assemble`) | `1-deskew/<DATASET>_1.zarr` (BF + fluorescence in one store) | `2-reconstruct/<DATASET>_1.zarr` | none: the arms are aligned on acquisition |

Previous registrations live in `1-preprocess/light-sheet/raw/1-register/` (the
`estimate-registration*.sh` script names the beads well; the
`estimate-registration-beads.yml` holds the seed and detection settings). Configs from
before 2026-03 use retired field names; `convert-settings` handles the current legacy
schema, older ones need the renames first (`beads_match_settings.t_reference` ->
`affine_transform_settings.t_reference`, `filter_*_distance_threshold` ->
`filter_matches_settings.*_distance_quantile`, `filter_angle_threshold` ->
`filter_matches_settings.angle_threshold`).

## Regenerating a cleaned beads FOV

If `0-deskew` / `0-reconstruct` were cleaned, rebuild only the beads position from the raw
stores in `0-convert/`, with the project's own configs, into a scratch folder:

```bash
biahub deskew -i <raw lightsheet.zarr>/<beads well> -c <project>/1-preprocess/light-sheet/raw/0-deskew/deskew_settings.yml -o <scratch>/light-sheet-deskew.zarr
biahub reconstruct -i <raw labelfree.zarr>/<beads well> -c <project>/1-preprocess/label-free/0-reconstruct/phase_config.yaml -o <scratch>/beads-recon/label-free-reconstruct.zarr
```

Run each reconstruct in its own folder: it writes `transfer_function_<config>.zarr` next to
the output, and two runs sharing a folder overwrite each other's (and `reconstruct` does not
report the failed job). Check every timepoint has data afterwards; a timepoint empty in both
the raw light-sheet and label-free is an acquisition gap.
