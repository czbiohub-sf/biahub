# Where the stores are, and the beads wells

## Layouts

| Project family | Moving (light-sheet, deskewed) | Reference (phase) | Beads well |
|---|---|---|---|
| tlg2 mantis (`/hpc/projects/tlg2_mantis/<DATASET>`) | `1-preprocess/light-sheet/raw/0-deskew/<DATASET>.zarr` | `1-preprocess/label-free/0-reconstruct/<DATASET>.zarr` | `0/8/000` or `0/8/000000` |
| A549 organelle dynamics, mantis v1 (`/hpc/projects/intracellular_dashboard/organelle_dynamics/<DATASET>`) | `1-preprocess/light-sheet/raw/0-deskew/<DATASET>.zarr` | `1-preprocess/label-free/0-reconstruct/<DATASET>.zarr` | `C/1/000000` |
| mantis v2 (`0-flatfield` ... `4-assemble`) | `1-deskew/<DATASET>_1.zarr` (BF + fluorescence in one store) | `2-reconstruct/<DATASET>_1.zarr` | none: the arms are aligned on acquisition |
| DragonFly spinning disk, organelle_box (`/hpc/projects/intracellular_dashboard/organelle_box/<DATASET>`) | `0-convert/<plate>.zarr`, a `*-Prime_*` channel (fluorescence camera; the store also holds `*-Zyla_BF`) | `1-reconstruct*/<plate>_Phase.zarr` (`Phase3D`) or `2-VS/VS_<plate>.zarr` (`nuclei_prediction`), both from the Zyla BF | some projects have `0-convert/beads.zarr` (one position, both cameras); most have none: register on the cells (`ants`) |

**DragonFly.** The two cameras differ in magnification by ~5% in y/x (2025_04_18: scale
0.950/0.952, z ~1.0, a ~6 voxel z offset) although both stores claim 0.103 µm pixels, so the
transform is a camera-to-camera geometry and needs a seed that carries that scale (an
earlier registration of the same instrument, or a measured one: `references/methods.md`,
"ANTs"). Often T=1 (set `fallback.repair: null`). The z offset can differ between wells by
~±2 voxels. Applying: pass `--channels` with the fluorescence channels only -- the moving
store also holds the BF channel, which is already on the reference camera.

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
