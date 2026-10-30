# Methods, preprocessing, score and thresholds

## Which method

| Job | Method | Notes |
|---|---|---|
| fluorescence -> label-free registration on a beads well | `beads` | the default; graph matching on detected beads, optional spectral arm for large offsets |
| registration without usable beads (e.g. fluorescence onto virtual staining) | `ants` | intensity-based; see "ANTs" below; `templates/estimate-transform-ants.yml` |
| a single hand-picked registration | `manual` | napari point annotation on one timepoint (`time_indices: <t>`), applied to all; `templates/estimate-transform-manual.yml` |
| stabilization of a channel over time (drift) | `phase_cross_corr` | translation only; crop with `center_crop_xy` |
| stabilization of phase (focus z + stackreg yx) | `focus_finding` | what the A549 projects used (`label-free/2-stabilize`) |

`reference.frame`: `cross` (another channel at the same timepoint: registration), `first`
or `previous` (the channel's own first / previous frame with data: stabilization).

## Preprocessing blocks: `preprocess`

What is done to the volumes before the method, in order; empty (the default) means the raw
volumes. Each block names the volumes it applies to, `target: [moving, reference]` (image
blocks default to both; a mask names its volume), and runs in that volume's own grid.

| Block | Kind | Use it when |
|---|---|---|
| `clip_percentiles: {low: 1, high: 99.5}` | image | hot pixels, saturated spots or bright debris dominate; each volume clipped to its own percentiles |
| `sobel: {}` | image | the two channels show the same structures with different intensities (fluorescence vs phase / virtual staining): compare edges |
| `mask_circle: {target: [reference], radius: 0.9}` | mask | a bright ring at the field edge (mantis v1 well edge); radius is a fraction of the half-width of the shorter yx axis |
| `mask_box: {target: [reference], box: [all, [y0, y1], [x0, x1]]}` | mask | only part of the field is informative (cells in one region, a damaged corner); keep it large and spread out -- scale and rotation are poorly constrained from a small region |

Image blocks apply to every method; mask blocks only to methods that can use them (`ants`;
the config refuses them for the others). Sobel before `beads` would break bead detection:
image blocks are for intensity methods unless there is a reason.

## Score and report: `score`

Every decision -- accept or flag, repair, the competitions -- uses one score per transform.

- `score.metric`: `overlap` / `residual` (beads), `correlation`, `mutual_information`,
  `gradient_correlation`; unset = the method's default (beads `overlap`, ants and the
  stabilizations `correlation`, manual `gradient_correlation`).
- `score.preprocess`: image blocks applied to the warped moving volume and the reference
  just before comparing (after the overlap is found on the raw volumes), e.g. `sobel` to
  compare edges. Intensity metrics only. The method's own `preprocess` never changes the
  score -- not even inside the method: the bead estimators' own choices (which arm, which
  pass) score the raw volumes too -- so scores from configs with different blocks stay
  comparable.
- `score.report`: more metrics computed for every transform on the raw warped pair (no
  score blocks) and written next to its score in `transforms.yml` (`report: {...}`): a fixed
  yardstick across runs, never used for decisions.

## Where each timepoint starts: `transform.seed_from`

- `input` (default): every timepoint starts from `transform.seed` (the approximate
  transform) and is estimated independently, one SLURM job per timepoint.
- `previous_timepoint`: each timepoint starts from the previous one's result, with the seed
  competing on the first pass; a failed timepoint passes its own seed on; empty frames are
  skipped. One sequential job (budgeted ~10 min per timepoint, capped at 48 h; set an sbatch time limit for
  large volumes). This is the legacy `use_prev_t_transform: true`, which most production
  beads configs used; `convert-settings` maps it for beads only.

Beads matching is deterministic but depends on where it starts: on a drifting series,
`previous_timepoint` finds better correspondences at hard timepoints.

## Stabilization: `reference.frame: first` or `previous`

- `first`: every timepoint against one frame (the first with data); absolute
  transforms, no chain. Fine while the sample stays recognisable against that frame.
- `previous`: each timepoint against the previous frame with data, steps chained onto
  the run's first frame. **Every step's error is carried by all later frames.** On a
  761-frame neuromast timelapse (GFP nuclei, whole-pixel phase correlation) the chain
  drifted -384 px in y (~0.5 px per frame) while the sample never left +-30 px of the
  centre; single-frame jumps that snap back are counted twice (out and back). Before
  applying a long `previous` chain, compare its cumulative translation with the sample's
  centroid over time; prefer `first`, and keep chains short. A neighbour-median
  reference (no chain) is planned.

## One transform for the plate, or one per position: `pool` and `transform.seed_file`

- `pool: {z: medoid | median}`: estimate on several positions and write ONE shared
  transform, the medoid of their accepted estimates (the one closest to all the others;
  a real estimate, not an average). `z: median` keeps its yx and moves z to the positions'
  median depth (a plate whose yx agrees and whose z varies by FOV, e.g. DragonFly). Each
  position's own estimate stays in the run folder (`per_position.yml`) for
  `evaluate-transform`; the entry's note gives the median gap to the medoid and the
  widest pair. Choose spread-out positions (`survey_positions`, `select_positions`).
- `transform.seed_file: <transforms file>`: start each position (and timepoint) from an
  earlier result instead of `transform.seed`. The estimate from it wins only if it
  scores strictly better, so a run never ends worse than the file; entries record
  `seeded_from`. A relative path is read from the config's folder; an edited file reruns
  the estimate (CLI fingerprint and Nextflow staging). Not with `seed_from:
  previous_timepoint`.

Plate then positions: pool a few positions into `plate/transforms.yml`, evaluate it on
held-out positions; if one transform is not enough (`transform_spread`'s `choice_z` /
`choice_yx`), estimate every position with `seed_file: plate/transforms.yml`. Use a
continuous score (`residual` for beads, an intensity metric otherwise) there: the beads
`overlap` score ties corrections under its 6-voxel radius, and a tie keeps the seed.

## ANTs

**How it works.** ANTs registers the moving volume onto the reference starting from
`transform.seed`: the seed is handed to ANTs as its starting transform (the moving volume is
never warped first), ANTs optimizes on top of it, and the result includes it. ANTs only
searches near where it starts, so **the seed decides which answer it finds**.

**Settings.**

- `transform.type: similarity` for most pairs: rotation, shift and one scale. `euclidean` is
  ANTs Rigid (no scale); `affine` can drift in z on stacks with few z slices.
- `preprocess`: `sobel` (both) for cross-modal pairs; `clip_percentiles` against bright
  outliers; masks as in the table above.
- `score.preprocess: [{sobel: {}}]` to score edges like the estimate; `score.report:
  [correlation, mutual_information]` to see raw-pair metrics next to it.
- `fallback.repair: null` for a single timepoint: there is nothing to repair from, and each
  repair is another full ANTs run.

**The seed.** A similarity transform has one scale for all axes, so an anisotropic
difference (two cameras: y/x scaled, z not) must come from the seed, with ANTs refining on
top. Identity is right when both channels share a camera grid. Otherwise estimate the shift
and scale first: an earlier run's matrix of the same instrument, or a quick measurement --
phase cross-correlation of max projections on a grid of tiles, an affine fit of the tile
shifts for y/x, and a 3D phase cross-correlation for z (what registered the DragonFly data,
within ~0.4 µm before ANTs ran). `seed_from_shapes` builds a seed from the store shapes
with the mantis light-sheet rotation built in: mantis only.

**Judging the result.** The score is a correlation; a correct cross-modal transform often
scores 0.1-0.3 and comes out `unreliable` against the bead-tuned thresholds (read it as
"check"), and on DragonFly the Sobel correlation even ranked the unregistered pair above a
good seed. Judge with overlays -- an XY max projection and an XZ section, on FOVs not used
for the estimate -- and with the report metrics against the unregistered pair.

**Resources.** A job is sized at ~30 volumes of memory (32 GB for a 66 x 2048 x 2048 pair;
the measured peak there was 14.5-22.6 GB) and 2 CPUs; ~10-30 min per estimate.

**Repeatability.** ANTs runs with a fixed random seed and as many ITK threads as the task
has CPUs (`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` overrides). One thread repeats exactly;
with several, runs agree to ~0.01 voxel at the same thread count and to ~0.1 voxel across
thread counts.

## Flagging, repair and thresholds

After estimating, timepoints below the run's `median - 2*MAD` (and below `floor`), or below
`hard_fail`, are flagged and re-estimated from other seeds (`fallback.repair`: neighbours,
consensus, the seed). Repair is skipped for methods that ignore the seed
(`phase_cross_corr`, `focus_finding`, `manual`).

The absolute values (`floor: 0.8`, `hard_fail: 0.4`) are tuned for the bead-overlap score;
for correlation scores see "Judging the result" above.

The bead-overlap score counts beads within `qc_settings.score_centroid_mask_radius` (6 voxels)
and quantizes at ~1/N beads: it cannot see misregistration below that radius, and score
differences of one bead are noise.
