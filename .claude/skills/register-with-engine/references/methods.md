# Methods, propagation and thresholds

## Which method

| Job | Method | Notes |
|---|---|---|
| light-sheet -> label-free registration on a beads well | `beads` | the default; graph matching on detected beads, optional spectral arm for large offsets |
| registration without usable beads, or cross-modal fine-tuning | `ants` | intensity-based; `ants.sobel_filter: true` matches edges (cross-modal pairs) |
| a single hand-picked registration | `manual` | napari point annotation, one timepoint, applied to all |
| stabilization of a channel over time (drift) | `phase-cross-corr` | translation only; crop with `center_crop_xy` |
| stabilization of phase (focus z + stackreg yx) | `focus-finding` | what the A549 projects used (`label-free/2-stabilize`) |

`reference.frame`: `cross` (another channel at the same timepoint: registration), `first`
or `previous` (the channel's own first / previous frame with data: stabilization).

## Where each timepoint starts: `transform.seed_from`

- `input` (default): every timepoint starts from `transform.seed` (the approximate
  transform) and is estimated independently, one SLURM job per timepoint.
- `previous_timepoint`: each timepoint starts from the previous one's result, with the seed
  competing on the first pass; a failed timepoint passes its own seed on; empty frames are
  skipped. One sequential job (~5 min per timepoint by default; set an sbatch time limit for
  large volumes). This is the legacy `use_prev_t_transform: true`, which most production
  beads configs used; `convert-settings` maps it for beads only.

Beads matching is deterministic but depends on where it starts: on a drifting series,
`previous_timepoint` finds better correspondences at hard timepoints.

## Flagging, repair and thresholds

After estimating, timepoints below the run's `median - 2*MAD` (and below `floor`), or below
`hard_fail`, are flagged and re-estimated from other seeds (`fallback.repair`: neighbours,
consensus, the seed). Repair is skipped for methods that ignore the seed
(`phase-cross-corr`, `focus-finding`, `manual`).

The absolute values (`floor: 0.8`, `hard_fail: 0.4`) are tuned for the bead-overlap score.
For correlation-based scores (ants, phase cross-correlation, focus finding) a correct
transform often scores 0.1-0.3, so many timepoints come out `unreliable`: read them as
"check", not as "wrong".

The bead-overlap score counts beads within `qc_settings.score_centroid_mask_radius` (6 voxels)
and quantizes at ~1/N beads: it cannot see misregistration below that radius, and score
differences of one bead are noise.
