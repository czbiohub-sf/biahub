# Trackers in `biahub track`

Select with `tracker:` in the track config. All four read the cellpose labels (`segmentation_method:
cellpose`); ultrack can also use `foreground_contour`. The other three need cellpose labels and
`output_mode: 2D` (refused at `--init` otherwise).

| tracker | how it links | settings block | runs on | needs |
|---|---|---|---|---|
| `ultrack` (default) | re-segments a hierarchy from the labels, then an ILP picks segments and links | `tracking_config` (ultrack's MainConfig) | CPU | nothing extra |
| `tracksdata` | region nodes, distance + IoU edges, ILP | `tracksdata_config` | CPU | ILP solver |
| `hoct` | learned edge scores (HOCT's model), then an ILP; edges may skip frames | `hoct_config` | GPU if present | model weights, ILP solver |
| `trackastra` | a transformer predicts links; greedy or ILP | `trackastra_config` | GPU if present | model weights (ILP mode: solver) |

## Settings

Distances are in um, converted with the plate's pixel size. Defaults are the A549 benchmark's
pixel settings at 0.1133 um/px (mantis v2).

- **tracksdata**: `distance_threshold_um` (candidate edges, 22.66), `n_neighbors` (3),
  `distance_norm_um` (edge weight `-IoU * exp(distance / norm)`, 11.33), `node_weight` (-2),
  `appearance_weight` (5), `disappearance_weight` (5), `division_weight` (1),
  `border_margin_um` (6.80) and `border_appearance_weight` (-1: appearing at the image edge is
  cheap). Raising `division_weight` allows fewer divisions; edge rewards are in [-1, 0], so
  weights on the scale of 10-100 suppress divisions entirely.
- **hoct**: `model` (`general_v1`, or a path), `distance_threshold_um` (33.99), `n_neighbors`
  (5), `max_delta_t` (3: edges may skip up to 2 frames), `solver` (HOCT's `ILPSolverConfig`
  fields). HOCT runs in pixel space, as its models were trained.
- **trackastra**: `model` (`general_2d`, `ctc`, or a trained model folder) and `mode`
  (`greedy`, `greedy_nodiv` for no divisions, `ilp`). No distance settings.
- **ultrack**: `tracking_config` with `segmentation_config` (`min_area`, `max_area`,
  `min_frontier`), `linking_config` (`max_distance` in um, `distance_weight`, `max_neighbors`)
  and `tracking_config` (appear / disappear / division weights).

## What the A549 benchmark found

Same cellpose labels and config (60-plane window, `min_sibling_length: 3`), untuned defaults
for the three new trackers. 2026_04_28: 4 TrackEdit FOVs, 78 divisions (division F1, with
precision / recall). 2026_04_21: 5 DENV FOVs, 38 divisions from mitosis labels.

| tracker | 04_28 F1 | precision | recall | mock F1 | DENV F1 | tracks (GT 287) | 04_21 divisions found |
|---|---|---|---|---|---|---|---|
| ultrack | 0.68 | 0.60 | 0.79 | 0.70 | 0.63 | 376 | 19 |
| trackastra | 0.63 | 0.71 | 0.57 | 0.65 | 0.60 | 355 | 14 |
| hoct | 0.53 | 0.68 | 0.43 | 0.59 | 0.47 | 311 | 7 |
| tracksdata | 0.52 | 0.47 | 0.57 | 0.48 | 0.55 | 331 | 17 |

- ultrack is the default: best F1 and recall, and the best tuned.
- Trackastra and HOCT call fewer divisions with higher precision (Trackastra 0.82 in mock
  wells): good when false divisions cost more than missed ones, and as independent opinions.
- tracksdata is the least accurate with these defaults and needs tuning (`division_weight`,
  `appearance_weight`); HOCT fragments least (311 tracks).
- Each new tracker takes ~13-16 min for 4-5 A549 FOVs on one GPU node (cellpose included),
  ~11 GB peak.

## Known pitfalls

- **ILP solver**: tracksdata, HOCT and Trackastra's `ilp` mode use Gurobi when a licence is
  available, otherwise SCIP. Personal Gurobi licences do not work on the GPU nodes; the log
  shows `Solver failed with Gurobi, trying Scip.` That is expected, not an error; SCIP is
  slower (tracksdata: ~14 min for 4 A549 FOVs).
- **Model weights** download on first use: `--init` fetches them once (HOCT to its pooch
  cache, Trackastra to `~/.local/share/trackastra/models`). A worker without network needs
  them cached first.
- **One-child links**: HOCT links across skipped frames; biahub merges a track that is its
  parent's only child back into the parent (the sibling filter requires 0 or 2 children).
- **2D only** for tracksdata, HOCT and Trackastra.
- Jobs belong on `gpu`, not `preempted`: tracking does not resume.
