# The transforms file

`estimate-transform` writes a `TransformSettings` YAML that `apply-transform` reads.

## Directions

| `direction` | Maps | Used for |
|---|---|---|
| `forward` | moving -> reference | what the engine estimates and writes |
| `inverse` | reference -> moving | what resampling uses; every legacy file (`registration_settings.yml`, `approx_transform`, `*_stabilization_settings.yml`) |

`inverse = forward^-1` (ANTs calls them the forward and inverse transforms). Every file
declares its direction and the code converts on demand, so mixing files is safe. To compare
matrices by hand, bring both to the same direction first. `transform.seed_direction`
(default `inverse`) says how the seed in an estimate config is written. Files written with
the old name `pull` load as `inverse`.

## Layout

```yaml
direction: forward
method: beads
moving_channels: [mCherry EX561 EM600-37]   # the estimation channel (provenance)
reference_channel: Phase3D                   # null for stabilization
transforms:                                  # one list shared by every position ...
  - {t: 0, matrix: [...], score: 0.71, status: accepted}
  - {t: 1, matrix: [...], score: null, status: unreliable, filled_from: seed,
     note: "EstimationError: too few matches ..."}
# positions:                                 # ... or a list per position (stabilization)
#   C/2/000000: [ {t: 0, ...}, ... ]
```

- Within a list, entries with `t` are per timepoint; a single entry without `t` applies to
  every timepoint (`estimated_at` says where it came from).
- `status`: `accepted`, or `unreliable` (no good transform: the pipeline's best result, or a
  stand-in when `filled_from` is set -- `seed` = the approximate transform, `identity` = a
  missing step of a `previous` chain), or `rejected` (decided by a person). A timepoint is
  never dropped; `apply-transform` writes every one and records the non-accepted ones in
  the output metadata.
- `repaired_from`: which repair candidate won (e.g. `consensus_full+polish1`).
- `method` on an entry: set when it came from another method's run
  (`substitute-transforms`).

## Old configs

`biahub convert-settings -c old.yml -o new.yml` converts any retired estimate / register /
stabilize config; several per-position stabilize files fold into one
(`-c "xyz_stabilization_settings/*.yml"`). The deprecated command names still run and
convert on the fly.
