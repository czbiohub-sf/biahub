# DCA finalize in a mantis-v2 run

`--dca_config` turns on the last step of the run. After QC, `biahub dca-finalize`
writes `attributes.dca` onto every position of the assembled plate, and of the
projected plate when project runs. `dca-validate` then checks each plate against
the DCA v0.2 spec and writes `<stem>.dca-validate.json` beside it. The step is
metadata-only and never cached, so every `-resume` reruns it and nothing else.

The projected plate currently fails validation on its array layout (see § When
finalize fails), so an A549 run with `--dca_config` stops at that plate's finalize
task until `project.yml`'s chunks and shards change. Say so in the plan.

Only A549 has a DCA config (`nextflow/configs/a549/dca_finalize.yml` and
`dca_base.toml`). For neuromast/zebrafish, delete the `--dca_config` line from the
run script and say in the plan that DCA metadata is not written yet.

## The files a run reads

All four live in `<OUTPUT>/configs/`. Both plates read the same confirmed TOML and
sample sheet.

| File | Holds | Filled by |
|---|---|---|
| `dca_base.toml` | experiment-wide values: organism, cell line, disease, development stage, specimen, imaging methods, study description, license, channel catalogue | the template; edit the run copy only where it is wrong for this dataset, and PR the template if the next dataset would need the same fix |
| `<DATASET>.dca-confirmed.toml` | values no source carries: experimenter, release date | the user, every run |
| `<DATASET>.dca-metadata.csv` | one row per well | the user, from the plate map |
| `dca_finalize.yml` | where the other three are | the scaffold |

The run copy of `dca_finalize.yml` names the run's files (relative paths resolve
against the YAML's directory):

```yaml
base_config: dca_base.toml
confirmed_config: <DATASET>.dca-confirmed.toml
metadata_table: <DATASET>.dca-metadata.csv
airtable: null
```

The confirmed TOML, filled from the user's answers. Never invent a name or a
date. A value left out makes finalize stop and list it, which is the intended
behaviour.

```toml
[dca.study]
experimenter = [{ first_name = "<first>", last_name = "<last>", institution = "<institution>" }]
release_date = "<YYYY-MM-DD>"

[dca.provenance.study.experimenter]
source_kind = "human"

[dca.provenance.study.release_date]
source_kind = "human"
```

The sample sheet has these columns, one row per well (`well` is `<row>/<col>`, e.g.
`C/2`). `<BIAHUB>/settings/example_dca_metadata.csv` is a filled example:

```
dataset,well,cell_line,marker,organelle,perturbation,moi,time_interval_min,hours_post_perturbation
```

`dataset` is `<DATASET>` (a blank matches any plate). `perturbation` takes one of
the `control_class` keys in `dca_base.toml`. `time_interval_min` is checked
against the store's time scale. Leave it blank if unknown.

### AirTable instead of a sheet

If the lab keeps the per-well metadata in AirTable, drop `metadata_table` and set:

```yaml
airtable:
  base_id: <base id>
  table: <table>
  token_env: AIRTABLE_API_KEY
```

A sample sheet, when one is configured or sits next to the plate, wins over
AirTable. pyairtable reads a personal access token from the variable named by
`token_env`. The user needs read access to the base (ask a biahub developer) and
a token exported in the shell that launches the run. Slurm passes the launching
shell's environment to the tasks. Check with
`printenv AIRTABLE_API_KEY >/dev/null && echo set || echo MISSING`.

## Show the metadata in the plan

Finalize checks that every value is present and valid against the schema. It
cannot check that a value is true. Show the user the base TOML's `[dca.study]`,
`[dca.biosample]`, `[dca.specimen]` and `[dca.study_component]` blocks, the
confirmed TOML, and the sample sheet as written, and get an explicit yes on them.
A wrong cell line or release date otherwise ships in every position's
attributes.

## When finalize fails

| Symptom | Cause | Fix |
|---|---|---|
| Task exits 1, `Error:` then one line per gap and `fill the missing values in <file>` | a value missing or invalid; nothing was written | fill the named file (or the sheet row) and relaunch the script; `-resume` reruns only finalize |
| `No metadata source` | the sheet path is wrong and no AirTable is set | fix `metadata_table` |
| `QC manifest not found` / `lists position tables for stages ...` | QC did not run on that plate, or ran under two stage numbers | rerun QC, or set `qc_stage` in the run's `dca_finalize.yml` |
| `dca-validate` exits non-zero | the written metadata breaks the spec | read `<stem>.dca-validate.json` beside the plate; fix the base TOML, confirmed TOML or sheet; relaunch |
| `dca-validate` fails on `<DATASET>_proj.zarr` with `Uncompressed chunk size 256 KB is below the required 512 KB minimum` and `Uncompressed shard size ... must be >= 1 GB` | known: `project.yml` writes 256 x 256 single-frame float32 chunks and 200 MB shards, below the DCA v0.2 array standard | not fixable from the DCA files; the metadata is written and the assembled plate is unaffected. Tell the user, and do not edit `project.yml` mid-run (it is in the project task's hash, so `-resume` would recompute project and every step after it) |

None of these is retried: the pipeline's retry rule covers signals and bare
`OSError` tracebacks only, so the run stops at the first failing plate.

## Rerun after the channel rename

Finalize runs inside the pipeline on the pre-rename channel names, and
`rename_channels.py` (§11) renames the channels afterwards. Rerun finalize and the
validator on each plate once the rename is done, from an interactive node:

```bash
for PLATE in <OUTPUT>/4-assemble/<DATASET>.zarr <OUTPUT>/4-assemble/<DATASET>_proj.zarr; do
    biahub dca-finalize -i "${PLATE}"/*/*/* -c <OUTPUT>/configs/dca_finalize.yml
    dca-validate --spec-version 0.2 --show-warnings --output json "${PLATE}" \
        > "${PLATE%.zarr}.dca-validate.json"
done
```

The base TOML's channel catalogue lists both naming sets, so the rerun resolves
the new names. It is metadata-only and safe after cleanup: it reads only the plate,
its `tables/qc/` and the configs.
