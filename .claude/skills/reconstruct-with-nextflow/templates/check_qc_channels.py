"""Check the channel couplings between the step configs and the stores they read.

    python check_qc_channels.py <RAW_STORE.zarr> <OUTPUT>/configs

Exits 1 and names every miss, 0 if every filter resolves. Three couplings:

1. Each QC config's `channels:` against the store it will QC.
2. `project.yml` (when present) against the assembled plate, both ways: a listed
   channel the plate will not have is a MISS (`biahub project --init` refuses
   it); an assembled channel the config does not list is printed as
   NOT PROJECTED. That one is a warning, not a failure -- the CLI drops the
   channel with a warning -- but the pipeline's intention is that every
   assembled channel is projected unless the user decided otherwise, so the
   skill carries these lines into the pre-run plan for the user to decide.
3. `track.yml`'s `target_channel` and `cellpose_config.input_channel` against
   the plate track reads: the projected plate when `project.yml` exists, else
   the assembled plate.

WHY PREDICT RATHER THAN READ. The stores QC runs over — the assembled plate and
the tracking plate — do not exist when configs are scaffolded; the pipeline's
init phase creates them. So this derives their channel lists from the configs
that will produce them, which is exactly the coupling that goes wrong:

    assembled = raw channels (flat-field and deskew preserve names)
              + reconstruct's output_channel_names (waveorder derives these,
                so they are read from the settings model, not the YAML text)
              + virtual-stain's data.init_args.target_channel
    tracking  = ["<track.yml target_channel>_labels"]

`qc_track.yaml` naming `nuclei_prediction_labels` only matches if `track.yml`
says `target_channel: nuclei_prediction`. Two files, no shared source of truth.

RELATIONSHIP TO imaging-qc#226. Since that landed, `plan-stage` refuses an
unknown channel name itself, so the pipeline fails at init rather than skipping
a metric silently. This runs EARLIER — before launch, while the configs are
being edited — so the mistake is fixed instead of costing a launch. It is a
convenience, not the safety net; the safety net is in the pipeline.

Assumes `concatenate.yml` takes `all` from each source, which both shipped
families do. A config that selects a channel subset will over-predict, so a
`MISS` is authoritative but a pass is not a proof.
"""

import sys

from pathlib import Path

import yaml

from iohub import open_ome_zarr


def qc_filters(cfg_path):
    """Every `channels:` list anywhere in a QC config, with the metric that owns it."""
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    found = []

    def walk(node, owner):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "channels" and isinstance(v, list):
                    found.extend((owner, c) for c in v if isinstance(c, str))
                else:
                    walk(v, k if isinstance(v, dict) else owner)
        elif isinstance(node, list):
            for v in node:
                walk(v, owner)

    walk(cfg, "<root>")
    return found


def assembled_channels(raw_zarr, reconstruct_cfg, virtual_stain_cfg):
    with open_ome_zarr(str(raw_zarr), mode="r") as s:
        _, fov = next(iter(s.positions()))
        raw = list(fov.channel_names)  # flat-field and deskew preserve names
    from waveorder.cli.settings import ReconstructionSettings

    rc = ReconstructionSettings(**yaml.safe_load(Path(reconstruct_cfg).read_text()))
    vs = yaml.safe_load(Path(virtual_stain_cfg).read_text())
    tgt = vs["data"]["init_args"]["target_channel"]
    tgt = [tgt] if isinstance(tgt, str) else list(tgt)
    return raw + list(rc.output_channel_names) + tgt


def tracking_channels(track_cfg):
    t = yaml.safe_load(Path(track_cfg).read_text())
    return [f"{t['target_channel']}_labels"]


def projected_channels(project_cfg):
    """Return the channels `biahub project` writes, in output order (the config's keys)."""
    return list(yaml.safe_load(Path(project_cfg).read_text())["channels"])


def track_inputs(track_cfg):
    """Return the channel names track reads from its input plate, keyed by the setting."""
    t = yaml.safe_load(Path(track_cfg).read_text())
    inputs = [("target_channel", t["target_channel"])]
    cellpose = t.get("cellpose_config") or {}
    if t.get("segmentation_method") == "cellpose" and "input_channel" in cellpose:
        inputs.append(("cellpose_config.input_channel", cellpose["input_channel"]))
    return inputs


if __name__ == "__main__":
    raw, cfgs = sys.argv[1], Path(sys.argv[2])
    assembled = assembled_channels(raw, cfgs / "reconstruct.yml", cfgs / "virtual_stain.yml")
    checks = [("qc.yaml", assembled)]
    if (cfgs / "qc_track.yaml").exists():
        checks.append(("qc_track.yaml", tracking_channels(cfgs / "track.yml")))
    bad = 0
    for name, predicted in checks:
        print(f"{name}: store will have {predicted}")
        for owner, ch in qc_filters(cfgs / name):
            ok = ch in predicted
            bad += not ok
            print(f"   {'ok  ' if ok else 'MISS'} {owner}.channels: {ch!r}")

    # project.yml reads the assembled plate and writes the plate track reads.
    track_reads = assembled
    if (cfgs / "project.yml").exists():
        projected = projected_channels(cfgs / "project.yml")
        track_reads = projected
        print(f"project.yml: assembled plate will have {assembled}")
        for ch in projected:
            ok = ch in assembled
            bad += not ok
            print(f"   {'ok  ' if ok else 'MISS'} channels: {ch!r}")
        for ch in assembled:
            if ch not in projected:
                # Not a failure: the CLI drops it with a warning. The skill asks the
                # user about every such channel in the pre-run plan.
                print(f"   NOT PROJECTED {ch!r}: not in project.yml, will be dropped")
    if (cfgs / "track.yml").exists():
        source = "projected" if (cfgs / "project.yml").exists() else "assembled"
        print(f"track.yml: {source} plate will have {track_reads}")
        for key, ch in track_inputs(cfgs / "track.yml"):
            ok = ch in track_reads
            bad += not ok
            print(f"   {'ok  ' if ok else 'MISS'} {key}: {ch!r}")
    sys.exit(1 if bad else 0)
