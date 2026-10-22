"""Check a QC config's `channels:` against the store it will actually QC.

    python check_qc_channels.py <RAW_STORE.zarr> <OUTPUT>/configs

Exits 1 and names every miss, 0 if every filter resolves.

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


if __name__ == "__main__":
    raw, cfgs = sys.argv[1], Path(sys.argv[2])
    checks = [
        (
            "qc.yaml",
            assembled_channels(raw, cfgs / "reconstruct.yml", cfgs / "virtual_stain.yml"),
        )
    ]
    if (cfgs / "qc_track.yaml").exists():
        checks.append(("qc_track.yaml", tracking_channels(cfgs / "track.yml")))
    bad = 0
    for name, predicted in checks:
        print(f"{name}: store will have {predicted}")
        for owner, ch in qc_filters(cfgs / name):
            ok = ch in predicted
            bad += not ok
            print(f"   {'ok  ' if ok else 'MISS'} {owner}.channels: {ch!r}")
    sys.exit(1 if bad else 0)
