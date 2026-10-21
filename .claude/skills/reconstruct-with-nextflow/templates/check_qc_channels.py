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
3. `segment.yml`'s model channels against the plate segment reads (the
   projected plate when `project.yml` exists, else the assembled plate), and
   `track.yml`'s `labels` against `segment.yml`'s model names, since track reads
   the label image segment wrote (`labels/<model>`).

WHY PREDICT RATHER THAN READ. The store QC runs over — the assembled plate —
does not exist when configs are scaffolded; the pipeline's init phase creates
it. So this derives its channel list from the configs that will produce it,
which is exactly the coupling that goes wrong:

    assembled = raw channels (flat-field and deskew preserve names)
              + reconstruct's output_channel_names (waveorder derives these,
                so they are read from the settings model, not the YAML text)
              + virtual-stain's data.init_args.target_channel
    labels    = <segment.yml model>           (<fov>/labels/<model>)
              + tracks_<track.yml labels>    (<fov>/labels/tracks_<model>)

Segment and track write their labels INTO the plate they read, so
`qc_project.yaml`'s `instance_count.label` only resolves if it is
`tracks_<track.yml labels>` (or a `segment.yml` model name), and its one anchor
channel, the channel the labels were segmented from, must be that model's
channel. Four files, no shared source of truth. Checked when `qc_project.yaml`
has an `instance_count` block and `track.yml` exists.

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


def instance_count_entries(qc_cfg):
    """`instance_count`'s label and anchor channels, or None without such a block."""
    ic = (yaml.safe_load(Path(qc_cfg).read_text()) or {}).get("instance_count")
    if not isinstance(ic, dict):
        return None
    entries = [("instance_count.label", ic.get("label"))]
    entries += [("instance_count.channels", c) for c in ic.get("channels") or []]
    return entries


def projected_channels(project_cfg):
    """Return the channels `biahub project` writes, in output order (the config's keys)."""
    return list(yaml.safe_load(Path(project_cfg).read_text())["channels"])


def segment_models(segment_cfg):
    """Return {model name: [channels]} of a segment config: the label images it writes."""
    models = yaml.safe_load(Path(segment_cfg).read_text())["models"]
    return {name: list(model["channels"]) for name, model in models.items()}


def tracked_label(track_cfg):
    """Return (input label, output label) of a track config, or None on the legacy path."""
    t = yaml.safe_load(Path(track_cfg).read_text())
    if t.get("segmentation_method", "labels") != "labels":
        return None
    return t["labels"], t.get("output_label") or f"tracks_{t['labels']}"


if __name__ == "__main__":
    if sys.argv[1:2] in (["-h"], ["--help"]):
        sys.exit(print(__doc__))
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("\n\n")[1])
    raw, cfgs = sys.argv[1], Path(sys.argv[2])
    bad = 0

    assembled = assembled_channels(raw, cfgs / "reconstruct.yml", cfgs / "virtual_stain.yml")
    print(f"qc.yaml: store will have {assembled}")
    for owner, ch in qc_filters(cfgs / "qc.yaml"):
        ok = ch in assembled
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
    source = "projected" if (cfgs / "project.yml").exists() else "assembled"

    # segment.yml reads the plate project (or assemble) writes and adds labels/<model>.
    models = {}
    if (cfgs / "segment.yml").exists():
        models = segment_models(cfgs / "segment.yml")
        print(f"segment.yml: {source} plate will have {track_reads}")
        for name, channels in models.items():
            for ch in channels:
                ok = ch in track_reads
                bad += not ok
                print(f"   {'ok  ' if ok else 'MISS'} models.{name}.channels: {ch!r}")

    # track.yml reads labels/<labels> of that plate and writes labels/tracks_<labels>.
    tracked = None
    if (cfgs / "track.yml").exists():
        tracked = tracked_label(cfgs / "track.yml")
        if tracked:
            print(f"track.yml: segment.yml writes labels/{{{', '.join(models) or '-'}}}")
            ok = tracked[0] in models
            bad += not ok
            print(f"   {'ok  ' if ok else 'MISS'} labels: {tracked[0]!r}")

    # qc_project.yaml QCs the plate segment and track write into: its `channels:`
    # against that plate, and its `instance_count` against the label images there
    # (the tracked one, labels/tracks_<track.yml labels>, or a segment model's) with
    # the anchor channel being the one the model segmented.
    if (cfgs / "qc_project.yaml").exists():
        print(f"qc_project.yaml: {source} plate will have {track_reads}")
        for owner, ch in qc_filters(cfgs / "qc_project.yaml"):
            ok = ch in track_reads
            bad += not ok
            print(f"   {'ok  ' if ok else 'MISS'} {owner}.channels: {ch!r}")
        entries = instance_count_entries(cfgs / "qc_project.yaml")
        if entries and (models or tracked):
            label_images = dict(models)
            if tracked and tracked[0] in models:
                label_images[tracked[1]] = models[tracked[0]]
            print(f"qc_project.yaml vs segment.yml/track.yml: label images {sorted(label_images)}")
            label = dict(entries).get("instance_count.label")
            for key, value in entries:
                if key == "instance_count.label":
                    ok = value in label_images
                else:
                    ok = value in label_images.get(label, [])
                bad += not ok
                print(f"   {'ok  ' if ok else 'MISS'} {key}: {value!r}")
    sys.exit(1 if bad else 0)
