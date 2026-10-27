"""`biahub evaluate-transform`: how well a transforms file registers, in µm.

Evaluates the file's transform -- and, when its run folder holds a stack's records, every
block's -- on the given positions with `biahub.registration.evaluate`, against the
unregistered pair, into one table (CSV). Independent of the score the estimate optimised.
"""

from __future__ import annotations

import csv

from pathlib import Path

import click

from biahub.cli.parsing import (
    config_filepath,
    moving_position_dirpaths,
    output_filepath,
    position_key,
    reference_position_dirpaths,
)
from biahub.registration.evaluate import block_candidates, evaluate_candidates
from biahub.settings import load_transform_settings


def _file_candidate(transforms, position_keys: list[str], t: int) -> dict:
    """Return the file's forward transform at `t`, per position (a per-position file)."""
    if transforms.per_position:
        return {(key, t): transforms.transform_for(t, key) for key in position_keys}
    return {t: transforms.transform_for(t)}


def _bead_detection(run_dir: Path, threshold: float | None) -> dict:
    """Return bead detection settings: `threshold` for both, else the estimate's own."""
    if threshold is not None:
        return {"threshold_abs": threshold}
    from biahub.settings import load_estimate_transform_settings

    settings_file = run_dir / "estimate_transform_settings.yml"
    final = None
    if settings_file.exists():
        final = load_estimate_transform_settings(settings_file).stack_settings()[-1]
    if final is None or final.bead_matching is None:
        raise click.UsageError(
            "--landmarks beads needs --bead-threshold, or a run folder whose settings "
            "detected beads"
        )

    def peaks(p):  # the estimate's detection, so the same beads are found
        return {
            "threshold_abs": p.threshold_abs,
            "nms_distance": p.nms_distance,
            "block_size": tuple(p.block_size),
        }

    return {
        "moving": peaks(final.bead_matching.source_peaks_settings),
        "reference": peaks(final.bead_matching.target_peaks_settings),
    }


def evaluate_transform(
    moving_position_dirpaths: list[Path],
    reference_position_dirpaths: list[Path],
    transforms_filepath: Path,
    output_filepath: Path,
    moving_channel: str | None = None,
    reference_channel: str | None = None,
    landmarks: str | None = None,
    t: int = 0,
    blocks: bool = True,
    figures: bool = False,
    tags: dict | None = None,
    bead_threshold: float | None = None,
    reference_t: int | None = None,
    **image_kwargs,
) -> list[dict]:
    """Evaluate a transforms file (and its stack's blocks) on positions; write the table.

    `tags` become constant columns, first (e.g. the case and pipeline of a benchmark), so
    tables from several runs can be joined.
    """
    transforms = load_transform_settings(transforms_filepath)
    keys = [position_key(p) for p in moving_position_dirpaths]
    # positions the file has no list for (held out from a per-position estimate): each
    # estimated position's transform is a candidate there, named after it
    held_out = transforms.per_position and not set(keys) <= set(transforms.positions)
    if held_out:
        candidates = {
            f"transforms ({pos})": {t: transforms.transform_for(t, pos)}
            for pos in transforms.positions
        }
    else:
        candidates = {"transforms": _file_candidate(transforms, keys, t)}
    run_dir = Path(transforms_filepath).with_suffix("")
    if blocks and run_dir.is_dir():
        for name, by_key in block_candidates(run_dir).items():
            if held_out:
                for key, transform in by_key.items():
                    if isinstance(key, tuple) and key[1] == t:
                        candidates[f"{name} ({key[0]})"] = {t: transform}
            else:
                candidates[name] = by_key
    output_filepath = Path(output_filepath)
    detect_kwargs = _bead_detection(run_dir, bead_threshold) if landmarks == "beads" else None
    rows = evaluate_candidates(
        candidates,
        list(zip(moving_position_dirpaths, reference_position_dirpaths, strict=True)),
        moving_channel or transforms.moving_channels[0],
        # a stabilization has no reference channel: its reference is the moving channel
        reference_channel or transforms.reference_channel or transforms.moving_channels[0],
        t=t,
        landmarks=landmarks,
        detect_kwargs=detect_kwargs,
        reference_t=reference_t,
        figures_dir=output_filepath.with_suffix("") if figures else None,
        **image_kwargs,
    )
    rows = [{**(tags or {}), **row} for row in rows]
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _parse_tags(tags) -> dict:
    parsed = {}
    for tag in tags:
        name, sep, value = tag.partition("=")
        if not sep or not name:
            raise click.BadParameter(f"a tag is name=value; got {tag!r}", param_hint="--tag")
        parsed[name] = value
    return parsed


@click.command("evaluate-transform")
@moving_position_dirpaths()
@reference_position_dirpaths()
@config_filepath()
@output_filepath()
@click.option(
    "--moving-channel", default=None, help="Default: the file's first moving channel."
)
@click.option(
    "--reference-channel", default=None, help="Default: the file's reference channel."
)
@click.option(
    "--landmarks",
    type=click.Choice(["nuclei", "beads", "none"]),
    default="none",
    show_default=True,
    help="Point-based residuals from detected nuclei or beads (z is read from points).",
)
@click.option(
    "--t", "t", type=int, default=0, show_default=True, help="Timepoint to evaluate."
)
@click.option(
    "--reference-t",
    type=int,
    default=None,
    help="Read the reference at this timepoint (a stabilization's reference frame, e.g. 0); "
    "default: --t.",
)
@click.option(
    "--tile-px", type=int, default=256, show_default=True, help="Tile size (pixels)."
)
@click.option(
    "--no-blocks", is_flag=True, help="Only the file's transform, not each block of its stack."
)
@click.option(
    "--figures",
    is_flag=True,
    help="Save a figure per candidate and position next to the table.",
)
@click.option(
    "--bead-threshold",
    type=float,
    default=None,
    help="With --landmarks beads: the detection threshold (default: the estimate's).",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="name=value, a constant column (e.g. --tag case=dragonfly); repeat for several.",
)
def evaluate_transform_cli(
    moving_position_dirpaths,
    reference_position_dirpaths,
    config_filepath,
    output_filepath,
    moving_channel,
    reference_channel,
    landmarks,
    t,
    reference_t,
    tile_px,
    no_blocks,
    figures,
    bead_threshold,
    tags,
) -> None:
    """Evaluate a transforms file on positions: residuals in µm, against unregistered.

    One row per candidate (the file's transform, each block of its stack, unregistered)
    and position: tile and point medians, residual scale and rotation, z and where it
    came from. Pass held-out positions to see whether the transform carries.

    >> biahub evaluate-transform -m moving.zarr/A/1/0 -r reference.zarr/A/1/0 -c out/transforms.yml -o out/evaluation.csv --landmarks nuclei
    """
    rows = evaluate_transform(
        moving_position_dirpaths,
        reference_position_dirpaths,
        config_filepath,
        output_filepath,
        moving_channel=moving_channel,
        reference_channel=reference_channel,
        landmarks=None if landmarks == "none" else landmarks,
        t=t,
        blocks=not no_blocks,
        figures=figures,
        tags=_parse_tags(tags),
        bead_threshold=bead_threshold,
        reference_t=reference_t,
        tile_px=tile_px,
    )
    for row in rows:
        if row["empty"]:
            click.echo(
                f"{row['position']}: WARNING the {row['empty']} image is blank, skipped"
            )
            continue
        tiles = row["tiles_median_um"]
        click.echo(
            f"{row['position']} {row['candidate']:>20}: tiles "
            + ("n/a" if tiles is None else f"{tiles:.2f} µm")
            + (
                f", z {row['z_um']:+.2f} µm ({row['z_from']})"
                if row["z_um"] is not None
                else ""
            )
        )
    click.echo(f"Evaluation saved to {Path(output_filepath).resolve()}")
