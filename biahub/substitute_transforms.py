"""Substitute the transforms of chosen timepoints with ones from another run.

A beads run that fails at a few timepoints can be completed with transforms estimated
there by another method (manual, ants, ...): estimate only those timepoints, then put
them in place of the base run's entries. Each substituted entry records the method it
came from, so the combined file says where every transform came from.

>> biahub estimate-transform ... -c beads.yml -o beads/transforms.yml
>> biahub estimate-transform ... -c manual.yml -o t82/transforms.yml   # time_indices: 82
>> biahub substitute-transforms -c beads/transforms.yml -s t82/transforms.yml -o final.yml
"""

from __future__ import annotations

from pathlib import Path

import click

from biahub.cli.parsing import config_filepath, output_filepath
from biahub.settings import TransformEntry, TransformSettings, load_transform_settings
from biahub.utils.config import model_to_yaml


def _timepoint_of(entry: TransformEntry, source: str) -> int:
    """Return the timepoint a substitute entry replaces: its `t`, else where it was estimated."""
    if entry.t is not None:
        return entry.t
    if entry.estimated_at is not None:
        return entry.estimated_at
    raise click.UsageError(
        f"{source}: an entry without t or estimated_at does not say which timepoint it "
        "replaces; estimate the substitute with time_indices set to that timepoint"
    )


def _substitute_list(
    base_entries: list[TransformEntry],
    sub_entries: list[TransformEntry],
    base: TransformSettings,
    sub: TransformSettings,
    source: str,
    where: str,
    allow_new: bool,
    log: list[str],
) -> list[TransformEntry]:
    if base_entries[0].t is None:
        raise click.UsageError(
            f"{where}: the base holds one transform for every timepoint; there is no "
            "per-timepoint entry to substitute"
        )
    by_t = {e.t: e for e in base_entries}
    for entry in sub_entries:
        t = _timepoint_of(entry, source)
        if t not in by_t and not allow_new:
            raise click.UsageError(
                f"{where}: timepoint {t} from {source} is not in the base "
                f"(it has {min(by_t)}..{max(by_t)}); pass --allow-new to add it"
            )
        old = by_t.get(t)
        method = entry.method or sub.method
        by_t[t] = TransformEntry(
            t=t,
            matrix=sub._as(entry.matrix, base.direction).tolist(),  # in the base's direction
            score=entry.score,
            repaired_from=entry.repaired_from,
            method=method if method != base.method else None,
            # a stand-in stays one, so apply-transform still reports it
            status=entry.status,
            filled_from=entry.filled_from,
            note=entry.note,
        )
        before = (
            "new"
            if old is None
            else f"{old.method or base.method} (score {old.score if old.score is not None else 'n/a'})"
        )
        log.append(
            f"{where} t={t}: {before} -> {method} (score {entry.score if entry.score is not None else 'n/a'}) from {source}"
        )
    return [by_t[t] for t in sorted(by_t)]


def substitute_transforms(
    base: TransformSettings,
    substitutes: list[tuple[str, TransformSettings]],
    positions: list[str] | None = None,
    allow_new: bool = False,
) -> tuple[TransformSettings, list[str]]:
    """Put each substitute's entries in place of the base's at the same timepoints.

    Substitutes are applied in order (a later one wins on a shared timepoint) and always
    win where given: scores of different methods are on different scales, so the choice
    is the caller's. `positions` picks which positions of a per-position base a shared
    substitute applies to. Returns the combined settings and one log line per entry.
    """
    log: list[str] = []
    transforms = list(base.transforms) if base.transforms is not None else None
    per_position = (
        {k: list(v) for k, v in base.positions.items()} if base.positions is not None else None
    )
    for name, model in (("the base", base), *substitutes):
        if model.reference_frame == "previous":
            raise click.UsageError(
                f"{name} was estimated with reference frame 'previous': its matrices are "
                "chained onto the first frame, so a substitute at one timepoint cannot fix "
                "the later ones chained through it, and a one-timepoint 'previous' estimate "
                "holds a single step. Re-estimate with reference frame 'first' instead."
            )
    for source, sub in substitutes:
        if (sub.moving_channels, sub.reference_channel) != (
            base.moving_channels,
            base.reference_channel,
        ):
            raise click.UsageError(
                f"{source}: maps {sub.moving_channels} onto {sub.reference_channel!r}, the "
                f"base maps {base.moving_channels} onto {base.reference_channel!r}"
            )
        if per_position is None:
            if sub.positions is not None:
                raise click.UsageError(
                    f"{source} holds a list per position but the base is one shared list"
                )
            transforms = _substitute_list(
                transforms, sub.transforms, base, sub, source, "transforms", allow_new, log
            )
            continue
        if sub.positions is not None:
            targets = {key: sub.positions[key] for key in (positions or sub.positions)}
        elif positions:
            targets = {key: sub.transforms for key in positions}
        else:
            raise click.UsageError(
                f"the base holds a list per position; say which positions {source} "
                "applies to with --positions"
            )
        for key, entries in targets.items():
            if key not in per_position:
                raise click.UsageError(f"position {key!r} is not in the base")
            per_position[key] = _substitute_list(
                per_position[key], entries, base, sub, source, key, allow_new, log
            )
    combined = base.model_copy(update={"transforms": transforms, "positions": per_position})
    return TransformSettings.model_validate(combined.model_dump()), log


@click.command("substitute-transforms")
@config_filepath()
@click.option(
    "--substitute-filepaths",
    "-s",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Transforms file(s) whose timepoints replace the base's; repeat -s for several, "
    "applied in order.",
)
@output_filepath()
@click.option(
    "--positions",
    multiple=True,
    help="Positions (row/col/fov) of a per-position base that a shared substitute applies "
    "to; repeat for several.",
)
@click.option(
    "--allow-new",
    is_flag=True,
    help="Add substitute timepoints the base does not have instead of refusing them.",
)
def substitute_transforms_cli(
    config_filepath: Path,
    substitute_filepaths: tuple[str, ...],
    output_filepath: str,
    positions: tuple[str, ...],
    allow_new: bool,
) -> None:
    """Replace chosen timepoints of a transforms file with another run's transforms.

    The base (-c) is typically the full-series run; each substitute (-s) holds the
    timepoints redone with another method. The base file is not modified.

    >> biahub substitute-transforms -c beads.yml -s manual_t82.yml -s ants_t140.yml -o final.yml
    """
    base = load_transform_settings(config_filepath)
    substitutes = [(str(p), load_transform_settings(p)) for p in substitute_filepaths]
    combined, log = substitute_transforms(
        base, substitutes, positions=list(positions) or None, allow_new=allow_new
    )
    model_to_yaml(combined, output_filepath)
    for line in log:
        click.echo(line)
    click.echo(f"Combined transforms saved to {Path(output_filepath).resolve()}")
