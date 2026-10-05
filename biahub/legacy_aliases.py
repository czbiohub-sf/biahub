"""Deprecated registration command names, kept as thin aliases of the unified commands.

`estimate-registration` and `estimate-stabilization` run `estimate-transform`;
`register` and `stabilize` run `apply-transform`. Each converts a retired config on the
fly (what `biahub convert-settings` does), warns that the name is deprecated, and writes
the new transforms file. `optimize-registration` has no direct equivalent.
"""

from __future__ import annotations

from pathlib import Path

import click

from biahub.apply_transform import apply_transform
from biahub.cli.option_eat_all import OptionEatAll
from biahub.cli.parsing import (
    _validate_and_process_paths,
    config_filepath,
    config_filepaths,
    input_position_dirpaths,
    local,
    monitor,
    output_dirpath,
    output_filepath,
    sbatch_filepath,
)
from biahub.convert_settings import (
    convert_settings,
    load_legacy_settings,
    per_position_stabilization,
)
from biahub.estimate_transform import estimate_transform
from biahub.settings import TransformSettings, load_transform_settings
from biahub.utils.config import model_to_yaml


def _warn(old: str, new: str) -> None:
    click.secho(
        f"DeprecationWarning: `biahub {old}` is deprecated and will be removed in a future "
        f"release; it now runs `biahub {new}`. Convert your config once with "
        "`biahub convert-settings -c <old>.yml -o <new>.yml` and call the new command.",
        fg="yellow",
        err=True,
    )


def _positions(flag: str, long: str, help: str) -> click.Option:
    return click.option(
        long,
        flag,
        required=True,
        cls=OptionEatAll,
        type=tuple,
        callback=_validate_and_process_paths,
        help=help,
    )


def _converted_estimate_config(config_filepath: Path, next_to: Path) -> Path:
    """Write the unified version of a retired estimate-* config next to the output."""
    settings, notes = convert_settings(load_legacy_settings(config_filepath))
    path = next_to.parent / f"{Path(config_filepath).stem}.estimate_transform.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_yaml(settings, path)
    for note in notes:
        click.echo(f"  note: {note}", err=True)
    return path


def _transforms_config(config_filepaths: list[Path], next_to: Path) -> tuple[Path, dict]:
    """Return a unified transforms file for register / stabilize and the legacy apply options.

    Accepts a new transforms file as is, a retired register / stabilize config (converted),
    or several per-position stabilize configs (folded into one file per position).
    """
    paths = [Path(p) for p in config_filepaths]
    if len(paths) == 1:
        try:
            load_transform_settings(paths[0])
            return paths[0], {}
        except ValueError:
            pass
    legacy = [load_legacy_settings(p) for p in paths]
    options = {
        key: getattr(legacy[0], key)
        for key in ("time_indices", "keep_overhang", "interpolation")
        if hasattr(legacy[0], key)
    }
    if len(paths) == 1:
        settings, notes = convert_settings(legacy[0])
    else:
        settings, notes = per_position_stabilization(paths)
    assert isinstance(settings, TransformSettings)
    path = next_to.parent / f"{paths[0].stem}.transforms.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_yaml(settings, path)
    return path, options


@click.command("estimate-registration", hidden=True)
@_positions("-s", "--source-position-dirpaths", "Moving (source) positions.")
@_positions("-t", "--target-position-dirpaths", "Reference (target) positions.")
@output_filepath()
@config_filepath()
@sbatch_filepath()
@local()
@click.option("--registration-target-channel", "-rt", default=None, hidden=True)
@click.option("--registration-source-channels", "-rs", multiple=True, hidden=True)
def estimate_registration_alias(
    source_position_dirpaths, target_position_dirpaths, output_filepath, config_filepath,
    sbatch_filepath, local, registration_target_channel, registration_source_channels,
):
    """Run `estimate-transform` on a converted estimate-registration config (deprecated)."""
    _warn("estimate-registration", "estimate-transform")
    if registration_target_channel or registration_source_channels:
        click.echo(
            "  note: -rt/-rs ignored; choose the channels to transform with "
            "`apply-transform --channels`",
            err=True,
        )
    output = Path(output_filepath)
    estimate_transform(
        source_position_dirpaths,
        _converted_estimate_config(config_filepath, output),
        output,
        reference_position_dirpaths=target_position_dirpaths,
        sbatch_filepath=sbatch_filepath,
        cluster="local" if local else "slurm",
    )


@click.command("estimate-stabilization", hidden=True)
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
def estimate_stabilization_alias(
    input_position_dirpaths, output_dirpath, config_filepath, sbatch_filepath, local
):
    """Run `estimate-transform` on a converted estimate-stabilization config (deprecated)."""
    _warn("estimate-stabilization", "estimate-transform")
    output = Path(output_dirpath) / "transforms.yml"
    estimate_transform(
        input_position_dirpaths,
        _converted_estimate_config(config_filepath, output),
        output,
        sbatch_filepath=sbatch_filepath,
        cluster="local" if local else "slurm",
    )


@click.command("register", hidden=True)
@_positions("-s", "--source-position-dirpaths", "Moving (source) positions.")
@_positions("-t", "--target-position-dirpaths", "Reference (target) positions.")
@config_filepath()
@output_dirpath()
@local()
@sbatch_filepath()
@monitor()
def register_alias(
    source_position_dirpaths, target_position_dirpaths, config_filepath, output_dirpath,
    local, sbatch_filepath, monitor,
):
    """Run `apply-transform` onto the target grid (deprecated)."""
    _warn("register", "apply-transform")
    config, options = _transforms_config([config_filepath], Path(output_dirpath))
    apply_transform(
        source_position_dirpaths,
        config,
        output_dirpath,
        reference_position_dirpaths=target_position_dirpaths,
        time_indices=options.get("time_indices", "all"),
        keep_overhang=options.get("keep_overhang", False),
        interpolation=options.get("interpolation", "linear"),
        sbatch_filepath=sbatch_filepath,
        cluster="local" if local else "slurm",
        monitor=monitor,
    )


@click.command("stabilize", hidden=True)
@input_position_dirpaths()
@output_dirpath()
@config_filepaths()
@sbatch_filepath()
@local()
@monitor()
def stabilize_alias(
    input_position_dirpaths, output_dirpath, config_filepaths, sbatch_filepath, local, monitor
):
    """Run `apply-transform` without a reference, onto each store's own grid (deprecated)."""
    _warn("stabilize", "apply-transform")
    config, options = _transforms_config(config_filepaths, Path(output_dirpath))
    apply_transform(
        input_position_dirpaths,
        config,
        output_dirpath,
        time_indices=options.get("time_indices", "all"),
        # legacy stabilize kept the full input grid
        keep_overhang=True,
        sbatch_filepath=sbatch_filepath,
        cluster="local" if local else "slurm",
        monitor=monitor,
    )


@click.command("optimize-registration", hidden=True)
@click.option("--source-position-dirpaths", "-s", multiple=True, hidden=True)
@click.option("--target-position-dirpaths", "-t", multiple=True, hidden=True)
@click.option("--config-filepath", "-c", hidden=True)
@click.option("--output-filepath", "-o", hidden=True)
@click.option("--display-viewer", "-d", is_flag=True, hidden=True)
def optimize_registration_alias(**_legacy_options):
    """Explain that this command was removed (old flags accepted so the message is reached)."""
    raise click.ClickException(
        "`biahub optimize-registration` has been removed. Refine a registration with "
        "`biahub estimate-transform` using method `ants` (intensity-based) or `manual` "
        "(point annotation in napari)."
    )
