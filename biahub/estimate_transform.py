"""Estimate a transform series with the registration engine.

Maps a moving channel onto a reference channel per timepoint and writes a config that
`register` / `stabilize` can apply.
"""

from __future__ import annotations

import json

from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr

from biahub.cli.parsing import (
    config_filepath,
    output_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.core.transform import Transform
from biahub.registration.beads import score_transform
from biahub.registration.estimators import NodeGraphEstimator
from biahub.registration.fallback import neighbour_consensus_config_candidates
from biahub.registration.legacy import forward_from_legacy_pull, legacy_pull_from_forward
from biahub.registration.orchestrator import SeriesResult, estimate_series, repair_series
from biahub.registration.reference_policy import CrossChannel
from biahub.registration.seed_policy import FixedSeed, PreviousSeed
from biahub.settings import (
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)
from biahub.utils.config import model_to_yaml, yaml_to_model


def _resolve_time_indices(time_indices, n_t: int) -> list[int]:
    if time_indices == "all":
        return list(range(n_t))
    if isinstance(time_indices, int):
        return [time_indices]
    return list(time_indices)


def _one_transform_per_timepoint(
    result: SeriesResult, time_indices: list[int], fallback: Transform
) -> list[Transform]:
    """Return one transform per requested timepoint.

    The accepted transform, else the nearest earlier accepted one, else `fallback`.
    """
    out, last = [], fallback
    for t in time_indices:
        last = result.transforms.get(t, last)
        out.append(last)
    return out


def _report(result: SeriesResult, time_indices: list[int]) -> dict:
    return {
        "run_id": result.journal.current_run_id,
        "time_indices": time_indices,
        "scores": {str(t): result.scores[t] for t in time_indices if t in result.scores},
        "errors": {str(t): e for t, e in result.errors.items()},
        "flagged": result.flagged,
        "repairs": {
            str(t): {
                "accepted": r.accepted,
                "source": r.source,
                "score": r.score,
                "candidate_scores": r.scores,
                "candidate_failures": r.failures,
            }
            for t, r in result.repairs.items()
        },
        "filled_from_neighbour": [t for t in time_indices if t not in result.transforms],
    }


def estimate_transform(
    source_position_dirpaths: list[Path],
    target_position_dirpaths: list[Path],
    config_filepath: Path,
    output_filepath: Path,
) -> None:
    """Estimate one transform per timepoint mapping the source channel onto the target.

    Reads an `EstimateRegistrationSettings` YAML (the same file `estimate-registration`
    takes; only `estimation_method: beads` is supported here) and writes a
    `RegistrationSettings` (single timepoint) or `StabilizationSettings` (series) YAML
    next to a `run_journal.json` and an `estimate_transform_report.json`. Runs in-process
    over the first source/target position.
    """
    output_filepath = Path(output_filepath)
    output_dir = output_filepath.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = yaml_to_model(config_filepath, EstimateRegistrationSettings)
    if settings.estimation_method != "beads":
        raise click.UsageError(
            f"estimate-transform supports estimation_method 'beads'; got "
            f"'{settings.estimation_method}'. Use estimate-registration for the others."
        )
    beads_match_settings = settings.beads_match_settings
    affine_transform_settings = settings.affine_transform_settings

    with open_ome_zarr(source_position_dirpaths[0], mode="r") as position:
        mov = position.data.dask_array()[
            :, position.channel_names.index(settings.source_channel_name)
        ]
    with open_ome_zarr(target_position_dirpaths[0], mode="r") as position:
        ref = position.data.dask_array()[
            :, position.channel_names.index(settings.target_channel_name)
        ]
        voxel_size = list(position.scale)

    time_indices = _resolve_time_indices(settings.time_indices, mov.shape[0])
    config_seed = forward_from_legacy_pull(
        affine_transform_settings.approx_transform, affine_transform_settings.transform_type
    )
    history: dict[int, Transform] = {}
    seed_policy = (
        PreviousSeed(history, fallback=FixedSeed(config_seed))
        if affine_transform_settings.use_prev_t_transform
        else FixedSeed(config_seed)
    )
    estimator = NodeGraphEstimator.from_beads_settings(
        beads_match_settings, affine_transform_settings
    )
    reference_policy = CrossChannel(ref)

    def score_fn(transform: Transform, mov_t: np.ndarray, ref_t: np.ndarray) -> float:
        return score_transform(transform, mov_t, ref_t, beads_match_settings)

    def echo_estimate(t: int, result: SeriesResult) -> None:
        line = f"t={t}: score={result.scores[t]:.4f}"
        if t in result.errors:
            line += f"  estimate failed: {result.errors[t]}"
        click.echo(line)

    def echo_repair(t: int, result: SeriesResult) -> None:
        outcome = result.repairs[t]
        click.echo(f"repair t={t}: {outcome.source} -> {outcome.score:.4f}")

    result = estimate_series(
        mov,
        reference_policy,
        estimator,
        seed_policy,
        score_fn,
        time_indices,
        history=history,
        on_timepoint=echo_estimate,
    )
    result = repair_series(
        mov,
        reference_policy,
        estimator,
        score_fn,
        result,
        neighbour_consensus_config_candidates(config_seed),
        on_timepoint=echo_repair,
    )

    transforms = _one_transform_per_timepoint(result, time_indices, config_seed)
    pull_matrices = [legacy_pull_from_forward(transform) for transform in transforms]
    if len(pull_matrices) == 1:
        model = RegistrationSettings(
            source_channel_names=[settings.source_channel_name],
            target_channel_name=settings.target_channel_name,
            affine_transform_zyx=pull_matrices[0],
        )
    else:
        model = StabilizationSettings(
            stabilization_estimation_channel=settings.target_channel_name,
            stabilization_type="affine",
            stabilization_method="beads",
            stabilization_channels=[
                settings.source_channel_name,
                settings.target_channel_name,
            ],
            affine_transform_zyx_list=pull_matrices,
            time_indices=settings.time_indices,
            output_voxel_size=voxel_size,
        )
    model_to_yaml(model, output_filepath)
    result.journal.save(output_dir / "run_journal.json")
    (output_dir / "estimate_transform_report.json").write_text(
        json.dumps(_report(result, time_indices), indent=2)
    )
    click.echo(f"Transform settings saved to {output_filepath.resolve()}")


@click.command("estimate-transform")
@source_position_dirpaths()
@target_position_dirpaths()
@config_filepath()
@output_filepath()
def estimate_transform_cli(
    source_position_dirpaths: list[Path],
    target_position_dirpaths: list[Path],
    config_filepath: Path,
    output_filepath: Path,
) -> None:
    r"""Estimate a transform series from source onto target with the registration engine.

    Takes the same YAML as `estimate-registration` (beads method only for now) and writes
    a config for `register` / `stabilize`, plus a run journal and a per-timepoint report.

    >>> biahub estimate-transform \\
        -s source.zarr/0/0/0 \\
        -t target.zarr/0/0/0 \\
        -c estimate-registration-beads.yml \\
        -o ./registration_settings.yml
    """
    estimate_transform(
        source_position_dirpaths, target_position_dirpaths, config_filepath, output_filepath
    )


if __name__ == "__main__":
    estimate_transform_cli()
