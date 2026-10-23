import click
import numpy as np
import pytest

from click.testing import CliRunner

from biahub.settings import TransformEntry, TransformSettings, load_transform_settings
from biahub.substitute_transforms import substitute_transforms, substitute_transforms_cli
from biahub.utils.config import model_to_yaml


def _shift(dx):
    m = np.eye(4)
    m[2, 3] = dx
    return m.tolist()


def _beads(n_t=4, **kw):
    """A beads run: forward +t in x at every timepoint."""
    return TransformSettings(
        direction="forward",
        moving_channels=["GFP"],
        reference_channel="Phase3D",
        method="beads",
        transforms=[TransformEntry(t=t, matrix=_shift(t), score=0.8) for t in range(n_t)],
        **kw,
    )


def _single(t, dx, method="manual", direction="forward"):
    """A one-timepoint run (what estimate-transform writes for time_indices: t)."""
    return TransformSettings(
        direction=direction,
        moving_channels=["GFP"],
        reference_channel="Phase3D",
        method=method,
        transforms=[TransformEntry(matrix=_shift(dx), estimated_at=t)],
    )


def test_a_single_timepoint_run_replaces_only_its_timepoint():
    combined, log = substitute_transforms(_beads(), [("manual.yml", _single(2, 50.0))])

    xs = [combined.matrix_for(t, "forward")[2, 3] for t in range(4)]
    assert xs == [0.0, 1.0, 50.0, 3.0]
    entry = combined.transforms[2]
    assert entry.t == 2 and entry.method == "manual" and entry.score is None
    assert combined.transforms[1].method is None  # still the file's method (beads)
    assert log == ["transforms t=2: beads (score 0.8) -> manual (score n/a) from manual.yml"]


def test_a_substituted_entry_keeps_its_status():
    # A stand-in from the other run stays unreliable, so apply-transform still reports it.
    stand_in = _single(2, 50.0, method="ants")
    stand_in.transforms[0] = stand_in.transforms[0].model_copy(
        update={"status": "unreliable", "filled_from": "seed", "note": "job failed"}
    )
    combined, _ = substitute_transforms(_beads(), [("ants.yml", stand_in)])

    entry = combined.transforms[2]
    assert (entry.status, entry.filled_from, entry.note) == (
        "unreliable",
        "seed",
        "job failed",
    )


def test_a_substituted_entry_keeps_its_report():
    # The donor's score.report (raw-pair and unregistered-pair metrics) belongs to its
    # transform; the combined file must not lose it.
    donor = _single(2, 50.0, method="ants")
    report = {"correlation": 0.71, "unregistered_correlation": 0.32, "unregistered_score": 0.4}
    donor.transforms[0] = donor.transforms[0].model_copy(
        update={"score": 0.7, "report": report}
    )
    combined, _ = substitute_transforms(_beads(), [("ants.yml", donor)])
    assert combined.transforms[2].report == report


def test_substitutes_are_converted_to_the_base_direction_and_applied_in_order():
    inverse = _single(1, -7.0, method="ants", direction="inverse")  # inverse -7 == forward +7
    later = _single(1, 9.0, method="manual")
    combined, _ = substitute_transforms(_beads(), [("ants.yml", inverse)])
    assert combined.matrix_for(1, "forward")[2, 3] == pytest.approx(7.0)
    combined, _ = substitute_transforms(
        _beads(), [("ants.yml", inverse), ("manual.yml", later)]
    )
    assert combined.matrix_for(1, "forward")[2, 3] == 9.0
    assert combined.transforms[1].method == "manual"


def test_substitution_refuses_what_it_cannot_place():
    other_channel = _single(1, 0.0).model_copy(update={"reference_channel": "Retardance"})
    with pytest.raises(Exception, match="the base maps"):
        substitute_transforms(_beads(), [("x.yml", other_channel)])
    with pytest.raises(Exception, match="--allow-new"):
        substitute_transforms(_beads(), [("x.yml", _single(9, 0.0))])
    added, _ = substitute_transforms(_beads(), [("x.yml", _single(9, 0.0))], allow_new=True)
    assert [e.t for e in added.transforms] == [0, 1, 2, 3, 9]
    no_timepoint = _single(1, 0.0)
    no_timepoint.transforms[0].estimated_at = None
    with pytest.raises(Exception, match="which timepoint it replaces"):
        substitute_transforms(_beads(), [("x.yml", no_timepoint)])
    series_wide = _single(0, 1.0, method="beads")
    with pytest.raises(Exception, match="no per-timepoint entry"):
        substitute_transforms(series_wide, [("x.yml", _single(0, 2.0))])


def test_a_per_position_base_takes_a_shared_substitute_on_the_named_positions():
    base = TransformSettings(
        direction="forward",
        moving_channels=["GFP"],
        reference_channel="Phase3D",
        method="beads",
        positions={
            key: [TransformEntry(t=t, matrix=_shift(t)) for t in range(3)]
            for key in ("0/2/000", "0/2/001")
        },
    )
    with pytest.raises(Exception, match="--positions"):
        substitute_transforms(base, [("x.yml", _single(1, 40.0))])

    combined, _ = substitute_transforms(
        base, [("x.yml", _single(1, 40.0))], positions=["0/2/001"]
    )

    assert combined.matrix_for(1, "forward", "0/2/001")[2, 3] == 40.0
    assert combined.matrix_for(1, "forward", "0/2/000")[2, 3] == 1.0


def test_substitute_transforms_cli_writes_the_combined_file(tmp_path):
    model_to_yaml(_beads(), tmp_path / "beads.yml")
    model_to_yaml(_single(2, 50.0), tmp_path / "manual.yml")
    model_to_yaml(_single(3, 60.0, method="ants"), tmp_path / "ants.yml")
    output = tmp_path / "final.yml"

    result = CliRunner().invoke(
        substitute_transforms_cli,
        [
            "-c",
            str(tmp_path / "beads.yml"),
            "-s",
            str(tmp_path / "manual.yml"),
            "-s",
            str(tmp_path / "ants.yml"),
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    final = load_transform_settings(output)
    assert [e.method for e in final.transforms] == [None, None, "manual", "ants"]
    assert "t=3: beads (score 0.8) -> ants" in result.output
    assert load_transform_settings(tmp_path / "beads.yml").transforms[2].matrix == _shift(2)


def test_a_previous_frame_file_cannot_take_or_give_substitutes():
    # Its matrices are cumulative (each timepoint chained onto the first frame): a
    # substitute at t cannot fix the later timepoints chained through the bad step, and a
    # one-timepoint 'previous' estimate holds a single step, not the accumulated drift.
    chained = _beads(reference_frame="previous")
    with pytest.raises(click.UsageError, match="reference frame 'previous'"):
        substitute_transforms(chained, [("manual.yml", _single(2, 50.0))])
    step_only = _single(2, 50.0)
    step_only.reference_frame = "previous"
    with pytest.raises(click.UsageError, match="reference frame 'previous'"):
        substitute_transforms(_beads(), [("prev.yml", step_only)])
    # an older file says nothing about its frame: allowed, as before
    substitute_transforms(_beads(), [("manual.yml", _single(2, 50.0))])


def test_a_substitute_from_another_voxel_grid_is_refused():
    # Matrices are in voxels: a run on a differently sampled grid would install a
    # transform that loads fine and is physically wrong.
    base = _beads(voxel_size=[1.0, 1.0, 0.174, 0.1494, 0.1494])
    other_grid = _single(1, 0.0).model_copy(
        update={"voxel_size": [1.0, 1.0, 0.2, 0.103, 0.103]}
    )
    with pytest.raises(click.UsageError, match="voxel size"):
        substitute_transforms(base, [("x.yml", other_grid)])
    same_grid = _single(1, 0.0).model_copy(update={"voxel_size": base.voxel_size})
    substitute_transforms(base, [("x.yml", same_grid)])  # same grid: fine
    substitute_transforms(base, [("x.yml", _single(1, 0.0))])  # undeclared: not checked


def test_a_selected_position_missing_from_a_substitute_is_named():
    # --positions picks a position the per-position substitute does not have: say which.
    base = TransformSettings(
        direction="forward",
        moving_channels=["GFP"],
        reference_channel="Phase3D",
        method="beads",
        positions={k: [TransformEntry(t=0, matrix=_shift(0))] for k in ("A/1/0", "A/1/1")},
    )
    sub = TransformSettings(
        direction="forward",
        moving_channels=["GFP"],
        reference_channel="Phase3D",
        method="manual",
        positions={"A/1/0": [TransformEntry(t=0, matrix=_shift(1))]},
    )
    with pytest.raises(click.UsageError, match=r"sub.yml.*A/1/1"):
        substitute_transforms(base, [("sub.yml", sub)], positions=("A/1/1",))
