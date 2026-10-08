"""--initial-transforms: per timepoint, the estimate, a refinement from the initial
transform and the initial transform as-is compete; ties keep the earlier one."""

import numpy as np
import pytest

from biahub.core.transform import Transform
from biahub.registration.engine import estimate_propagated, estimate_series
from biahub.registration.estimators import EstimationError
from biahub.registration.policies import CrossChannel, FixedSeed

IDENTITY = Transform.identity(3)


def _shift_x(dx):
    return Transform.from_translation([0.0, 0.0, float(dx)])


def _frames(n):
    return np.stack([np.full((2, 2, 2), float(t + 1)) for t in range(n)])


class _AddOne:
    """Moves its seed by +1 in x; fails when started from x == `fail_from`."""

    def __init__(self, fail_from=None):
        self.fail_from = fail_from

    def estimate(self, mov, ref, seed=None):
        start = seed or IDENTITY
        if self.fail_from is not None and start.translation[2] == self.fail_from:
            raise EstimationError("too few matches")
        return start @ _shift_x(1)


def _score_near(target):
    """Higher the closer a transform's x is to `target`."""
    return lambda transform, mov, ref: -abs(transform.translation[2] - target)


@pytest.mark.parametrize(
    "target, expected_x, seeded_from",
    [
        (1.0, 1.0, None),  # the estimate (seed 0 -> 1) is best
        (11.0, 11.0, "initial+refined"),  # refined from the initial (10 -> 11) is best
        (10.0, 10.0, "initial"),  # the initial transform as-is is best
    ],
)
def test_the_best_of_estimate_refined_and_initial_wins(target, expected_x, seeded_from):
    mov = _frames(1)
    result = estimate_series(
        mov, CrossChannel(mov), _AddOne(), FixedSeed(IDENTITY), _score_near(target), [0],
        initial={0: _shift_x(10)},
    )  # fmt: skip
    assert result.transforms[0].translation[2] == expected_x
    assert result.seeded_from.get(0) == seeded_from


def test_a_tie_keeps_the_estimate():
    mov = _frames(1)
    result = estimate_series(
        mov, CrossChannel(mov), _AddOne(), FixedSeed(IDENTITY),
        lambda transform, mov, ref: 1.0, [0], initial={0: _shift_x(10)},
    )  # fmt: skip
    assert result.transforms[0].translation[2] == 1.0 and 0 not in result.seeded_from


def test_an_initial_transform_rescues_a_failed_estimate():
    mov = _frames(1)
    result = estimate_series(
        mov, CrossChannel(mov), _AddOne(fail_from=0.0), FixedSeed(IDENTITY),
        _score_near(11.0), [0], initial={0: _shift_x(10)},
    )  # fmt: skip
    assert result.transforms[0].translation[2] == 11.0 and 0 not in result.errors
    assert result.seeded_from[0] == "initial+refined"


def test_timepoints_without_an_initial_transform_are_estimated_as_before():
    mov = _frames(2)
    result = estimate_series(
        mov, CrossChannel(mov), _AddOne(), FixedSeed(IDENTITY), _score_near(10.0), [0, 1],
        initial={1: _shift_x(10)},
    )  # fmt: skip
    assert result.transforms[0].translation[2] == 1.0 and 0 not in result.seeded_from
    assert result.transforms[1].translation[2] == 10.0 and result.seeded_from[1] == "initial"


def test_propagation_lets_the_initial_transform_compete_and_passes_the_winner_on():
    mov = _frames(2)
    result = estimate_propagated(
        mov, CrossChannel(mov), _AddOne(), IDENTITY, _score_near(10.0), [0, 1],
        initial={0: _shift_x(10)},
    )  # fmt: skip
    assert result.transforms[0].translation[2] == 10.0 and result.seeded_from[0] == "initial"
    # t=1 starts from t=0's winner (x=10): 10 + 1 = 11, against the input seed's 0 + 1
    assert result.transforms[1].translation[2] == 11.0


def test_the_transforms_file_says_which_entries_came_from_the_initial_file():
    from biahub.estimate_transform import transform_entries
    from biahub.registration.engine import SeriesResult

    result = SeriesResult(
        transforms={0: _shift_x(1), 1: _shift_x(10), 2: _shift_x(11)},
        scores={0: 0.9, 1: 0.9, 2: 0.9},
        seeded_from={1: "initial", 2: "initial+refined"},
        provenance={2: "t-1"},  # t=2 was then replaced by a repair
    )
    entries = transform_entries(
        result, [0, 1, 2], [_shift_x(1), _shift_x(10), _shift_x(11)], 0.4
    )
    assert [e.seeded_from for e in entries] == [None, "initial", None]


def test_records_keep_seeded_from_through_a_reload(tmp_path):
    import json

    from biahub.registration.engine import _load_series

    (tmp_path / "0.json").write_text(
        json.dumps(
            {"t": 0, "matrix": _shift_x(10).matrix.tolist(), "score": 0.9, "error": None,
             "seeded_from": "initial"}
        )
    )  # fmt: skip
    result = _load_series(tmp_path, [0], "euclidean", resumed=[0])
    assert result.seeded_from == {0: "initial"}


def _blank_plate(tmp_path):
    """Three timepoints of beads; the last moving frame has none (its estimate fails)."""
    from scipy.ndimage import shift as ndi_shift

    from tests.test_estimate_transform import (
        APPLIED_SHIFT_ZYX,
        SHAPE,
        _synthetic_bead_volume,
        _write_plate,
    )

    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    blank = rng.normal(0, 5.0, size=SHAPE).astype(np.float32)
    return _write_plate(tmp_path / "plate.zarr", [(ref, mov), (ref, mov), (ref, blank)])


def test_the_driver_reads_initial_transforms_from_the_run_folder(tmp_path):
    # A plain run given initial transforms: every job reads them (the run folder's
    # initial.json), and where the estimate fails (t=2) the initial transform stands in.
    from biahub.registration.engine import estimate_transform_series
    from biahub.settings import load_estimate_transform_settings
    from tests.test_estimate_transform import APPLIED_SHIFT_ZYX, _write_config

    plate = _blank_plate(tmp_path)
    settings = load_estimate_transform_settings(_write_config(tmp_path))
    truth = Transform.from_translation([-a for a in APPLIED_SHIFT_ZYX])  # forward
    run = tmp_path / "run"
    result, _ts, _transforms = estimate_transform_series(
        plate, plate, settings, run, cluster="debug", initial={2: truth}
    )
    assert (run / "initial.json").exists()
    assert result.seeded_from.get(2) == "initial"
    np.testing.assert_allclose(result.transforms[2].matrix, truth.matrix)


def test_methods_that_ignore_seeds_refuse_initial_transforms(tmp_path):
    import click

    from biahub.registration.engine import init_run
    from biahub.settings import PhaseCrossCorrSettings, load_estimate_transform_settings
    from tests.test_estimate_transform import SHAPE, _write_config

    plate = _blank_plate(tmp_path)
    settings = load_estimate_transform_settings(
        _write_config(
            tmp_path,
            reference="first",
            method="phase-cross-corr",
            phase_cross_corr=PhaseCrossCorrSettings(center_crop_xy=[SHAPE[1], SHAPE[2]]),
        )
    )
    with pytest.raises(click.UsageError, match="ignores seeds"):
        init_run(plate, plate, settings, tmp_path / "run", initial={0: IDENTITY})


def test_resume_refuses_different_initial_transforms(tmp_path):
    import click

    from biahub.registration.engine import init_run
    from biahub.settings import load_estimate_transform_settings
    from tests.test_estimate_transform import _write_config

    plate = _blank_plate(tmp_path)
    settings = load_estimate_transform_settings(_write_config(tmp_path))
    run = tmp_path / "run"
    init_run(plate, plate, settings, run, initial={2: _shift_x(1)})
    init_run(plate, plate, settings, run, resume=True, initial={2: _shift_x(1)})  # same: fine
    with pytest.raises(click.UsageError, match="initial_sha256"):
        init_run(plate, plate, settings, run, resume=True, initial={2: _shift_x(2)})


def _truth_file(tmp_path, name="initial.yml", t=2):
    """A transforms file holding the true transform at one timepoint."""
    from biahub.settings import TransformEntry, TransformSettings
    from biahub.utils.config import model_to_yaml
    from tests.test_estimate_transform import APPLIED_SHIFT_ZYX

    truth = Transform.from_translation([-a for a in APPLIED_SHIFT_ZYX])
    path = tmp_path / name
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            reference_channel="Phase3D",
            transforms=[TransformEntry(t=t, matrix=truth.to_list(), score=0.9)],
        ),
        path,
    )
    return path, truth


def _cli(*args):
    from click.testing import CliRunner

    from biahub.cli.main import cli

    return CliRunner().invoke(cli, ["estimate-transform", *map(str, args)])


def test_estimate_transform_takes_initial_transforms_in_one_call_and_by_steps(tmp_path):
    from biahub.settings import load_transform_settings
    from tests.test_estimate_transform import _write_config

    plate = _blank_plate(tmp_path)
    config = _write_config(tmp_path)
    initial, truth = _truth_file(tmp_path)
    one = tmp_path / "one" / "transforms.yml"
    result = _cli("-m", plate, "-r", plate, "-c", config, "-o", one,
                  "--initial-transforms", initial, "--cluster", "debug")  # fmt: skip
    assert result.exit_code == 0, result.output
    entry = load_transform_settings(one).transforms[2]
    # t=2's moving frame is blank: the initial transform is used (not the config seed as
    # a stand-in), and still unreliable, since nothing in the frame can confirm it
    assert entry.seeded_from == "initial" and entry.filled_from is None
    assert entry.status == "unreliable"
    np.testing.assert_allclose(np.asarray(entry.matrix), truth.matrix)
    report = (one.with_suffix("") / "estimate_transform_report.json").read_text()
    assert '"seeded_from"' in report

    steps = tmp_path / "steps" / "transforms.yml"
    common = ["-m", plate, "-r", plate, "-c", config, "-o", steps]
    for args in (["--init", "--initial-transforms", initial], ["--step", "estimate"],
                 ["--step", "flag"], ["--step", "finalize"]):  # fmt: skip
        result = _cli(*args, *common)
        assert result.exit_code == 0, result.output
    assert load_transform_settings(steps) == load_transform_settings(one)


def test_initial_transforms_are_never_overwritten_or_misplaced(tmp_path):
    from tests.test_estimate_transform import _write_config

    plate = _blank_plate(tmp_path)
    config = _write_config(tmp_path)
    initial, _truth = _truth_file(tmp_path, name="transforms.yml")
    common = ["-m", plate, "-r", plate, "-c", config]

    result = _cli(*common, "-o", initial, "--initial-transforms", initial)
    assert result.exit_code != 0 and "new file" in result.output
    result = _cli(*common, "-o", tmp_path / "x.yml", "--step", "estimate",
                  "--initial-transforms", initial)  # fmt: skip
    assert result.exit_code != 0 and "--init" in result.output


def test_a_per_position_initial_file_must_cover_the_positions(tmp_path):
    from biahub.settings import TransformEntry, TransformSettings
    from biahub.utils.config import model_to_yaml
    from tests.test_estimate_transform import _write_config

    plate = _blank_plate(tmp_path)  # position A/1/0
    other = tmp_path / "other.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            positions={"B/2/0": [TransformEntry(t=0, matrix=IDENTITY.to_list())]},
        ),
        other,
    )
    result = _cli("-m", plate, "-r", plate, "-c", _write_config(tmp_path),
                  "-o", tmp_path / "out.yml", "--initial-transforms", other)  # fmt: skip
    assert result.exit_code != 0 and "no list for positions ['A/1/0']" in result.output
