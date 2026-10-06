"""estimate-transform --init / --step: the phases one at a time, as Nextflow runs them."""

import json

import numpy as np
import pytest

from click.testing import CliRunner
from iohub import open_ome_zarr
from scipy.ndimage import shift as ndi_shift

from biahub.cli.main import cli
from biahub.estimate_transform import estimate_transform
from biahub.settings import PhaseCrossCorrSettings, load_transform_settings
from tests.test_estimate_transform import (
    APPLIED_SHIFT_ZYX,
    SHAPE,
    _synthetic_bead_volume,
    _write_config,
    _write_plate,
)


@pytest.fixture
def beads_plate_with_a_blank_timepoint(tmp_path):
    """Three timepoints; the last GFP frame has no beads, only noise."""
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    blank = rng.normal(0, 5.0, size=SHAPE).astype(np.float32)
    return _write_plate(tmp_path / "beads_blank.zarr", [(ref, mov), (ref, mov), (ref, blank)])


SWEEP = {"grid": [{"beads.hungarian_match_settings.cost_threshold": [0.05, 0.2]}]}


def _cli(*args):
    result = CliRunner().invoke(cli, ["estimate-transform", *map(str, args)])
    return result


def _ok(*args):
    result = _cli(*args)
    assert result.exit_code == 0, result.output
    return result.output


def _plan(output: str) -> dict:
    (line,) = [line for line in output.splitlines() if line.startswith("PLAN:")]
    return json.loads(line.removeprefix("PLAN:"))


def _report(output):
    report = json.loads(
        (output.with_suffix("") / "estimate_transform_report.json").read_text()
    )
    report.pop("run_id")
    return report


def _by_steps(movings, config, output, references=()):
    """Run every step the way the Nextflow module does; return the init plan."""
    common = ["-c", config, "-o", output]
    if references:
        common += ["-r", *references]
    init = _ok("--init", "-m", *movings, *common)
    assert any(line.startswith("RESOURCES:") for line in init.splitlines())
    plan = _plan(init)
    for moving, key in zip(movings, plan["positions"], strict=True):
        if plan["propagated"]:
            _ok("--step", "estimate", "-m", moving, *common)
        else:
            for t in plan["time_indices"]:
                _ok("--step", "estimate", "--timepoints", t, "-m", moving, *common)
        flags = _plan(_ok("--step", "flag", "-m", moving, *common))[key]
        for step in ("repair", "sweep"):
            for t in flags[step]:
                _ok("--step", step, "--timepoints", t, "-m", moving, *common)
    _ok("--step", "finalize", "-m", *movings, *common)
    return plan


def test_the_steps_write_what_one_call_writes(beads_plate_with_a_blank_timepoint, tmp_path):
    # t=2 has no beads: flagged, repaired and swept, so every step runs.
    plate = beads_plate_with_a_blank_timepoint
    config = _write_config(tmp_path, fallback={"sweep": SWEEP})
    one_call = tmp_path / "one" / "transforms.yml"
    estimate_transform(
        [plate], config, one_call, reference_position_dirpaths=[plate], cluster="debug"
    )

    by_steps = tmp_path / "steps" / "transforms.yml"
    plan = _by_steps([plate], config, by_steps, references=[plate])

    assert plan["positions"] == ["A/1/0"] and plan["time_indices"] == [0, 1, 2]
    assert set(plan["resources"]) == {"estimate", "repair", "sweep"}
    assert load_transform_settings(by_steps) == load_transform_settings(one_call)
    report = _report(by_steps)
    assert report == _report(one_call)
    assert report["flagged"] == [2] and set(report["repairs"]) == set(report["sweeps"]) == {
        "2"
    }


def test_the_steps_refuse_what_init_did_not_start(
    beads_plate_with_a_blank_timepoint, tmp_path
):
    plate = beads_plate_with_a_blank_timepoint
    config = _write_config(tmp_path)
    output = tmp_path / "out" / "transforms.yml"
    common = ["-m", plate, "-r", plate, "-c", config, "-o", output]

    result = _cli("--step", "estimate", *common)
    assert result.exit_code != 0 and "run with --init first" in result.output
    result = _cli("--init", "--step", "flag", *common)
    assert result.exit_code != 0 and "separate calls" in result.output

    _ok("--init", *common)
    _ok("--step", "estimate", *common)
    _ok("--step", "flag", *common)
    result = _cli("--step", "repair", "--timepoints", "0", *common)  # only t=2 is flagged
    assert result.exit_code != 0 and "not in this run's repair list" in result.output
    result = _cli("--step", "flag", "--timepoints", "0", *common)
    assert result.exit_code != 0 and "--timepoints goes with" in result.output

    _write_config(tmp_path, transform_type="affine")  # same path, changed config
    result = _cli("--step", "finalize", *common)
    assert result.exit_code != 0 and "config changed since --init" in result.output


def test_the_steps_estimate_several_positions_each_on_its_own(tmp_path):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    drift = {"0": np.array(APPLIED_SHIFT_ZYX), "1": -np.array(APPLIED_SHIFT_ZYX)}
    path = tmp_path / "two.zarr"
    with open_ome_zarr(
        path, layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]
    ) as plate:
        for fov, d in drift.items():
            frames = [
                ndi_shift(ref, shift=tuple(t * d), order=1, mode="constant") for t in range(3)
            ]
            plate.create_position("A", "1", fov)["0"] = np.stack(
                [np.stack([f, f]) for f in frames]
            ).astype(np.float32)
    config = _write_config(
        tmp_path,
        reference="first",
        method="phase-cross-corr",
        phase_cross_corr=PhaseCrossCorrSettings(center_crop_xy=[SHAPE[1], SHAPE[2]]),
    )
    movings = [path / "A" / "1" / "0", path / "A" / "1" / "1"]
    output = tmp_path / "out" / "transforms.yml"

    _by_steps(movings, config, output)

    model = load_transform_settings(output)
    assert model.per_position and sorted(model.positions) == ["A/1/0", "A/1/1"]
    for fov, d in drift.items():
        for t in range(3):
            np.testing.assert_allclose(
                model.matrix_for(t, "inverse", f"A/1/{fov}")[:3, 3], t * d, atol=0.5
            )
    result = _cli("--step", "finalize", "-m", movings[0], "-c", config, "-o", output)
    assert result.exit_code != 0 and "finalize writes every position" in result.output


def test_a_propagated_run_is_one_estimate_step(tmp_path):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    step = np.array([2.0, -2.0, 2.0])
    plate = _write_plate(
        tmp_path / "drift.zarr",
        [
            (ref, ndi_shift(ref, shift=tuple((t + 1) * step), order=1, mode="constant"))
            for t in range(3)
        ],
    )
    config = _write_config(tmp_path, seed_from="previous_timepoint")
    output = tmp_path / "out" / "transforms.yml"
    common = ["-m", plate, "-r", plate, "-c", config, "-o", output]

    assert _plan(_ok("--init", *common))["propagated"] is True
    result = _cli("--step", "estimate", "--timepoints", "1", *common)
    assert result.exit_code != 0 and "drop --timepoints" in result.output

    plan = _by_steps([plate], config, output, references=[plate])
    assert plan["propagated"] is True
    one_call = tmp_path / "one" / "transforms.yml"
    estimate_transform(
        [plate], config, one_call, reference_position_dirpaths=[plate], cluster="debug"
    )
    assert load_transform_settings(output) == load_transform_settings(one_call)


@pytest.mark.parametrize("step", ["estimate", "repair"])
def test_a_retried_step_with_resume_keeps_finished_timepoints(
    beads_plate_with_a_blank_timepoint, tmp_path, monkeypatch, step
):
    plate = beads_plate_with_a_blank_timepoint
    config = _write_config(tmp_path)
    output = tmp_path / "out" / "transforms.yml"
    common = ["-m", plate, "-r", plate, "-c", config, "-o", output]
    _ok("--init", *common)
    _ok("--step", "estimate", *common)
    _ok("--step", "flag", *common)
    _ok("--step", "repair", *common)

    import biahub.registration.engine as engine

    def _must_not_run(*args, **kwargs):
        raise AssertionError("a finished timepoint was redone")

    monkeypatch.setattr(engine, f"_{step}_timepoint_job", _must_not_run)
    _ok("--step", step, "--resume", *common)
