"""nextflow/registration.nf end to end, local executor (skipped where Nextflow is absent)."""

import json
import os
import shutil
import subprocess
import sys

from pathlib import Path

import numpy as np
import pytest

from iohub import open_ome_zarr
from scipy.ndimage import shift as ndi_shift

from biahub.apply_transform import apply_transform
from biahub.settings import (
    AntsRegistrationSettings,
    ChannelSettings,
    EstimateTransformSettings,
    FocusSettings,
    PhaseCrossCorrSettings,
    ReferenceSettings,
    load_transform_settings,
)
from biahub.utils.config import model_to_yaml
from tests.test_estimate_transform import (
    APPLIED_SHIFT_ZYX,
    SHAPE,
    _synthetic_bead_volume,
    _write_config,
    _write_defocusing_plate,
    _write_plate,
)

REPO = Path(__file__).resolve().parents[1]
NEXTFLOW_DIR = REPO / "nextflow"

pytestmark = pytest.mark.skipif(shutil.which("nextflow") is None, reason="needs nextflow")


def _env(**extra):
    """This checkout's biahub on the tasks' PATH, whichever checkout the venv installed."""
    env = dict(os.environ, **extra)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    env.pop("CLAUDECODE", None)  # Nextflow's agent mode changes its output
    return env


def _nextflow(tmp_path, *params, env=None):
    env = env or _env()
    result = subprocess.run(
        ["nextflow", "run", str(NEXTFLOW_DIR / "registration.nf")]
        + ["-c", str(NEXTFLOW_DIR / "nextflow.config"), "-work-dir", str(tmp_path / "work")]
        + [str(p) for p in params],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result


def _data(position):
    with open_ome_zarr(position, mode="r") as p:
        return list(p.channel_names), np.asarray(p.data)


def _beads_plate(tmp_path, blank_last=False):
    """GFP is Phase3D shifted by APPLIED_SHIFT_ZYX; optionally no beads at the last t."""
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    last = rng.normal(0, 5.0, size=SHAPE).astype(np.float32) if blank_last else mov
    return _write_plate(tmp_path / "data.zarr", [(ref, mov), (ref, mov), (ref, last)])


def _drifting_plate(tmp_path):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    frames = [
        ndi_shift(ref, shift=tuple(t * np.array(APPLIED_SHIFT_ZYX)), order=1, mode="constant")
        for t in range(3)
    ]
    return _write_plate(tmp_path / "data.zarr", [(f, f) for f in frames])


def _focus_config(tmp_path):
    config = tmp_path / "estimate.yml"
    model_to_yaml(
        EstimateTransformSettings(
            moving=ChannelSettings(channel="Phase3D"),
            reference=ReferenceSettings(frame="first"),
            method="focus-finding",
            focus_finding=FocusSettings(axes="xyz", center_crop_xy=[48, 48]),
        ),
        config,
    )
    return config


# Every method that runs as a batch (manual is interactive: refused, tested below).
# Each case: (plate builder, config builder, cross registration?)
CASES = {
    # t=2 has no beads: flag, repair and finalize all run
    "beads": (
        lambda tmp: _beads_plate(tmp, blank_last=True),
        lambda tmp: _write_config(tmp),
        True,
    ),
    "beads-propagated": (
        lambda tmp: _beads_plate(tmp),
        lambda tmp: _write_config(tmp, seed_from="previous_timepoint"),
        True,
    ),
    "ants": (
        lambda tmp: _beads_plate(tmp),
        lambda tmp: _write_config(
            tmp, method="ants", ants=AntsRegistrationSettings(), transform_type="similarity"
        ),
        True,
    ),
    "ants-sobel": (
        lambda tmp: _beads_plate(tmp),
        lambda tmp: _write_config(
            tmp,
            method="ants",
            ants=AntsRegistrationSettings(sobel_filter=True),
            transform_type="similarity",
        ),
        True,
    ),
    "phase-cross-corr": (
        _drifting_plate,
        lambda tmp: _write_config(
            tmp,
            reference="first",
            method="phase-cross-corr",
            phase_cross_corr=PhaseCrossCorrSettings(center_crop_xy=[SHAPE[1], SHAPE[2]]),
        ),
        False,
    ),
    "focus-finding": (_write_defocusing_plate, _focus_config, False),
}


@pytest.mark.parametrize("case", list(CASES))
def test_registration_nf_runs_every_method_as_the_cli_does(tmp_path, case):
    make_plate, make_config, cross = CASES[case]
    plate = make_plate(tmp_path)
    store = plate.parents[2]
    config = make_config(tmp_path)
    reference = [plate] if cross else None
    out = tmp_path / "run"
    # One ITK thread on both sides: ANTs repeats exactly then, while different thread
    # counts (the task's CPUs vs this node's) differ by up to ~0.1 voxel.
    env = _env(ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1")

    _nextflow(
        tmp_path,
        "--moving", store, *(["--reference", store] if cross else []),
        "--estimate_config", config, "--estimate_positions", "A/1/0",
        "--apply", "--output", out,
        env=env,
    )  # fmt: skip

    # The plain CLI in a fresh process, so the thread count is set before ITK starts.
    cli = tmp_path / "cli" / "transforms.yml"
    ref_args = ["-r", str(plate)] if cross else []
    result = subprocess.run(
        [sys.executable, "-m", "biahub.cli.main", "estimate-transform", "--cluster", "debug"]
        + ["-m", str(plate), *ref_args, "-c", str(config), "-o", str(cli)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    nf_model = load_transform_settings(out / "transforms.yml")
    cli_model = load_transform_settings(cli)
    assert nf_model == cli_model
    assert _report(out / "transforms") == _report(cli.with_suffix(""))

    apply_transform([plate], cli, tmp_path / "cli.zarr", reference, cluster="debug")
    # where the logs are, as every step's init writes it (main's convention)
    assert "Where are the logs?" in (out / "slurm_output" / "README.md").read_text()
    names, data = _data(out / f"{store.stem}.zarr" / "A" / "1" / "0")
    cli_names, cli_data = _data(tmp_path / "cli.zarr" / "A" / "1" / "0")
    assert names == cli_names
    np.testing.assert_array_equal(data, cli_data)


def _report(run_dir):
    report = json.loads((run_dir / "estimate_transform_report.json").read_text())
    report.pop("run_id")
    return report


def test_registration_nf_stabilizes_each_position_and_refuses_manual(tmp_path):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    drift = {"0": np.array(APPLIED_SHIFT_ZYX), "1": -np.array(APPLIED_SHIFT_ZYX)}
    store = tmp_path / "two.zarr"
    with open_ome_zarr(store, layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]) as p:
        for fov, d in drift.items():
            frames = [
                ndi_shift(ref, shift=tuple(t * d), order=1, mode="constant") for t in range(3)
            ]
            p.create_position("A", "1", fov)["0"] = np.stack(
                [np.stack([f, f]) for f in frames]
            ).astype(np.float32)
    config = _write_config(
        tmp_path,
        reference="first",
        method="phase-cross-corr",
        phase_cross_corr=PhaseCrossCorrSettings(center_crop_xy=[SHAPE[1], SHAPE[2]]),
    )
    out = tmp_path / "run"

    _nextflow(
        tmp_path,
        "--moving", store, "--estimate_config", config, "--estimate_positions", "*/*/*",
        "--apply", "--output", out,
    )  # fmt: skip

    model = load_transform_settings(out / "transforms.yml")
    assert model.per_position and sorted(model.positions) == ["A/1/0", "A/1/1"]
    for fov, d in drift.items():
        for t in range(3):
            np.testing.assert_allclose(
                model.matrix_for(t, "inverse", f"A/1/{fov}")[:3, 3], t * d, atol=0.5
            )
        assert (out / "two.zarr" / "A" / "1" / fov).exists()

    manual = _write_config(tmp_path, method="manual", time_indices=0)
    env = _env()
    result = subprocess.run(
        ["nextflow", "run", str(NEXTFLOW_DIR / "registration.nf")]
        + ["-c", str(NEXTFLOW_DIR / "nextflow.config"), "-work-dir", str(tmp_path / "work2")]
        + ["--moving", str(store), "--reference", str(store), "--estimate_config", str(manual)]
        + ["--estimate_positions", "A/1/0", "--output", str(tmp_path / "manual")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Manual registration is interactive" in result.stdout + result.stderr


def test_registration_nf_resolves_relative_inputs_and_wants_an_absolute_output(tmp_path):
    # Tasks run in their own work directory: relative inputs are resolved at launch, and a
    # relative --output (which the shared log paths read as given) is refused up front.
    plate = _beads_plate(tmp_path)
    _write_config(tmp_path)
    _nextflow(
        tmp_path,
        "--moving", "data.zarr", "--reference", "data.zarr",
        "--estimate_config", "estimate.yml", "--estimate_positions", "A/1/0",
        "--apply", "--apply_output", "registered.zarr", "--output", tmp_path / "run",
    )  # fmt: skip
    assert load_transform_settings(tmp_path / "run" / "transforms.yml").transforms
    assert (tmp_path / "registered.zarr" / "A" / "1" / "0").exists()  # where it was asked for
    assert plate.exists()

    result = subprocess.run(
        ["nextflow", "run", str(NEXTFLOW_DIR / "registration.nf")]
        + ["-c", str(NEXTFLOW_DIR / "nextflow.config"), "-work-dir", str(tmp_path / "w2")]
        + ["--moving", "data.zarr", "--estimate_config", "estimate.yml"]
        + ["--estimate_positions", "A/1/0", "--output", "relative_run"],
        cwd=tmp_path,
        env=_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--output must be an absolute path" in result.stdout + result.stderr


def test_registration_nf_reruns_when_a_file_is_rewritten_in_place(tmp_path):
    # Nextflow caches tasks by their inputs; a file's name alone would let -resume reuse
    # tasks after the file was rewritten (e.g. by substitute-transforms -o <same file>).
    from biahub.settings import TransformEntry, TransformSettings

    plate = _drifting_plate(tmp_path)
    store = plate.parents[2]
    transforms = tmp_path / "final.yml"

    def write(dy):
        matrix = np.eye(4)
        matrix[1, 3] = dy
        model_to_yaml(
            TransformSettings(
                direction="forward",
                moving_channels=["GFP"],
                transforms=[TransformEntry(matrix=matrix.tolist())],
            ),
            transforms,
        )

    def run():
        _nextflow(
            tmp_path,
            "--moving", store, "--transforms", transforms, "--output", tmp_path / "run",
            "-resume",
        )  # fmt: skip
        return _data(tmp_path / "run" / "data.zarr" / "A" / "1" / "0")[1]

    write(0.0)
    first = run()
    write(3.0)  # same path, new contents
    second = run()
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("seed_from", ["input", "previous_timepoint"])
def test_registration_nf_finishes_past_a_broken_chunk_and_resume_redoes_it(
    tmp_path, seed_from
):
    # A timepoint whose data cannot be read: the run retries it, then finishes with that
    # timepoint as a stand-in (the error in its note); once the data is fixed, -resume
    # redoes it (a failed task is never cached). With propagation the one estimate task per
    # position fails: the run must still flag and write the file, not stop silently.
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    plate = _write_plate(tmp_path / "data.zarr", [(ref, mov)] * 3)
    store = plate.parents[2]
    chunk = plate / "0" / "c" / "1" / "1" / "0" / "0" / "0"  # t=1, the moving channel
    assert chunk.is_file()
    chunk.write_bytes(b"not a chunk")
    config = _write_config(tmp_path, seed_from=seed_from)
    params = (
        "--moving", store, "--reference", store, "--estimate_config", config,
        "--estimate_positions", "A/1/0", "--output", tmp_path / "run", "-resume",
    )  # fmt: skip

    _nextflow(tmp_path, *params)
    entry = load_transform_settings(tmp_path / "run" / "transforms.yml").transforms[1]
    assert entry.status == "unreliable" and "job failed" in entry.note

    with open_ome_zarr(plate, mode="r+") as position:
        position["0"][1, 1] = mov  # the data is fixed
    _nextflow(tmp_path, *params)
    entry = load_transform_settings(tmp_path / "run" / "transforms.yml").transforms[1]
    assert entry.status == "accepted" and entry.note is None


def test_registration_nf_takes_initial_transforms_as_the_cli_does(tmp_path):
    # t=2's moving frame has no beads: the initial transform stands there, as in one call.
    from tests.test_initial_transforms import _truth_file

    plate = _beads_plate(tmp_path, blank_last=True)
    store = plate.parents[2]
    config = _write_config(tmp_path)
    initial, _truth = _truth_file(tmp_path)
    out = tmp_path / "run"
    env = _env(ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1")
    _nextflow(
        tmp_path,
        "--moving", store, "--reference", store, "--estimate_config", config,
        "--estimate_positions", "A/1/0", "--initial_transforms", initial, "--output", out,
        env=env,
    )  # fmt: skip

    cli = tmp_path / "cli" / "transforms.yml"
    result = subprocess.run(
        [sys.executable, "-m", "biahub.cli.main", "estimate-transform", "--cluster", "debug"]
        + ["-m", str(plate), "-r", str(plate), "-c", str(config), "-o", str(cli)]
        + ["--initial-transforms", str(initial)],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr[-3000:]
    nf_model = load_transform_settings(out / "transforms.yml")
    assert nf_model == load_transform_settings(cli)
    assert nf_model.transforms[2].seeded_from == "initial"


def test_registration_nf_reruns_the_estimate_when_its_config_is_edited(tmp_path):
    # main's convention (#397): the config is staged into every estimate task, so an
    # edit in place reruns them under -resume instead of reusing results of the old one.
    plate = _beads_plate(tmp_path)
    store = plate.parents[2]
    config = _write_config(tmp_path)
    params = (
        "--moving", store, "--reference", store, "--estimate_config", config,
        "--estimate_positions", "A/1/0", "--output", tmp_path / "run", "-resume",
    )  # fmt: skip
    _nextflow(tmp_path, *params)
    assert len(load_transform_settings(tmp_path / "run" / "transforms.yml").transforms) == 3

    _write_config(tmp_path, time_indices=[0, 1])  # same path, new contents
    _nextflow(tmp_path, *params)
    assert len(load_transform_settings(tmp_path / "run" / "transforms.yml").transforms) == 2
