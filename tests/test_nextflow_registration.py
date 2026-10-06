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
from biahub.estimate_transform import estimate_transform
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


def _env():
    """This checkout's biahub on the tasks' PATH, whichever checkout the venv installed."""
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    env.pop("CLAUDECODE", None)  # Nextflow's agent mode changes its output
    return env


def _nextflow(tmp_path, *params):
    env = _env()
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

    _nextflow(
        tmp_path,
        "--moving", store, *(["--reference", store] if cross else []),
        "--estimate_config", config, "--estimate_positions", "A/1/0",
        "--apply", "--output", out,
    )  # fmt: skip

    cli = tmp_path / "cli" / "transforms.yml"
    estimate_transform(
        [plate], config, cli, reference_position_dirpaths=reference, cluster="debug"
    )
    nf_model = load_transform_settings(out / "transforms.yml")
    cli_model = load_transform_settings(cli)
    if case.startswith("ants"):
        # Multithreaded ANTs repeats to ~0.01 voxel, not bit for bit (the Nextflow task
        # and this process may run different thread counts).
        for nf_entry, cli_entry in zip(nf_model.transforms, cli_model.transforms, strict=True):
            np.testing.assert_allclose(nf_entry.matrix, cli_entry.matrix, atol=0.05)
            assert nf_entry.status == cli_entry.status
        return
    assert nf_model == cli_model
    assert _report(out / "transforms") == _report(cli.with_suffix(""))

    apply_transform([plate], cli, tmp_path / "cli.zarr", reference, cluster="debug")
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
