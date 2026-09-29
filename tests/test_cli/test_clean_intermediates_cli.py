import os
import time

import numpy as np
import pytest

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.clean_intermediates import MARKER, _sample_timepoints
from biahub.cli.main import cli

DS = "DS"
POSITIONS = [("A", "1", "000"), ("A", "1", "001")]
SOURCES = {
    "0-flatfield": ["raw BF"],
    "1-deskew": ["BF", "GFP"],
    "2-reconstruct": ["Phase3D"],
    "3-virtual-stain": ["nuclei", "membrane"],
}


@pytest.fixture()
def project(tmp_path):
    """A finished mantis-v2 project whose assembled plate is an exact copy."""
    rng = np.random.default_rng(0)
    data = {}
    for step, channels in SOURCES.items():
        (tmp_path / step / "slurm_output").mkdir(parents=True)
        with open_ome_zarr(
            tmp_path / step / f"{DS}.zarr", layout="hcs", mode="w", channel_names=channels
        ) as plate:
            for row, col, fov in POSITIONS:
                arr = rng.random((5, len(channels), 2, 8, 8)).astype(np.float32) + 0.1
                plate.create_position(row, col, fov).create_image("0", arr)
                data[step, row, col, fov] = arr
    (tmp_path / "2-reconstruct" / "transfer_function.zarr").mkdir()

    assembled = [
        c for step in ["1-deskew", "2-reconstruct", "3-virtual-stain"] for c in SOURCES[step]
    ]
    with open_ome_zarr(
        tmp_path / "4-assemble" / f"{DS}.zarr", layout="hcs", mode="w", channel_names=assembled
    ) as plate:
        for row, col, fov in POSITIONS:
            arr = np.concatenate(
                [
                    data[s, row, col, fov]
                    for s in ["1-deskew", "2-reconstruct", "3-virtual-stain"]
                ],
                axis=1,
            )
            plate.create_position(row, col, fov).create_image("0", arr)

    (tmp_path / "nextflow").mkdir()
    trace = tmp_path / "nextflow" / "trace.txt"
    rows = ["task_id\thash\tnative_id\tname\tstatus"] + [
        f"1\tx\t1\tassemble_run_wf:run_concatenate ({r}/{c}/{f})\tCOMPLETED"
        for r, c, f in POSITIONS
    ]
    trace.write_text("\n".join(rows) + "\n")
    # The pipeline launch happened before any verification.
    past = time.time() - 120
    os.utime(trace, (past, past))
    (tmp_path / ".nextflow.log").write_text("...\n> Execution complete -- Goodbye\n")
    return tmp_path


def _run(*args):
    return CliRunner().invoke(cli, ["nf", "clean-intermediates", *map(str, args)])


def _edit(project, store, position, fn):
    with open_ome_zarr(project / store / f"{DS}.zarr" / position, mode="r+") as pos:
        fn(pos["0"])


def _step_zarrs(project):
    return sorted(p.parent.name for p in project.glob("*-*/DS.zarr"))


def test_sample_timepoints():
    assert _sample_timepoints(91, 3) == [0, 45, 90]
    assert _sample_timepoints(91, 0) == list(range(91))
    assert _sample_timepoints(4, 10) == [0, 1, 2, 3]
    assert _sample_timepoints(91, 1) == [0]


def test_verified_project_is_cleaned(project):
    assert _run("check", project).exit_code == 0
    assert _run("submit", project, "--cluster", "debug").exit_code == 0

    status = _run("status", project)
    assert status.exit_code == 0
    assert "verified 2/2 pass" in status.output
    assert "all timepoints" in status.output

    dry = _run("delete", project)
    assert dry.exit_code == 0 and "dry run" in dry.output
    assert len(_step_zarrs(project)) == 5  # nothing deleted yet

    assert _run("delete", project, "--yes").exit_code == 0
    assert _step_zarrs(project) == ["4-assemble"]
    assert (project / MARKER).exists()
    assert (project / "2-reconstruct" / "transfer_function.zarr").exists()
    assert all((project / step / "slurm_output").exists() for step in SOURCES)


def test_changed_voxel_refuses_delete(project):
    def change_one_voxel(arr):
        arr[2, 3, 1, 1, 1] = 99.0

    _edit(project, "4-assemble", "A/1/001", change_one_voxel)
    _run("submit", project, "--cluster", "debug")

    status = _run("status", project)
    assert status.exit_code == 1
    assert "FAIL A/1/001" in status.output
    assert "1/128 voxels differ" in status.output

    delete = _run("delete", project, "--yes")
    assert delete.exit_code == 1 and "REFUSED" in delete.output
    assert len(_step_zarrs(project)) == 5


def test_unwritten_assembled_volume_is_caught(project):
    def never_written(arr):
        arr[1, 4] = 0.0

    _edit(project, "4-assemble", "A/1/000", never_written)
    _run("submit", project, "--cluster", "debug")
    status = _run("status", project)
    assert status.exit_code == 1
    assert "assembled volume all zero" in status.output


def test_blank_source_volume_is_reported_not_blocking(project):
    def blank_frame(arr):
        arr[3, 0] = 0.0

    _edit(project, "1-deskew", "A/1/000", blank_frame)
    _edit(project, "4-assemble", "A/1/000", blank_frame)
    _run("submit", project, "--cluster", "debug")

    status = _run("status", project)
    assert status.exit_code == 0
    assert "empty in source (copied as empty): A/1/000 deskew[c0]" in status.output
    assert _run("delete", project, "--yes").exit_code == 0


def test_sampling_is_labelled_and_can_miss(project):
    def change_t1(arr):
        arr[1, 0] = 5.0

    _edit(project, "4-assemble", "A/1/000", change_t1)
    # 3 of 5 timepoints: t = 0, 2, 4, so the bad t=1 is not seen.
    _run("submit", project, "--cluster", "debug", "--timepoints", 3)
    status = _run("status", project)
    assert status.exit_code == 0
    assert "SAMPLED" in status.output
    assert "SAMPLED" in _run("delete", project).output

    _run("submit", project, "--cluster", "debug", "--reverify")
    assert _run("status", project).exit_code == 1


def test_unverified_positions_refuse_delete(project):
    delete = _run("delete", project, "--yes")
    assert delete.exit_code == 1
    assert "not pixel-verified" in delete.output


def test_verification_before_last_launch_refuses_delete(project):
    _run("submit", project, "--cluster", "debug")
    future = time.time() + 120
    os.utime(project / "nextflow" / "trace.txt", (future, future))
    delete = _run("delete", project, "--yes")
    assert delete.exit_code == 1
    assert "verified before the last pipeline launch" in delete.output


def test_check_refuses_running_run(project):
    (project / ".nextflow.log").write_text("... still running\n")
    check = _run("check", project)
    assert check.exit_code == 1
    assert "has not finished" in check.output


def test_check_refuses_unassembled_position(project):
    trace = project / "nextflow" / "trace.txt"
    trace.write_text("\n".join(trace.read_text().splitlines()[:-1]) + "\n")
    check = _run("check", project)
    assert check.exit_code == 1
    assert "no finished run_concatenate" in check.output


def test_check_refuses_channel_subset(project, tmp_path_factory):
    store = project / "4-assemble" / f"{DS}.zarr"
    with open_ome_zarr(
        store, layout="hcs", mode="w", channel_names=["BF", "Phase3D"]
    ) as plate:
        for row, col, fov in POSITIONS:
            plate.create_position(row, col, fov).create_image(
                "0", np.zeros((5, 2, 2, 8, 8), np.float32)
            )
    check = _run("check", project)
    assert check.exit_code == 1
    assert "sources sum to 5" in check.output


def test_interrupted_delete_is_finished(project):
    _run("submit", project, "--cluster", "debug")
    for step in SOURCES:
        store = project / step / f"{DS}.zarr"
        store.rename(store.with_name(f"{DS}.zarr.deleting-20260101T000000"))
    result = _run("delete", project, "--yes")
    assert result.exit_code == 0 and "interrupted delete" in result.output
    assert not list(project.glob("*-*/*.deleting-*"))
