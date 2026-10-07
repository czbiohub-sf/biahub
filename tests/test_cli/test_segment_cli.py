"""biahub segment: Nextflow-style modes (--init, --cluster debug, --resume)."""

import json

import numpy as np
import pytest
import torch
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli


@pytest.fixture
def segment_config(tmp_path):
    def _write(pretrained_model="cpsam_v2", version=None):
        config = {
            "models": {
                "nuc": {
                    "pretrained_model": pretrained_model,
                    "channels": ["GFP"],
                    "z_slice_2D": 1,
                    "eval_args": {"diameter": None},
                }
            }
        }
        if version:
            config["output_ome_zarr_version"] = version
        path = tmp_path / "segment.yml"
        path.write_text(yaml.safe_dump(config))
        return path

    return _write


@pytest.fixture
def cpu_cellpose(monkeypatch):
    """Workers ask for the GPU; run the (fake) model on the CPU instead."""
    monkeypatch.setattr("biahub.segment.cellpose_device", lambda gpu: torch.device("cpu"))


def _run(*args):
    return CliRunner().invoke(cli, ["segment", *map(str, args)])


def test_segment_init_prints_resources_and_stamps_provenance(
    fake_cellpose, example_plate, segment_config, tmp_path
):
    plate_path, _ = example_plate  # (T=3, C=6, Z=4, Y=5, X=6)
    out = tmp_path / "seg.zarr"

    result = _run(
        "--init", "-i", plate_path / "A" / "1" / "0", "-o", out, "-c", segment_config()
    )

    assert result.exit_code == 0, result.output
    line = next(ln for ln in result.output.splitlines() if ln.startswith("RESOURCES:"))
    assert set(json.loads(line.removeprefix("RESOURCES:"))) == {
        "cpus",
        "mem_gb",
        "time_minutes",
    }
    with open_ome_zarr(out / "A" / "1" / "0", mode="r") as ds:
        assert ds.channel_names == ["nuc_labels"]
        assert ds.data.shape == (3, 1, 1, 5, 6)
        assert ds.data.dtype == np.uint32
        recorded = dict(ds.zattrs)["biahub-segment"]  # provenance lives on each position
    # Provenance is the config as written: channel names, not resolved indices.
    assert recorded["models"]["nuc"]["channels"] == ["GFP"]
    assert fake_cellpose.evaluated == []  # --init segments nothing


def test_segment_init_rejects_an_unknown_model(
    fake_cellpose, example_plate, segment_config, tmp_path
):
    plate_path, _ = example_plate
    result = _run(
        "--init",
        "-i",
        plate_path / "A" / "1" / "0",
        "-o",
        tmp_path / "seg.zarr",
        "-c",
        segment_config(pretrained_model="nuclei"),
    )
    assert result.exit_code != 0
    assert "Unknown cellpose model 'nuclei'" in str(result.exception) + result.output


def test_segment_debug_runs_one_position_in_process(
    fake_cellpose, cpu_cellpose, example_plate, segment_config, tmp_path
):
    plate_path, _ = example_plate
    out, config = tmp_path / "seg.zarr", segment_config()
    position = plate_path / "B" / "1" / "0"
    assert _run("--init", "-i", position, "-o", out, "-c", config).exit_code == 0
    fake_cellpose.built.clear()  # --init warmed the weights; count the worker's loads only

    result = _run("--cluster", "debug", "-i", position, "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    assert "Segmentation complete" in result.output
    with open_ome_zarr(out / "B" / "1" / "0", mode="r") as ds:
        assert np.all(np.asarray(ds.data) == 1)  # the stand-in labels every pixel 1
    assert fake_cellpose.built == ["cpsam_v2"]  # one model load for the whole position
    assert len(fake_cellpose.evaluated) == 3  # one per timepoint


def test_segment_resume_skips_finished_work(
    fake_cellpose, cpu_cellpose, example_plate, segment_config, tmp_path
):
    plate_path, _ = example_plate
    out, config = tmp_path / "seg.zarr", segment_config(version="0.5")  # resume needs Zarr v3
    position = plate_path / "B" / "1" / "0"
    assert _run("--init", "-i", position, "-o", out, "-c", config).exit_code == 0
    assert _run("--cluster", "debug", "-i", position, "-o", out, "-c", config).exit_code == 0
    n_first = len(fake_cellpose.evaluated)

    result = _run("--cluster", "debug", "--resume", "-i", position, "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    assert len(fake_cellpose.evaluated) == n_first


def test_segment_legacy_local_flag_is_gone(example_plate, segment_config, tmp_path):
    plate_path, _ = example_plate
    result = _run(
        "--local",
        "-i",
        plate_path / "A" / "1" / "0",
        "-o",
        tmp_path / "o.zarr",
        "-c",
        segment_config(),
    )
    assert result.exit_code == 2
    assert "No such option '--local'" in result.output


class _RecordingExecutor:
    """Stand-in for submitit.AutoExecutor that records parameters and runs nothing."""

    parameters: dict = {}

    def __init__(self, folder, cluster):
        _RecordingExecutor.parameters = {"cluster": cluster}

    def update_parameters(self, **kwargs):
        _RecordingExecutor.parameters.update(kwargs)

    def batch(self):
        import contextlib

        return contextlib.nullcontext()

    def submit(self, *args, **kwargs):
        class Job:
            job_id = "0"

            def wait(self):
                pass

        return Job()


@pytest.mark.parametrize("cluster, expected", [("local", 1), ("slurm", None), ("debug", None)])
def test_segment_gives_local_jobs_a_gpu(
    fake_cellpose, example_plate, segment_config, tmp_path, monkeypatch, cluster, expected
):
    """submitit's local executor only exposes GPUs through gpus_per_node; without it the
    local subprocess runs with CUDA_VISIBLE_DEVICES='' and cellpose_device refuses."""
    monkeypatch.setattr("biahub.segment.submitit.AutoExecutor", _RecordingExecutor)
    monkeypatch.delenv("CI", raising=False)  # CI forces cluster=debug
    plate_path, _ = example_plate
    out = tmp_path / "seg.zarr"

    result = _run(
        "--cluster",
        cluster,
        "-i",
        plate_path / "A" / "1" / "0",
        "-o",
        out,
        "-c",
        segment_config(),
    )

    assert result.exit_code == 0, result.output
    assert _RecordingExecutor.parameters["cluster"] == cluster
    assert _RecordingExecutor.parameters.get("gpus_per_node") == expected


def test_segment_worker_stages_weights_before_cellpose_is_imported(
    fake_cellpose, cpu_cellpose, example_plate, segment_config, tmp_path, monkeypatch
):
    """Validating eval_args imports cellpose.models, which fixes the weights directory;
    staging after that is a no-op and every worker reads the checkpoint over NFS."""
    import biahub.segment
    import biahub.settings

    calls = []
    real_valid_args = biahub.settings.get_valid_eval_args
    monkeypatch.setattr(
        biahub.segment, "stage_cellpose_weights", lambda: calls.append("stage")
    )
    monkeypatch.setattr(
        biahub.settings,
        "get_valid_eval_args",
        lambda: calls.append("import cellpose") or real_valid_args(),
    )
    plate_path, _ = example_plate
    out, config = tmp_path / "seg.zarr", segment_config()
    position = plate_path / "B" / "1" / "0"
    assert _run("--init", "-i", position, "-o", out, "-c", config).exit_code == 0
    calls.clear()

    assert _run("--cluster", "debug", "-i", position, "-o", out, "-c", config).exit_code == 0

    assert calls and calls[0] == "stage", calls


def _resources(output):
    line = next(ln for ln in output.splitlines() if ln.startswith("RESOURCES:"))
    return json.loads(line.removeprefix("RESOURCES:"))


def test_segment_requests_less_time_for_2d_than_3d(fake_cellpose, example_plate, tmp_path):
    """Calibrated on a real 2D run (~5 s/frame, 6 GB); 3D keeps the conservative rate."""
    plate_path, _ = example_plate
    position = plate_path / "A" / "1" / "0"
    res = {}
    for kind, model in {
        "2d": {"z_slice_2D": 1, "eval_args": {}},
        "3d": {"eval_args": {"do_3D": True}},
    }.items():
        cfg = tmp_path / f"{kind}.yml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "nuc": {"pretrained_model": "cpsam_v2", "channels": ["GFP"], **model}
                    }
                }
            )
        )
        result = _run("--init", "-i", position, "-o", tmp_path / f"{kind}.zarr", "-c", cfg)
        assert result.exit_code == 0, result.output
        res[kind] = _resources(result.output)
    assert res["2d"]["cpus"] <= 4
    assert res["2d"]["time_minutes"] < res["3d"]["time_minutes"]


def test_segment_local_jobs_get_the_step_time_limit(
    fake_cellpose, example_plate, segment_config, tmp_path, monkeypatch
):
    """submitit's local executor kills jobs after timeout_min (default 2 min)."""
    monkeypatch.setattr("biahub.segment.submitit.AutoExecutor", _RecordingExecutor)
    monkeypatch.delenv("CI", raising=False)
    plate_path, _ = example_plate

    result = _run(
        "--cluster",
        "local",
        "-i",
        plate_path / "A" / "1" / "0",
        "-o",
        tmp_path / "s.zarr",
        "-c",
        segment_config(),
    )

    assert result.exit_code == 0, result.output
    assert (
        _RecordingExecutor.parameters["timeout_min"]
        == _resources(result.output)["time_minutes"]
    )


def test_segment_init_checks_models_before_creating_the_plate(
    fake_cellpose, example_plate, segment_config, tmp_path
):
    plate_path, _ = example_plate
    out = tmp_path / "seg.zarr"
    result = _run(
        "--init",
        "-i",
        plate_path / "A" / "1" / "0",
        "-o",
        out,
        "-c",
        segment_config(pretrained_model="nuclei"),
    )
    assert result.exit_code != 0
    assert not out.exists()
