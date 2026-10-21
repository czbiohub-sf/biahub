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
def plate_2d(create_custom_plate, tmp_path):
    """A projected (Z=1) plate, what 2D models segment: (T=3, C=3, Z=1, Y=5, X=6)."""
    return create_custom_plate(tmp_path, z_size=1)


@pytest.fixture
def cpu_cellpose(monkeypatch):
    """Workers ask for the GPU; run the (fake) model on the CPU instead."""
    monkeypatch.setattr("biahub.segment.cellpose_device", lambda gpu: torch.device("cpu"))


@pytest.fixture(autouse=True)
def run_in_tmp_path(monkeypatch, tmp_path):
    """Without -o the submitit folder is ./slurm_output: keep it out of the repo."""
    monkeypatch.chdir(tmp_path)


def _run(*args):
    return CliRunner().invoke(cli, ["segment", *map(str, args)])


def test_segment_init_prints_resources_and_stamps_provenance(
    fake_cellpose, plate_2d, segment_config, tmp_path
):
    plate_path, _ = plate_2d  # (T=3, C=3, Z=1, Y=5, X=6)
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


# ---------------------------------------------------------------------------
# No -o: the labels go into the -i plate as label images, labels/<model>
# ---------------------------------------------------------------------------


def test_segment_init_creates_label_images_in_the_input_plate(
    fake_cellpose, plate_2d, segment_config, tmp_path
):
    plate_path, _ = plate_2d  # (T=3, C=3, Z=1, Y=5, X=6)
    positions = [plate_path / "A" / "1" / "0", plate_path / "B" / "1" / "0"]

    result = _run("--init", "-i", *positions, "-c", segment_config())

    assert result.exit_code == 0, result.output
    assert "RESOURCES:" in result.output
    assert "Created labels/{nuc}" in result.output
    for position in positions:
        with open_ome_zarr(position, mode="r") as pos:
            assert pos.label_names() == ["nuc"]
            label = pos.get_label("nuc")["0"]
            assert label.shape == (3, 1, 5, 6)  # the image's (T, Z, Y, X)
            assert label.dtype == np.uint32
            assert not label[:].any()
            assert pos.channel_names == ["GFP", "RFP", "Phase3D"]  # the image is untouched
        attrs = yaml.safe_load((position / "labels" / "nuc" / "zarr.json").read_text())
        # Provenance on the label image: the config as written (names, not indices).
        assert attrs["attributes"]["biahub-segment"]["models"]["nuc"]["channels"] == ["GFP"]
    assert fake_cellpose.evaluated == []


def test_segment_debug_fills_the_label_image(
    fake_cellpose, cpu_cellpose, plate_2d, segment_config, tmp_path
):
    plate_path, _ = plate_2d
    config = segment_config()
    position = plate_path / "B" / "1" / "0"
    assert _run("--init", "-i", position, "-c", config).exit_code == 0
    fake_cellpose.built.clear()

    result = _run("--cluster", "debug", "-i", position, "-c", config)

    assert result.exit_code == 0, result.output
    assert "Segmentation complete" in result.output
    with open_ome_zarr(position, mode="r") as pos:
        assert np.all(pos.get_label("nuc")["0"][:] == 1)  # the stand-in labels every pixel 1
        assert pos.data.shape == (3, 3, 1, 5, 6)
    assert fake_cellpose.built == ["cpsam_v2"]
    assert len(fake_cellpose.evaluated) == 3  # one per timepoint


def test_segment_resume_skips_finished_label_images(
    fake_cellpose, cpu_cellpose, create_custom_plate, segment_config, tmp_path
):
    """Resume works on the label image as on an image (iohub output_label)."""
    from iohub.ngff.utils import create_empty_plate

    # A Zarr v3 (OME-Zarr 0.5) plate: resume needs it.
    plate_path = tmp_path / "v05.zarr"
    create_empty_plate(
        store_path=plate_path,
        position_keys=[("A", "1", "0")],
        channel_names=["GFP"],
        shape=(3, 1, 1, 5, 6),
        version="0.5",
    )
    with open_ome_zarr(plate_path / "A/1/0", mode="r+") as pos:
        pos["0"][:] = np.random.randint(1, 100, size=(3, 1, 1, 5, 6)).astype(np.float32)
    config = segment_config()
    position = plate_path / "A" / "1" / "0"
    assert _run("--init", "-i", position, "-c", config).exit_code == 0
    assert _run("--cluster", "debug", "-i", position, "-c", config).exit_code == 0
    n_first = len(fake_cellpose.evaluated)
    assert n_first == 3

    result = _run("--cluster", "debug", "--resume", "-i", position, "-c", config)

    assert result.exit_code == 0, result.output
    assert len(fake_cellpose.evaluated) == n_first


def test_segment_writes_one_label_image_per_model(
    fake_cellpose, cpu_cellpose, plate_2d, tmp_path
):
    plate_path, _ = plate_2d
    position = plate_path / "B" / "1" / "0"
    config = _two_model_config(tmp_path / "two.yml", ["nuc", "mem"])

    assert _run("--cluster", "debug", "-i", position, "-c", config).exit_code == 0

    with open_ome_zarr(position, mode="r") as pos:
        assert pos.label_names() == ["mem", "nuc"]
        for name in ("nuc", "mem"):
            assert np.all(pos.get_label(name)["0"][:] == 1)
    assert len(fake_cellpose.evaluated) == 6  # 3 timepoints x 2 models


def test_segment_refuses_a_label_image_of_another_shape(
    fake_cellpose, plate_2d, example_plate, tmp_path
):
    """A model renamed from 3D to 2D would otherwise write into the old stack."""
    model = {"pretrained_model": "cpsam_v2", "channels": ["GFP"]}
    three_d = tmp_path / "3d.yml"
    three_d.write_text(
        yaml.safe_dump({"models": {"nuc": {**model, "eval_args": {"do_3D": True}}}})
    )
    stack = example_plate[0] / "B" / "1" / "0"  # Z=4
    assert _run("--init", "-i", stack, "-c", three_d).exit_code == 0
    with open_ome_zarr(stack, mode="r+") as pos:
        pos.get_label("nuc")["0"].resize((3, 2, 5, 6))  # pretend the image changed

    result = _run("--init", "-i", stack, "-c", three_d)

    assert result.exit_code != 0
    assert "already has a label image 'nuc'" in str(result.exception)


def test_segment_resume_token_names_the_model_in_the_input_plate(tmp_path):
    from biahub.segment import _resume_token
    from biahub.settings import SegmentationSettings
    from biahub.utils.config import yaml_to_model

    a = yaml_to_model(
        _two_model_config(tmp_path / "a.yml", ["nuc", "mem"]), SegmentationSettings
    )
    b = yaml_to_model(
        _two_model_config(tmp_path / "b.yml", ["mem", "nuc"]), SegmentationSettings
    )

    # Each model has its own label image, so the model order does not matter there.
    assert _resume_token(a, "nuc") == _resume_token(b, "nuc")
    assert _resume_token(a, "nuc") != _resume_token(a, "mem")


# ---------------------------------------------------------------------------
# -o: a separate label plate, one channel per model
# ---------------------------------------------------------------------------


def test_segment_init_rejects_an_unknown_model(
    fake_cellpose, plate_2d, segment_config, tmp_path
):
    plate_path, _ = plate_2d
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
    fake_cellpose, cpu_cellpose, plate_2d, segment_config, tmp_path
):
    plate_path, _ = plate_2d
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
    fake_cellpose, cpu_cellpose, plate_2d, segment_config, tmp_path
):
    plate_path, _ = plate_2d
    out, config = tmp_path / "seg.zarr", segment_config(version="0.5")  # resume needs Zarr v3
    position = plate_path / "B" / "1" / "0"
    assert _run("--init", "-i", position, "-o", out, "-c", config).exit_code == 0
    assert _run("--cluster", "debug", "-i", position, "-o", out, "-c", config).exit_code == 0
    n_first = len(fake_cellpose.evaluated)

    result = _run("--cluster", "debug", "--resume", "-i", position, "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    assert len(fake_cellpose.evaluated) == n_first


def test_segment_legacy_local_flag_is_gone(plate_2d, segment_config, tmp_path):
    plate_path, _ = plate_2d
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
    fake_cellpose, plate_2d, segment_config, tmp_path, monkeypatch, cluster, expected
):
    """submitit's local executor only exposes GPUs through gpus_per_node; without it the
    local subprocess runs with CUDA_VISIBLE_DEVICES='' and cellpose_device refuses."""
    monkeypatch.setattr("biahub.segment.submitit.AutoExecutor", _RecordingExecutor)
    monkeypatch.delenv("CI", raising=False)  # CI forces cluster=debug
    plate_path, _ = plate_2d
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
    fake_cellpose, cpu_cellpose, plate_2d, segment_config, tmp_path, monkeypatch
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
    plate_path, _ = plate_2d
    out, config = tmp_path / "seg.zarr", segment_config()
    position = plate_path / "B" / "1" / "0"
    assert _run("--init", "-i", position, "-o", out, "-c", config).exit_code == 0
    calls.clear()

    assert _run("--cluster", "debug", "-i", position, "-o", out, "-c", config).exit_code == 0

    assert calls and calls[0] == "stage", calls


def _resources(output):
    line = next(ln for ln in output.splitlines() if ln.startswith("RESOURCES:"))
    return json.loads(line.removeprefix("RESOURCES:"))


def test_segment_requests_less_time_for_2d_than_3d(
    fake_cellpose, plate_2d, example_plate, tmp_path
):
    """Calibrated on a real 2D run (~5 s/frame, 6 GB); 3D keeps the conservative rate."""
    positions = {
        "2d": plate_2d[0] / "A" / "1" / "0",  # Z=1
        "3d": example_plate[0] / "A" / "1" / "0",  # Z=4
    }
    res = {}
    for kind, model in {
        "2d": {"eval_args": {}},
        "3d": {"eval_args": {"do_3D": True}},
    }.items():
        position = positions[kind]
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


def test_segment_slurm_head_does_not_stage_weights(
    fake_cellpose, plate_2d, segment_config, tmp_path, monkeypatch
):
    """Staging on the submitting node would export its CELLPOSE_LOCAL_MODELS_PATH to every
    SLURM worker, which then skips its own staging. Only in-process workers stage early."""
    staged = []
    monkeypatch.setattr("biahub.segment.stage_cellpose_weights", lambda: staged.append(1))
    monkeypatch.setattr("biahub.segment.submitit.AutoExecutor", _RecordingExecutor)
    monkeypatch.delenv("CI", raising=False)
    plate_path, _ = plate_2d

    result = _run(
        "--cluster",
        "slurm",
        "-i",
        plate_path / "A" / "1" / "0",
        "-o",
        tmp_path / "s.zarr",
        "-c",
        segment_config(),
    )

    assert result.exit_code == 0, result.output
    assert staged == []


def test_segment_local_jobs_get_the_step_time_limit(
    fake_cellpose, plate_2d, segment_config, tmp_path, monkeypatch
):
    """submitit's local executor kills jobs after timeout_min (default 2 min)."""
    monkeypatch.setattr("biahub.segment.submitit.AutoExecutor", _RecordingExecutor)
    monkeypatch.delenv("CI", raising=False)
    plate_path, _ = plate_2d

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
    fake_cellpose, plate_2d, segment_config, tmp_path
):
    plate_path, _ = plate_2d
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


def test_segment_local_sbatch_file_overrides_computed_parameters(
    fake_cellpose, plate_2d, segment_config, tmp_path, monkeypatch
):
    """A '#LOCAL --timeout-min' line and the computed local timeout used to collide
    (TypeError: multiple values for keyword argument 'timeout_min')."""
    monkeypatch.setattr("biahub.segment.submitit.AutoExecutor", _RecordingExecutor)
    monkeypatch.delenv("CI", raising=False)  # conftest forces CI (debug) otherwise
    sbatch = tmp_path / "local.sbatch"
    sbatch.write_text("#LOCAL --timeout-min=5\n")
    plate_path, _ = plate_2d

    result = _run(
        "--cluster",
        "local",
        "-sb",
        sbatch,
        "-i",
        plate_path / "A" / "1" / "0",
        "-o",
        tmp_path / "s.zarr",
        "-c",
        segment_config(),
    )

    assert result.exit_code == 0, result.output
    assert _RecordingExecutor.parameters["timeout_min"] == 5  # the explicit setting wins
    assert _RecordingExecutor.parameters["gpus_per_node"] == 1


def _two_model_config(path, order):
    models = {
        "nuc": {"pretrained_model": "cpsam_v2", "channels": ["GFP"]},
        "mem": {"pretrained_model": "cpsam_v2", "channels": ["RFP"]},
    }
    config = {
        "models": {name: models[name] for name in order},
        "output_ome_zarr_version": "0.5",
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_segment_refuses_an_output_with_a_different_model_order(
    fake_cellpose, cpu_cellpose, plate_2d, tmp_path
):
    """Reordering the models used to resume into swapped label channels."""
    plate_path, _ = plate_2d
    out, position = tmp_path / "seg.zarr", plate_path / "B" / "1" / "0"
    first = _two_model_config(tmp_path / "first.yml", ["nuc", "mem"])
    assert _run("--init", "-i", position, "-o", out, "-c", first).exit_code == 0
    assert _run("--cluster", "debug", "-i", position, "-o", out, "-c", first).exit_code == 0
    n_first = len(fake_cellpose.evaluated)

    swapped = _two_model_config(tmp_path / "swapped.yml", ["mem", "nuc"])
    result = _run("--cluster", "debug", "--resume", "-i", position, "-o", out, "-c", swapped)

    assert result.exit_code != 0
    assert "['nuc_labels', 'mem_labels']" in str(result.exception)
    assert len(fake_cellpose.evaluated) == n_first  # nothing was segmented into it
    with open_ome_zarr(out / "B" / "1" / "0", mode="r") as ds:
        assert ds.channel_names == ["nuc_labels", "mem_labels"]


def test_segment_resume_token_depends_on_model_order(tmp_path):
    from biahub.segment import _resume_token
    from biahub.settings import SegmentationSettings
    from biahub.utils.config import yaml_to_model

    a = yaml_to_model(
        _two_model_config(tmp_path / "a.yml", ["nuc", "mem"]), SegmentationSettings
    )
    b = yaml_to_model(
        _two_model_config(tmp_path / "b.yml", ["mem", "nuc"]), SegmentationSettings
    )

    assert _resume_token(a) != _resume_token(b)
    assert _resume_token(a) == _resume_token(a.model_copy(deep=True))


def test_segment_mem_per_cpu_override_drops_the_computed_total(
    fake_cellpose, plate_2d, segment_config, tmp_path, monkeypatch
):
    """The --sbatch-filepath help's own example, '#SBATCH --mem-per-cpu=16G', left
    --mem set too, and SLURM rejects jobs with both."""
    monkeypatch.setattr("biahub.segment.submitit.AutoExecutor", _RecordingExecutor)
    monkeypatch.delenv("CI", raising=False)  # conftest forces CI (debug) otherwise
    sbatch = tmp_path / "mem.sbatch"
    sbatch.write_text("#SBATCH --mem-per-cpu=16G\n")
    plate_path, _ = plate_2d

    result = _run(
        "--cluster",
        "slurm",
        "-sb",
        sbatch,
        "-i",
        plate_path / "A" / "1" / "0",
        "-o",
        tmp_path / "s.zarr",
        "-c",
        segment_config(),
    )

    assert result.exit_code == 0, result.output
    assert _RecordingExecutor.parameters["slurm_mem_per_cpu"] == "16G"
    assert "slurm_mem" not in _RecordingExecutor.parameters


def test_segment_refuses_an_output_with_a_different_shape(
    fake_cellpose, plate_2d, example_plate, tmp_path
):
    """Switching a same-named model from 3D to 2D kept the old multi-plane array."""
    out = tmp_path / "seg.zarr"
    model = {"pretrained_model": "cpsam_v2", "channels": ["GFP"]}
    three_d = tmp_path / "3d.yml"
    three_d.write_text(
        yaml.safe_dump({"models": {"nuc": {**model, "eval_args": {"do_3D": True}}}})
    )
    two_d = tmp_path / "2d.yml"
    two_d.write_text(yaml.safe_dump({"models": {"nuc": model}}))
    stack = example_plate[0] / "B" / "1" / "0"  # Z=4
    assert _run("--init", "-i", stack, "-o", out, "-c", three_d).exit_code == 0

    result = _run("--init", "-i", plate_2d[0] / "B" / "1" / "0", "-o", out, "-c", two_d)

    assert result.exit_code != 0
    assert "(3, 1, 4, 5, 6)" in str(result.exception)
    with open_ome_zarr(out / "B" / "1" / "0", mode="r") as ds:
        assert ds.data.shape == (3, 1, 4, 5, 6)
