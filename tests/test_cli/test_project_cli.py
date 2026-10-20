"""biahub project: 3D plate -> 2D plate, one projection method per channel."""

import json

import numpy as np
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli

Z = 8


@pytest.fixture
def plate(create_custom_plate, tmp_path):
    """(T=3, C=3, Z=8, Y=5, X=6) uint16, channels GFP, RFP, Phase3D."""
    return create_custom_plate(tmp_path, z_size=Z)


def _config(tmp_path, channels, focus=None, **extra):
    config = {"channels": channels, **extra}
    if focus is not None:
        config["focus"] = focus
    path = tmp_path / "project.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))  # channel order is output order
    return path


def _run(*args):
    return CliRunner().invoke(cli, ["project", *map(str, args)])


def _read(position_path):
    with open_ome_zarr(str(position_path), mode="r") as ds:
        return np.asarray(ds.data), ds.channel_names, dict(ds.zattrs)


def _fake_focus(monkeypatch, planes):
    """Make find_focus return the given plane for each timepoint, in order."""
    it = iter(planes)
    monkeypatch.setattr(
        "biahub.registration.methods.focus.find_focus", lambda *a, **k: next(it)
    )


def test_project_init_scaffolds_a_z1_plate_and_names_dropped_channels(plate, tmp_path):
    plate_path, _ = plate
    out = tmp_path / "proj.zarr"
    config = _config(
        tmp_path, {"RFP": {"method": "max"}, "GFP": {"method": "slice", "z": {"center": 2}}}
    )

    result = _run("--init", "-i", plate_path / "A" / "1" / "0", "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    line = next(ln for ln in result.output.splitlines() if ln.startswith("RESOURCES:"))
    assert set(json.loads(line.removeprefix("RESOURCES:"))) == {
        "cpus",
        "mem_gb",
        "time_minutes",
    }
    assert "dropped: ['Phase3D']" in result.output
    data, channel_names, zattrs = _read(out / "A" / "1" / "0")
    assert channel_names == ["RFP", "GFP"]  # config order, not input order
    assert data.shape == (3, 2, 1, 5, 6)
    assert data.dtype == np.float32  # any max/mean/sum channel makes the plate float32
    assert zattrs["biahub-project"]["channels"]["GFP"]["z"]["center"] == 2
    assert np.all(data == 0)  # --init projects nothing


def test_project_methods_match_numpy(plate, tmp_path):
    plate_path, _ = plate
    out = tmp_path / "proj.zarr"
    config = _config(
        tmp_path,
        {
            "GFP": {"method": "max", "z": {"start": 2, "stop": 7, "step": 2}},
            "RFP": {"method": "mean"},
            "Phase3D": {"method": "slice", "z": {"center": 3}},
        },
    )
    position = plate_path / "B" / "1" / "0"

    result = _run("--cluster", "debug", "-i", position, "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    assert "Projection complete" in result.output
    source, _, _ = _read(position)
    data, channel_names, _ = _read(out / "B" / "1" / "0")
    assert channel_names == ["GFP", "RFP", "Phase3D"]
    assert data.dtype == np.float32  # mean promotes the whole plate
    assert np.array_equal(data[:, 0, 0], source[:, 0, 2:7:2].max(axis=1))
    np.testing.assert_allclose(data[:, 1, 0], source[:, 1].mean(axis=1, dtype=np.float64))
    assert np.array_equal(data[:, 2, 0], source[:, 2, 3])


def test_project_sum_over_a_window_around_a_center(plate, tmp_path):
    plate_path, _ = plate
    out = tmp_path / "proj.zarr"
    config = _config(
        tmp_path, {"GFP": {"method": "sum", "z": {"center": 6, "below": 2, "above": 5}}}
    )
    position = plate_path / "A" / "1" / "0"

    result = _run("--cluster", "debug", "-i", position, "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    source, _, _ = _read(position)
    data, _, _ = _read(out / "A" / "1" / "0")
    # 4..10 clipped to the 8-plane stack: 4..7
    np.testing.assert_allclose(data[:, 0, 0], source[:, 0, 4:8].sum(axis=1, dtype=np.float64))


def test_project_focus_per_fov_takes_the_median_plane(plate, tmp_path, monkeypatch):
    plate_path, _ = plate
    out = tmp_path / "proj.zarr"
    config = _config(
        tmp_path,
        {
            "GFP": {"method": "mean", "z": {"center": "focus", "below": 1, "above": 2}},
            "Phase3D": {"method": "slice", "z": {"center": "focus"}},
        },
        focus={"channel": "Phase3D", "per": "fov"},
    )
    position = plate_path / "A" / "1" / "0"
    _fake_focus(monkeypatch, [1, 6, 2])  # one per timepoint; median 2

    result = _run("--cluster", "debug", "-i", position, "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    source, _, _ = _read(position)
    data, _, zattrs = _read(out / "A" / "1" / "0")
    np.testing.assert_allclose(data[:, 0, 0], source[:, 0, 1:4].mean(axis=1, dtype=np.float64))
    assert np.array_equal(data[:, 1, 0], source[:, 2, 2])
    # The record carries the focus settings (minus the engine's `axes`) plus the result.
    assert zattrs["biahub-project"]["focus"] == {
        "channel": "Phase3D",
        "per": "fov",
        "center_crop_xy": [800, 800],
        "na_det": 1.35,
        "lambda_ill": 0.5,
        "z": 2,
        "z_per_t": [1, 6, 2],
    }


def test_project_focus_per_timepoint_follows_the_plane(plate, tmp_path, monkeypatch):
    plate_path, _ = plate
    out = tmp_path / "proj.zarr"
    config = _config(
        tmp_path,
        {
            "GFP": {"method": "max", "z": {"center": "focus", "below": 1, "above": 2}},
            "Phase3D": {"method": "slice", "z": {"center": "focus"}},
        },
        focus={"channel": "Phase3D", "per": "timepoint"},
    )
    position = plate_path / "A" / "1" / "0"
    planes = [1, 6, 2]
    _fake_focus(monkeypatch, planes)

    result = _run("--cluster", "debug", "-i", position, "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    source, _, _ = _read(position)
    data, _, zattrs = _read(out / "A" / "1" / "0")
    for t, plane in enumerate(planes):
        assert np.array_equal(data[t, 0, 0], source[t, 0, plane - 1 : plane + 2].max(axis=0))
        assert np.array_equal(data[t, 1, 0], source[t, 2, plane])
    assert zattrs["biahub-project"]["focus"]["z"] is None
    assert zattrs["biahub-project"]["focus"]["z_per_t"] == planes


def test_project_focus_channel_need_not_be_an_output(plate, tmp_path, monkeypatch):
    plate_path, _ = plate
    out = tmp_path / "proj.zarr"
    config = _config(
        tmp_path,
        {"GFP": {"method": "slice", "z": {"center": "focus"}}},
        focus={"channel": "Phase3D"},
    )
    _fake_focus(monkeypatch, [5, 5, 5])

    result = _run("--cluster", "debug", "-i", plate_path / "A/1/0", "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    source, _, _ = _read(plate_path / "A/1/0")
    data, channel_names, _ = _read(out / "A" / "1" / "0")
    assert channel_names == ["GFP"]
    assert data.dtype == np.uint16  # an all-slice plate keeps the input dtype
    assert np.array_equal(data[:, 0, 0], source[:, 0, 5])


@pytest.mark.parametrize(
    "channels, message",
    [
        ({"GFP": {"method": "max", "z": {"start": 10, "stop": 20}}}, "select nothing"),
        ({"GFP": {"method": "slice", "z": {"center": 42}}}, "select nothing"),
        ({"DAPI": {"method": "max"}}, "not in the input channels"),
    ],
)
def test_project_init_fails_on_a_config_the_plate_cannot_satisfy(
    plate, tmp_path, channels, message
):
    plate_path, _ = plate
    out = tmp_path / "proj.zarr"

    result = _run(
        "--init", "-i", plate_path / "A/1/0", "-o", out, "-c", _config(tmp_path, channels)
    )

    assert result.exit_code != 0
    assert message in str(result.exception)
    assert not out.exists()


def test_project_refuses_a_plate_that_is_already_2d(create_custom_plate, tmp_path):
    plate_path, _ = create_custom_plate(tmp_path, z_size=1)
    config = _config(tmp_path, {"GFP": {"method": "max"}})

    result = _run(
        "--init", "-i", plate_path / "A/1/0", "-o", tmp_path / "o.zarr", "-c", config
    )

    assert result.exit_code != 0
    assert "already has one Z-plane" in str(result.exception)


def test_focus_pre_pass_in_parallel_matches_serial(plate):
    """The timepoints are independent; the pool gives the same planes as the loop."""
    from biahub.project import _find_focus_per_t
    from biahub.settings import ProjectionFocusSettings

    plate_path, _ = plate
    position = plate_path / "A" / "1" / "0"
    focus = ProjectionFocusSettings(channel="Phase3D")

    serial = _find_focus_per_t(position, 2, focus, pixel_size=0.1, num_workers=1)
    parallel = _find_focus_per_t(position, 2, focus, pixel_size=0.1, num_workers=2)

    assert serial.shape == (3,)
    assert np.array_equal(serial, parallel)
    assert np.all((0 <= serial) & (serial < Z))


def test_project_writes_the_configured_chunks_and_shards(plate, tmp_path):
    """Shards spanning timepoints: iohub writes one shard's timepoints per unit."""
    plate_path, _ = plate
    out = tmp_path / "proj.zarr"
    config = _config(
        tmp_path,
        {"GFP": {"method": "max"}, "Phase3D": {"method": "slice", "z": {"center": 1}}},
        output_ome_zarr_version="0.5",
        chunks_czyx=[1, 1, 4, 4],
        shards_ratio=[2, 1, 1, 2, 2],  # (2, 1, 1, 8, 8): two timepoints per shard
    )
    position = plate_path / "A" / "1" / "0"

    result = _run("--cluster", "debug", "-i", position, "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    with open_ome_zarr(str(out / "A" / "1" / "0"), mode="r") as ds:
        assert ds.data.chunks == (1, 1, 1, 4, 4)
        assert ds.data.shards == (2, 1, 1, 8, 8)
    source, _, _ = _read(position)
    data, _, _ = _read(out / "A" / "1" / "0")
    assert np.array_equal(data[:, 0, 0], source[:, 0].max(axis=1))  # all 3 timepoints
    assert np.array_equal(data[:, 1, 0], source[:, 2, 1])


def test_project_resume_skips_finished_timepoints(plate, tmp_path):
    plate_path, _ = plate
    out = tmp_path / "proj.zarr"
    config = _config(
        tmp_path, {"GFP": {"method": "max"}}, output_ome_zarr_version="0.5"
    )  # resume needs Zarr v3
    position = plate_path / "A" / "1" / "0"
    assert _run("--cluster", "debug", "-i", position, "-o", out, "-c", config).exit_code == 0
    before, _, _ = _read(out / "A" / "1" / "0")

    result = _run("--cluster", "debug", "--resume", "-i", position, "-o", out, "-c", config)

    assert result.exit_code == 0, result.output
    assert "already written" in result.output
    after, _, _ = _read(out / "A" / "1" / "0")
    assert np.array_equal(before, after)
