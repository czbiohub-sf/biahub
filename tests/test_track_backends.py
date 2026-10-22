"""biahub track with the tracksdata and HOCT trackers (``tracker:`` in the config)."""

import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr
from pydantic import ValidationError

from biahub.cli.main import cli
from biahub.settings import HoctConfig, TrackastraConfig, TrackingSettings, TracksdataConfig

INPUT_IMAGES = [{"path": None, "channels": {"nuclei_prediction": []}}]


def _settings(**kw):
    base = {
        "input_images": INPUT_IMAGES,
        "segmentation_method": "cellpose",
        "cellpose_config": {},
    }
    return TrackingSettings(**{**base, **kw})


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def _dividing_movie(T=6, size=64):
    """One nucleus for 3 frames, then two daughters side by side (labels and image)."""
    labels = np.zeros((T, size, size), dtype=np.int32)
    for t in range(3):
        labels[t][_disk((size, size), 32, 32, 7)] = 1
    for t in range(3, T):
        labels[t][_disk((size, size), 32, 26, 5)] = 1
        labels[t][_disk((size, size), 32, 38, 5)] = 2
    return labels, (labels > 0).astype(np.float32)


def _check_one_division(tracks: pd.DataFrame, tracked: np.ndarray):
    assert {"track_id", "t", "y", "x", "parent_track_id"} <= set(tracks.columns)
    roots = tracks[tracks.parent_track_id == -1].track_id.unique()
    children = tracks[tracks.parent_track_id != -1]
    assert len(roots) == 1
    assert children.track_id.nunique() == 2
    assert set(children.parent_track_id) == {roots[0]}
    assert tracks[tracks.track_id == roots[0]].t.max() == 2
    # Labels carry the track id, as ultrack's do.
    for row in tracks.itertuples():
        assert tracked[row.t, int(round(row.y)), int(round(row.x))] == row.track_id


# --- settings ---------------------------------------------------------------------


def test_ultrack_stays_the_default_tracker():
    s = _settings()
    assert s.tracker == "ultrack"
    assert isinstance(s.tracksdata_config, TracksdataConfig)
    assert isinstance(s.hoct_config, HoctConfig)


@pytest.mark.parametrize("tracker", ["tracksdata", "hoct", "trackastra"])
def test_label_trackers_need_cellpose_labels(tracker):
    with pytest.raises(ValidationError, match="segmentation_method: cellpose"):
        TrackingSettings(
            input_images=INPUT_IMAGES,
            tracker=tracker,
            segmentation_method="foreground_contour",
        )


@pytest.mark.parametrize("tracker", ["tracksdata", "hoct", "trackastra"])
def test_label_trackers_are_2d_only(tracker):
    with pytest.raises(ValidationError, match="2D"):
        _settings(tracker=tracker, output_mode="3D")


def test_tracker_defaults_are_the_benchmark_settings_in_um():
    # The A549 benchmark's pixel settings at 0.1133 um/px.
    td = TracksdataConfig()
    assert td.distance_threshold_um == pytest.approx(200 * 0.1133, abs=0.05)
    assert td.distance_norm_um == pytest.approx(100 * 0.1133, abs=0.05)
    assert td.border_margin_um == pytest.approx(60 * 0.1133, abs=0.05)
    assert (td.division_weight, td.appearance_weight) == (1.0, 5.0)
    assert HoctConfig().distance_threshold_um == pytest.approx(300 * 0.1133, abs=0.05)
    assert HoctConfig().model == "general_v1"


def test_trackastra_defaults():
    t = TrackastraConfig()
    assert (t.model, t.mode) == ("general_2d", "greedy")
    with pytest.raises(ValidationError):
        TrackastraConfig(mode="hungarian")


def test_unknown_tracker_settings_are_refused():
    with pytest.raises(ValidationError):
        TracksdataConfig(division_wieght=2.0)


# --- trackers on a synthetic division -----------------------------------------------


def test_tracksdata_tracks_a_division():
    from biahub.track import run_tracksdata

    labels, _ = _dividing_movie()

    tracked, tracks = run_tracksdata(labels, TracksdataConfig(), pixel_size=1.0)

    assert tracked.shape == labels.shape
    _check_one_division(tracks, tracked)


def _stub_hoct(monkeypatch, labels, calls):
    """Real HOCT graph construction (recorded); the model and predict are stubbed."""
    import hoct
    import hoct.features

    from biahub.track import _solve_tracksdata

    real_create_graph = hoct.features.create_graph

    def create_graph(labels_, **kw):
        calls["create_graph"] = kw
        return real_create_graph(labels_, **kw)

    def predict(model, *, graph, **kw):
        calls.update(model=model, graph=graph, **kw)
        return _solve_tracksdata(labels, TracksdataConfig(), pixel_size=1.0)

    monkeypatch.setattr(hoct.features, "create_graph", create_graph)
    monkeypatch.setattr(hoct, "load_model", lambda model, device="cpu": ("model", model))
    monkeypatch.setattr(hoct, "predict", predict)


def test_hoct_gets_pixel_settings_and_the_image(monkeypatch):
    from biahub.track import run_hoct

    labels, image = _dividing_movie()
    calls = {}
    _stub_hoct(monkeypatch, labels, calls)

    tracked, tracks = run_hoct(
        labels, image, HoctConfig(distance_threshold_um=3.0), pixel_size=0.5
    )

    graph_kw = calls["create_graph"]
    assert calls["model"] == ("model", "general_v1")
    assert graph_kw["distance_threshold"] == pytest.approx(6.0)  # 3 um at 0.5 um/px
    assert graph_kw["n_neighbors"] == 5 and graph_kw["delta_t"] == 3
    assert graph_kw.get("scale") is None  # pixel space, as the model was trained
    assert np.array_equal(graph_kw["images"], image)
    _check_one_division(tracks, tracked)


def test_hoct_solver_overrides_merge_into_its_defaults(monkeypatch):
    from hoct.tracking import ILPSolverConfig

    from biahub.track import run_hoct

    labels, image = _dividing_movie()
    calls = {}
    _stub_hoct(monkeypatch, labels, calls)

    run_hoct(labels, image, HoctConfig(solver={"division_weight": 1.0}), pixel_size=1.0)

    solver = calls["solver_config"]
    assert solver.division_weight == 1.0
    assert solver.node_weight == ILPSolverConfig.default().node_weight


@pytest.mark.parametrize("frames", [1, 2])
def test_hoct_without_candidate_links_keeps_each_detection(frames, monkeypatch):
    """HOCT's real graph construction finds no edges: no inference, isolated tracks."""
    import hoct

    from biahub.track import run_hoct

    def no_model(*args, **kw):
        raise AssertionError("HOCT inference must not run on an edgeless graph")

    monkeypatch.setattr(hoct, "load_model", no_model)
    monkeypatch.setattr(hoct, "predict", no_model)
    labels = np.zeros((frames, 64, 64), dtype=np.int32)
    labels[0, 5:10, 5:10] = 1
    labels[0, 50:55, 50:55] = 2
    if frames == 2:
        labels[1, 30:35, 5:10] = 1

    tracked, tracks = run_hoct(
        labels,
        labels.astype(np.float32),
        HoctConfig(distance_threshold_um=5.0),
        pixel_size=1.0,
    )

    assert tracked.shape == labels.shape
    assert len(tracks) == int((labels > 0).any(axis=(1, 2)).sum() + 1)
    assert tracks.track_id.is_unique and (tracks.parent_track_id == -1).all()


def _trackastra_graph(labels):
    """The graph Trackastra returns for the dividing movie: nodes (time, label), edges in time."""
    import networkx as nx

    graph = nx.DiGraph()
    for t in range(labels.shape[0]):
        for label in np.unique(labels[t])[1:]:
            graph.add_node((t, int(label)), time=t, label=int(label))
    for t in range(labels.shape[0] - 1):
        for a in np.unique(labels[t])[1:]:
            for b in np.unique(labels[t + 1])[1:]:
                if np.any((labels[t] == a) & (labels[t + 1] == b)):
                    graph.add_edge((t, int(a)), (t + 1, int(b)))
    return graph


def test_trackastra_gets_the_image_and_mode(monkeypatch):
    """Trackastra's model is stubbed (no download); its own graph_to_ctc converts."""
    import trackastra.model

    calls = {}

    class Model:
        def track(self, imgs, masks, mode="greedy", **kw):
            calls.update(imgs=imgs, masks=masks, mode=mode)
            return _trackastra_graph(masks), masks

    def from_pretrained(name, device=None, **kw):
        calls["name"] = name
        return Model()

    monkeypatch.setattr(
        trackastra.model.Trackastra, "from_pretrained", staticmethod(from_pretrained)
    )
    from biahub.track import run_trackastra

    labels, image = _dividing_movie()

    tracked, tracks = run_trackastra(labels, image, TrackastraConfig(mode="ilp"))

    assert calls["name"] == "general_2d" and calls["mode"] == "ilp"
    assert np.array_equal(calls["imgs"], image)
    assert tracked.shape == labels.shape
    _check_one_division(tracks, tracked)


def _trackastra_weights_cached():
    from platformdirs import user_data_dir

    return (pathlib.Path(user_data_dir("trackastra")) / "models" / "general_2d").exists()


@pytest.mark.skipif(
    not _trackastra_weights_cached(), reason="Trackastra weights not downloaded"
)
def test_trackastra_real_model_tracks_a_division():
    from biahub.track import run_trackastra

    labels, image = _dividing_movie()
    tracked, tracks = run_trackastra(labels, image, TrackastraConfig())
    assert tracked.shape == labels.shape
    _check_one_division(tracks, tracked)


def _hoct_weights_cached():
    try:
        import hoct._models as models

        return (models._cache_dir() / "general_v1.pt").exists()
    except Exception:
        return False


@pytest.mark.skipif(not _hoct_weights_cached(), reason="HOCT weights not downloaded")
def test_hoct_real_model_tracks_a_division():
    from biahub.track import run_hoct

    labels, image = _dividing_movie()
    tracked, tracks = run_hoct(labels, image, HoctConfig(), pixel_size=0.1133)
    assert tracked.shape == labels.shape  # HOCT's internal (T, 1, Y, X) is squeezed back
    _check_one_division(tracks, tracked)


# --- the CLI ------------------------------------------------------------------------


@pytest.fixture
def dividing_plate(tmp_path):
    labels, image = _dividing_movie(T=6, size=64)
    plate_path = tmp_path / "plate.zarr"
    with open_ome_zarr(
        plate_path, layout="hcs", mode="w", channel_names=["nuclei_prediction"]
    ) as plate:
        position = plate.create_position("A", "1", "0")
        position.create_image("0", image[:, None, None])  # (T, C=1, Z=1, Y, X)
    return plate_path, labels


@pytest.mark.parametrize("tracker", ["tracksdata", "hoct", "trackastra"])
def test_track_debug_run_with_a_label_tracker(tracker, dividing_plate, tmp_path, monkeypatch):

    import biahub.track as track_mod

    plate_path, labels = dividing_plate
    # cellpose is stubbed: its labels are the synthetic ones.
    monkeypatch.setattr(track_mod, "run_cellpose_per_frame", lambda images, **kw: labels)
    monkeypatch.setattr(track_mod, "cellpose_device", lambda gpu: "cpu")
    monkeypatch.setattr(track_mod, "stage_cellpose_weights", lambda: None)
    monkeypatch.setattr(track_mod, "warm_cellpose_weights", lambda name: None)
    warmed = []
    monkeypatch.setattr(track_mod, "warm_hoct_weights", lambda model: warmed.append(model))
    monkeypatch.setattr(
        track_mod, "warm_trackastra_weights", lambda model: warmed.append(model)
    )
    import trackastra.model

    class Model:
        def track(self, imgs, masks, mode="greedy", **kw):
            return _trackastra_graph(masks), masks

    monkeypatch.setattr(
        trackastra.model.Trackastra,
        "from_pretrained",
        staticmethod(lambda name, **kw: Model()),
    )
    _stub_hoct(monkeypatch, labels, {})
    config = tmp_path / "track.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "output_mode": "2D",
                "target_channel": "nuclei_prediction",
                "input_images": INPUT_IMAGES,
                "segmentation_method": "cellpose",
                "cellpose_config": {"gpu": False},
                "tracker": tracker,
                "min_sibling_length": 1,
            }
        )
    )
    position, out = str(plate_path / "A" / "1" / "0"), str(tmp_path / "out.zarr")

    init = CliRunner().invoke(
        cli, ["track", "--init", "-i", position, "-o", out, "-c", str(config)]
    )
    assert init.exit_code == 0, init.output
    assert warmed == {"hoct": ["general_v1"], "trackastra": ["general_2d"]}.get(tracker, [])

    result = CliRunner().invoke(
        cli, ["track", "--cluster", "debug", "-i", position, "-o", out, "-c", str(config)]
    )

    assert result.exit_code == 0, result.output
    assert f"Tracking with {tracker}" in result.output
    assert "Sibling filter (min_length=1)" in result.output
    tracks = pd.read_csv(tmp_path / "out.zarr" / "A" / "1" / "0" / "tracks_A_1_0.csv")
    with open_ome_zarr(tmp_path / "out.zarr" / "A" / "1" / "0", mode="r") as ds:
        tracked = np.asarray(ds.data[:, 0, 0])
    _check_one_division(tracks, tracked)


def test_a_one_child_link_is_merged_into_its_parent():
    """Linking across a skipped frame splits a track into parent -> one child; the sibling
    filter (and ultrack's tables) only allow 0 or 2 children."""
    from biahub.track import _merge_single_children

    labels = np.zeros((5, 16, 16), dtype=np.int32)
    labels[0:2, 2:6, 2:6] = 1  # track 1, frames 0-1
    labels[3:5, 2:6, 2:6] = 2  # track 2 continues it after a skipped frame
    labels[0:5, 10:14, 10:14] = 3  # an unrelated track
    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2, 3, 3, 3, 3, 3],
            "t": [0, 1, 3, 4, 0, 1, 2, 3, 4],
            "y": [3.5] * 4 + [11.5] * 5,
            "x": [3.5] * 4 + [11.5] * 5,
            "parent_track_id": [-1, -1, 1, 1, -1, -1, -1, -1, -1],
        }
    )

    merged, relabeled = _merge_single_children(tracks, labels)

    assert sorted(merged.track_id.unique()) == [1, 3]
    assert (merged.parent_track_id == -1).all()
    assert merged[merged.track_id == 1].t.tolist() == [0, 1, 3, 4]
    assert set(np.unique(relabeled[3:5, 2:6, 2:6])) == {1}
    assert set(np.unique(relabeled[:, 10:14, 10:14])) == {3}


def test_merge_keeps_divisions_and_follows_chains():
    from biahub.track import _merge_single_children

    labels = np.zeros((4, 8, 8), dtype=np.int32)
    tracks = pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4, 5],
            "t": [0, 1, 2, 2, 3],
            "y": [0.0] * 5,
            "x": [0.0] * 5,
            # 1 -> 2 (one child), 2 -> 3 and 4 (a division), 4 -> 5 (one child)
            "parent_track_id": [-1, 1, 2, 2, 4],
        }
    )

    merged, _ = _merge_single_children(tracks, labels)

    parents = dict(zip(merged.track_id, merged.parent_track_id, strict=False))
    assert sorted(merged.track_id.unique()) == [1, 3, 4]
    assert parents[3] == 1 and parents[4] == 1  # the division survives, on the merged root
    assert merged[merged.track_id == 4].t.tolist() == [2, 3]


def test_label_trackers_need_a_cellpose_config():
    with pytest.raises(ValidationError, match="cellpose_config"):
        TrackingSettings(
            input_images=INPUT_IMAGES,
            tracker="trackastra",
            segmentation_method="cellpose",
            cellpose_config=None,
        )


def test_thread_count_without_cpu_affinity(monkeypatch):
    """os.sched_getaffinity does not exist on macOS or Windows."""
    import os

    from biahub.track import _num_threads

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert _num_threads() == (os.cpu_count() or 1)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert _num_threads() == 3


@pytest.mark.parametrize("frames", [1, 2])
def test_tracksdata_without_candidate_links_keeps_each_detection(frames):
    """No edges (one frame, or objects beyond distance_threshold_um): the solver has
    nothing to solve; each detection becomes its own track instead of an error."""
    from biahub.track import run_tracksdata

    labels = np.zeros((frames, 64, 64), dtype=np.int32)
    labels[0, 5:10, 5:10] = 1
    labels[0, 50:55, 50:55] = 2
    if frames == 2:
        labels[1, 30:35, 5:10] = 1  # 25 px away; threshold below is 5 px
    config = TracksdataConfig(distance_threshold_um=5.0)

    tracked, tracks = run_tracksdata(labels, config, pixel_size=1.0)

    assert tracked.shape == labels.shape
    assert len(tracks) == int((labels > 0).any(axis=(1, 2)).sum() + 1)  # one row per detection
    assert tracks.track_id.is_unique
    assert (tracks.parent_track_id == -1).all()
    assert {"track_id", "t", "y", "x", "parent_track_id"} <= set(tracks.columns)
    for row in tracks.itertuples():
        assert tracked[row.t, int(round(row.y)), int(round(row.x))] == row.track_id


def _check_empty(tracked, tracks, shape):
    assert tracked.shape == shape and not tracked.any()
    assert len(tracks) == 0
    assert list(tracks.columns) == ["track_id", "t", "y", "x", "parent_track_id"]


def test_tracksdata_on_an_empty_movie_returns_empty_outputs():
    from biahub.track import run_tracksdata

    labels = np.zeros((3, 32, 32), dtype=np.int32)
    _check_empty(*run_tracksdata(labels, TracksdataConfig(), pixel_size=1.0), labels.shape)


def test_tracksdata_selecting_nothing_returns_empty_outputs():
    """With the defaults a still interior cell over two frames costs more than nothing."""
    from biahub.track import run_tracksdata

    labels = np.zeros((2, 64, 64), dtype=np.int32)
    labels[:, 28:36, 28:36] = 1
    config = TracksdataConfig(
        node_weight=0.0, appearance_weight=50.0, disappearance_weight=50.0
    )

    _check_empty(*run_tracksdata(labels, config, pixel_size=1.0), labels.shape)


@pytest.mark.parametrize("tracker", ["tracksdata", "hoct", "trackastra"])
def test_an_empty_result_reaches_the_outputs(tracker, dividing_plate, tmp_path, monkeypatch):
    """No cells at all: the run still writes an empty CSV and zero labels, sibling filter on."""
    import biahub.track as track_mod

    plate_path, labels = dividing_plate
    monkeypatch.setattr(
        track_mod, "run_cellpose_per_frame", lambda images, **kw: np.zeros_like(labels)
    )
    monkeypatch.setattr(track_mod, "cellpose_device", lambda gpu: "cpu")
    monkeypatch.setattr(track_mod, "stage_cellpose_weights", lambda: None)
    monkeypatch.setattr(track_mod, "warm_cellpose_weights", lambda name: None)
    monkeypatch.setattr(track_mod, "warm_hoct_weights", lambda model: None)
    monkeypatch.setattr(track_mod, "warm_trackastra_weights", lambda model: None)
    config = tmp_path / "track.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "output_mode": "2D",
                "target_channel": "nuclei_prediction",
                "input_images": INPUT_IMAGES,
                "segmentation_method": "cellpose",
                "cellpose_config": {"gpu": False},
                "tracker": tracker,
                "min_sibling_length": 3,
            }
        )
    )
    position, out = str(plate_path / "A" / "1" / "0"), str(tmp_path / "out.zarr")
    assert (
        CliRunner()
        .invoke(cli, ["track", "--init", "-i", position, "-o", out, "-c", str(config)])
        .exit_code
        == 0
    )

    result = CliRunner().invoke(
        cli, ["track", "--cluster", "debug", "-i", position, "-o", out, "-c", str(config)]
    )

    assert result.exit_code == 0, result.output
    tracks = pd.read_csv(tmp_path / "out.zarr" / "A" / "1" / "0" / "tracks_A_1_0.csv")
    assert len(tracks) == 0 and "track_id" in tracks.columns
    with open_ome_zarr(tmp_path / "out.zarr" / "A" / "1" / "0", mode="r") as ds:
        assert not np.asarray(ds.data).any()


def test_hoct_solver_override_typos_are_refused(monkeypatch):
    from biahub.track import run_hoct

    labels, image = _dividing_movie()
    _stub_hoct(monkeypatch, labels, {})
    with pytest.raises(ValueError, match="divison_weight"):
        run_hoct(labels, image, HoctConfig(solver={"divison_weight": 1.0}), pixel_size=1.0)


def test_a_missing_hoct_says_how_to_get_it(monkeypatch):
    """HOCT is installed only on Python < 3.14 (its gurobipy pin has no 3.14 wheels)."""
    import sys

    from biahub.track import warm_hoct_weights

    monkeypatch.setitem(sys.modules, "hoct", None)  # import hoct -> ImportError
    with pytest.raises(ImportError, match="Python < 3.14"):
        warm_hoct_weights("general_v1")


def test_hoct_solver_preference_is_refused_in_the_settings():
    """The pinned tracksdata solver has no solver_preference; it picks Gurobi or SCIP."""
    with pytest.raises(ValidationError, match="solver_preference"):
        HoctConfig(solver={"solver_preference": "scip"})


def test_track_init_checks_the_hoct_solver_override(dividing_plate, tmp_path, monkeypatch):
    """A bad solver key fails at --init, before any worker runs cellpose."""
    import biahub.track as track_mod

    plate_path, _ = dividing_plate
    monkeypatch.setattr(track_mod, "warm_cellpose_weights", lambda name: None)
    monkeypatch.setattr(track_mod, "warm_hoct_weights", lambda model: None)
    config = tmp_path / "track.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "output_mode": "2D",
                "target_channel": "nuclei_prediction",
                "input_images": INPUT_IMAGES,
                "segmentation_method": "cellpose",
                "cellpose_config": {"gpu": False},
                "tracker": "hoct",
                "hoct_config": {"solver": {"divison_weight": 1.0}},
            }
        )
    )
    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--init",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config),
        ],
    )
    assert result.exit_code != 0
    assert "divison_weight" in str(result.exception) + result.output


def test_trackastra_on_one_frame_keeps_each_detection(monkeypatch):
    """Trackastra needs two frames; one frame gives isolated tracks without the model."""
    import trackastra.model

    from biahub.track import run_trackastra

    def no_model(*args, **kw):
        raise AssertionError("Trackastra must not run on one frame")

    monkeypatch.setattr(trackastra.model.Trackastra, "from_pretrained", staticmethod(no_model))
    labels = np.zeros((1, 32, 32), dtype=np.int32)
    labels[0, 2:8, 2:8] = 1
    labels[0, 20:26, 20:26] = 2

    tracked, tracks = run_trackastra(labels, labels.astype(np.float32), TrackastraConfig())

    assert tracked.shape == labels.shape
    assert len(tracks) == 2 and tracks.track_id.is_unique
    assert (tracks.parent_track_id == -1).all()
    for row in tracks.itertuples():
        assert tracked[row.t, int(round(row.y)), int(round(row.x))] == row.track_id


@pytest.mark.skipif(not _hoct_weights_cached(), reason="HOCT weights not downloaded")
def test_hoct_real_solver_path_with_an_override():
    from biahub.track import run_hoct

    labels, image = _dividing_movie()
    tracked, tracks = run_hoct(
        labels, image, HoctConfig(solver={"timeout": 600.0}), pixel_size=0.1133
    )
    _check_one_division(tracks, tracked)


def _one_link_movie():
    labels = np.zeros((2, 64, 64), dtype=np.int32)
    labels[0, 28:36, 28:36] = 1
    labels[1, 29:37, 29:37] = 1
    return labels


def test_hoct_with_a_single_candidate_link_skips_inference(monkeypatch):
    """HOCT 0.3.0's inference skips batches with <= 1 edge and then fails on none."""
    import hoct

    from biahub.track import run_hoct

    def no_model(*args, **kw):
        raise AssertionError("HOCT inference must not run on a single-edge graph")

    monkeypatch.setattr(hoct, "load_model", no_model)
    monkeypatch.setattr(hoct, "predict", no_model)
    labels = _one_link_movie()

    tracked, tracks = run_hoct(labels, labels.astype(np.float32), HoctConfig(), pixel_size=1.0)

    assert tracks.track_id.nunique() == 1 and tracks.t.tolist() == [0, 1]
    assert (tracks.parent_track_id == -1).all()
    assert set(np.unique(tracked[labels > 0])) == {tracks.track_id.iloc[0]}


@pytest.mark.skipif(not _hoct_weights_cached(), reason="HOCT weights not downloaded")
def test_hoct_real_model_on_a_single_candidate_link():
    from biahub.track import run_hoct

    labels = _one_link_movie()
    tracked, tracks = run_hoct(
        labels, labels.astype(np.float32), HoctConfig(), pixel_size=0.1133
    )
    assert tracks.track_id.nunique() == 1 and len(tracks) == 2
