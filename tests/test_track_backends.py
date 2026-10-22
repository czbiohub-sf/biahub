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


def test_hoct_gets_pixel_settings_and_the_image(monkeypatch):
    """HOCT's model is stubbed (no download); the stub solves with tracksdata."""
    import hoct

    from biahub.track import _solve_tracksdata, run_hoct

    calls = {}
    monkeypatch.setattr(hoct, "load_model", lambda model, device="cpu": ("model", model))

    def predict(model, *, labels, images, distance_threshold, n_neighbors, max_delta_t, **kw):
        calls.update(
            model=model,
            images=images,
            distance_threshold=distance_threshold,
            n_neighbors=n_neighbors,
            max_delta_t=max_delta_t,
            **kw,
        )
        return _solve_tracksdata(labels, TracksdataConfig(), pixel_size=1.0)

    monkeypatch.setattr(hoct, "predict", predict)
    labels, image = _dividing_movie()

    tracked, tracks = run_hoct(
        labels, image, HoctConfig(distance_threshold_um=3.0), pixel_size=0.5
    )

    assert calls["model"] == ("model", "general_v1")
    assert calls["distance_threshold"] == pytest.approx(6.0)  # 3 um at 0.5 um/px
    assert calls["n_neighbors"] == 5 and calls["max_delta_t"] == 3
    assert calls.get("scale") is None  # pixel space, as the model was trained
    assert np.array_equal(calls["images"], image)
    _check_one_division(tracks, tracked)


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
    import hoct

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
    monkeypatch.setattr(hoct, "load_model", lambda model, device="cpu": "model")
    monkeypatch.setattr(
        hoct,
        "predict",
        lambda model, *, labels, **kw: track_mod._solve_tracksdata(
            labels, TracksdataConfig(), pixel_size=1.0
        ),
    )
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
