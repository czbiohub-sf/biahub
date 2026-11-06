"""The benchmark helpers the registration skill uses to choose and run a pipeline."""

import csv

import pytest


def _table(tmp_path, rows):
    """Rows of (case, pipeline, candidate, position, tiles, points[, points_n]); n=20 if absent."""
    path = tmp_path / "benchmark.csv"
    fields = ["case", "pipeline", "candidate", "position", "tiles_median_um"]
    fields += ["points_median_um", "points_n"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = tuple(row) + ((20,) if len(row) == 6 else ())
            writer.writerow(dict(zip(fields, row, strict=True)))
    return path


def _benchmark(tmp_path, ants_points=(0.6, 0.5), tiles_ants_points=(0.5, 0.6)):
    rows = []
    for pipeline, points in (("ants", ants_points), ("tiles_ants", tiles_ants_points)):
        for k, position in enumerate(("A/1/0", "A/2/0")):
            rows.append(("c", pipeline, "unregistered", position, 3.0, 3.0))
            rows.append(("c", pipeline, "transforms", position, 0.5, points[k]))
            rows.append(
                ("c", pipeline, "1:block", position, 1.0, 2.0)
            )  # blocks are not choices
    return _table(tmp_path, rows)


def test_the_winner_is_the_lowest_median_on_held_out_positions(tmp_path):
    from biahub.registration.benchmark import select_pipeline

    choice = select_pipeline(_benchmark(tmp_path, ants_points=(1.0, 1.2)))
    assert choice["pipeline"] == "tiles_ants" and choice["metric"] == "points_median_um"
    assert choice["median_um"] == pytest.approx(0.55)
    assert "tiles_ants" in choice["reason"]


def test_a_tie_goes_to_the_simpler_pipeline(tmp_path):
    # equal within the tolerance: fewer blocks wins
    from biahub.registration.benchmark import select_pipeline

    choice = select_pipeline(_benchmark(tmp_path), blocks={"ants": 1, "tiles_ants": 2})
    assert choice["pipeline"] == "ants" and "simpler" in choice["reason"]


def test_tiles_decide_when_there_are_no_landmarks(tmp_path):
    from biahub.registration.benchmark import select_pipeline

    rows = [
        ("c", p, cand, "A/1/0", tiles, "")
        for p, tiles in (("ants", 0.9), ("tiles_ants", 0.4))
        for cand, tiles in (("unregistered", 3.0), ("transforms", tiles))
    ]
    choice = select_pipeline(_table(tmp_path, rows))
    assert (choice["pipeline"], choice["metric"]) == ("tiles_ants", "tiles_median_um")


def test_no_winner_when_nothing_clearly_beats_unregistered(tmp_path):
    from biahub.registration.benchmark import select_pipeline

    rows = [
        ("c", "ants", cand, "A/1/0", tiles, tiles)
        for cand, tiles in (("unregistered", 1.0), ("transforms", 0.9))
    ]
    choice = select_pipeline(_table(tmp_path, rows))
    assert choice["pipeline"] is None and "unregistered" in choice["reason"]


# --- choosing positions and writing the benchmark file ------------------------------------


def _survey():
    return [
        {"position": "A/1/0", "empty": False, "nan_fraction": 0.0, "spread": 0.9},
        {"position": "A/1/1", "empty": False, "nan_fraction": 0.0, "spread": 1.0},
        {"position": "A/2/0", "empty": False, "nan_fraction": 0.0, "spread": 0.8},
        {"position": "B/1/0", "empty": False, "nan_fraction": 0.0, "spread": 0.7},
        {"position": "B/1/1", "empty": True},
        {"position": "B/2/0", "empty": False, "nan_fraction": 0.6, "spread": 1.0},
    ]


def test_positions_are_chosen_from_the_survey_held_out_in_other_wells_first():
    from biahub.registration.benchmark import select_positions

    estimate, held_out = select_positions(_survey(), n_held_out=3)
    assert estimate == "A/1/1"  # the most spread out of the usable ones
    # other wells first (by spread), then the estimate's own well; never empty or mostly NaN
    assert held_out == ["A/2/0", "B/1/0", "A/1/0"]


def _store(tmp_path, name, channels):
    import numpy as np

    from iohub import open_ome_zarr

    path = tmp_path / name
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=channels) as plate:
        for well, fov in (("1", "0"), ("2", "0")):
            plate.create_position("A", well, fov)["0"] = np.zeros(
                (1, len(channels), 2, 4, 4), np.float32
            )
    return path


def test_the_benchmark_file_is_written_and_checked(tmp_path):
    import yaml

    from biahub.registration.benchmark import write_benchmark_config

    moving, reference = (
        _store(tmp_path, "m.zarr", ["DAPI"]),
        _store(tmp_path, "r.zarr", ["VS"]),
    )
    templates = (
        __import__("pathlib").Path(__file__).parents[1]
        / ".claude/skills/register-with-engine/templates/pipelines"
    )
    path = write_benchmark_config(
        tmp_path / "benchmark.yml",
        "case",
        moving,
        reference,
        "DAPI",
        "VS",
        "A/1/0",
        ["A/2/0"],
        {"local_pcc_ants": templates / "local_pcc_ants.yml", "ants": templates / "ants.yml"},
        landmarks="nuclei",
    )
    spec = yaml.safe_load(path.read_text())
    assert spec["cases"][0]["evaluate_positions"] == ["A/2/0"]
    assert [p["name"] for p in spec["pipelines"]] == ["local_pcc_ants", "ants"]

    with pytest.raises(ValueError, match="not in"):  # a position the store does not have
        write_benchmark_config(
            tmp_path / "b.yml",
            "case",
            moving,
            reference,
            "DAPI",
            "VS",
            "A/1/0",
            ["C/9/9"],
            {"ants": templates / "ants.yml"},
        )
    with pytest.raises(ValueError, match="held out"):  # estimated on is not held out
        write_benchmark_config(
            tmp_path / "b.yml",
            "case",
            moving,
            reference,
            "DAPI",
            "VS",
            "A/1/0",
            ["A/1/0"],
            {"ants": templates / "ants.yml"},
        )


def test_every_pipeline_template_loads():
    from pathlib import Path

    import yaml

    from biahub.settings import EstimateTransformSettings

    folder = (
        Path(__file__).parents[1] / ".claude/skills/register-with-engine/templates/pipelines"
    )
    names = sorted(p.stem for p in folder.glob("*.yml"))
    assert names == [
        "ants",
        "bead_matching",
        "local_pcc",
        "local_pcc_ants",
        "local_pcc_nuclei_matching",
    ]
    for path in folder.glob("*.yml"):
        EstimateTransformSettings(**yaml.safe_load(path.read_text()))


def test_points_decide_only_when_every_pipeline_matched_enough_nuclei(tmp_path):
    # A pipeline that matched 4 nuclei shows a small median from its few near pairs:
    # ranking then falls back to tiles, which every pipeline measured over the whole
    # field. Beating unregistered is judged on tiles too (its points match few nuclei).
    from biahub.registration.benchmark import select_pipeline

    rows = []
    for pipeline, tiles, points, n in (("good", 0.4, 0.6, 30), ("few", 2.0, 0.3, 4)):
        rows.append(("c", pipeline, "unregistered", "A/1/0", 3.0, 0.5, 3))
        rows.append(("c", pipeline, "transforms", "A/1/0", tiles, points, n))
    choice = select_pipeline(_table(tmp_path, rows))
    assert (choice["pipeline"], choice["metric"]) == ("good", "tiles_median_um")
