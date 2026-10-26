"""The benchmark helpers the registration skill uses to choose and run a pipeline."""

import csv

import pytest


def _table(tmp_path, rows):
    path = tmp_path / "benchmark.csv"
    fields = [
        "case",
        "pipeline",
        "candidate",
        "position",
        "tiles_median_um",
        "points_median_um",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
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
    from biahub.registration.benchmark import pick_pipeline

    choice = pick_pipeline(_benchmark(tmp_path, ants_points=(1.0, 1.2)))
    assert choice["pipeline"] == "tiles_ants" and choice["metric"] == "points_median_um"
    assert choice["median_um"] == pytest.approx(0.55)
    assert "tiles_ants" in choice["reason"]


def test_a_tie_goes_to_the_simpler_pipeline(tmp_path):
    # equal within the tolerance: fewer blocks wins
    from biahub.registration.benchmark import pick_pipeline

    choice = pick_pipeline(_benchmark(tmp_path), blocks={"ants": 1, "tiles_ants": 2})
    assert choice["pipeline"] == "ants" and "simpler" in choice["reason"]


def test_tiles_decide_when_there_are_no_landmarks(tmp_path):
    from biahub.registration.benchmark import pick_pipeline

    rows = [
        ("c", p, cand, "A/1/0", tiles, "")
        for p, tiles in (("ants", 0.9), ("tiles_ants", 0.4))
        for cand, tiles in (("unregistered", 3.0), ("transforms", tiles))
    ]
    choice = pick_pipeline(_table(tmp_path, rows))
    assert (choice["pipeline"], choice["metric"]) == ("tiles_ants", "tiles_median_um")


def test_no_winner_when_nothing_clearly_beats_unregistered(tmp_path):
    from biahub.registration.benchmark import pick_pipeline

    rows = [
        ("c", "ants", cand, "A/1/0", tiles, tiles)
        for cand, tiles in (("unregistered", 1.0), ("transforms", 0.9))
    ]
    choice = pick_pipeline(_table(tmp_path, rows))
    assert choice["pipeline"] is None and "unregistered" in choice["reason"]
