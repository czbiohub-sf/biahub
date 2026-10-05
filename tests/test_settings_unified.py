import numpy as np
import pytest

from biahub.settings import (
    ChannelSettings,
    EstimateTransformSettings,
    ReferenceSettings,
    TransformEntry,
    TransformSettings,
    load_estimate_transform_settings,
    load_transform_settings,
)
from biahub.utils.config import model_to_yaml


def test_cross_frame_needs_a_channel_and_self_frames_forbid_one():
    with pytest.raises(ValueError, match="needs a reference channel"):
        ReferenceSettings(frame="cross")
    with pytest.raises(ValueError, match="drop reference.channel"):
        ReferenceSettings(frame="first", channel="GFP")
    stabilization = EstimateTransformSettings(
        moving=ChannelSettings(channel="GFP"),
        reference=ReferenceSettings(frame="previous"),
        method="phase-cross-corr",
    )
    assert stabilization.reference_channel == "GFP"
    assert stabilization.phase_cross_corr is not None, "the chosen method's block is filled in"
    assert stabilization.beads is None
    assert stabilization.effective_score_metric == "correlation"


def test_loader_reads_unified_configs_and_points_old_ones_at_convert_settings(tmp_path):
    unified = EstimateTransformSettings(
        moving=ChannelSettings(channel="GFP"),
        reference=ReferenceSettings(frame="cross", channel="Phase3D"),
        method="beads",
        score_metric="residual",
    )
    model_to_yaml(unified, tmp_path / "unified.yml")
    assert load_estimate_transform_settings(tmp_path / "unified.yml") == unified

    (tmp_path / "legacy.yml").write_text(
        "source_channel_name: GFP\ntarget_channel_name: Phase3D\nestimation_method: beads\n"
    )
    with pytest.raises(ValueError, match="convert-settings"):
        load_estimate_transform_settings(tmp_path / "legacy.yml")
    (tmp_path / "first_unified.yml").write_text(
        "direction: forward\nmatrices: []\nsource_channels: [GFP]\n"
    )
    with pytest.raises(ValueError, match="convert-settings"):
        load_transform_settings(tmp_path / "first_unified.yml")


def test_transform_settings_series_wide_entry_and_direction():
    forward = np.eye(4)
    forward[:3, 3] = [-2.0, 3.0, -4.0]
    series = TransformSettings(
        direction="forward",
        moving_channels=["GFP"],
        transforms=[TransformEntry(matrix=forward.tolist())],
    )
    assert series.series_wide and series.timepoints() is None
    np.testing.assert_allclose(series.matrix_for(17, "forward"), forward)
    np.testing.assert_allclose(series.matrix_for(17, "pull")[:3, 3], [2, -3, 4])
    with pytest.raises(ValueError, match="at least one entry"):
        TransformSettings(direction="forward", moving_channels=["GFP"], transforms=[])


def _beads_settings(sweep):
    return EstimateTransformSettings(
        moving=ChannelSettings(channel="GFP"),
        reference=ReferenceSettings(frame="cross", channel="Phase3D"),
        method="beads",
        fallback={"sweep": sweep},
    )


def test_sweep_trials_are_the_union_of_each_sub_grids_product():
    settings = _beads_settings(
        {
            "grid": [
                {
                    "beads.hungarian_match_settings.cost_threshold": [0.05, 0.1],
                    "beads.hungarian_match_settings.cost_matrix_settings.weights.dist": [
                        0.25,
                        1.0,
                    ],
                },
                {
                    "beads.algorithm": ["spectral"],
                    "beads.spectral_match_settings.sigma": [1, 5],
                },
            ]
        }
    )
    trials = settings.sweep_trials()

    assert len(trials) == 4 + 2
    trial = trials[
        "beads.hungarian_match_settings.cost_threshold=0.05,"
        "beads.hungarian_match_settings.cost_matrix_settings.weights.dist=1.0"
    ]
    assert trial.beads.hungarian_match_settings.cost_threshold == 0.05
    assert trial.beads.hungarian_match_settings.cost_matrix_settings.weights["dist"] == 1.0
    assert trial.fallback.sweep is None, "a trial does not sweep again"
    spectral = trials["beads.algorithm=spectral,beads.spectral_match_settings.sigma=5"]
    assert spectral.beads.algorithm == "spectral"
    assert spectral.beads.spectral_match_settings.sigma == 5
    assert settings.beads.algorithm == "hungarian", "the base settings are untouched"


def test_sweep_trials_revalidate_so_dependent_defaults_follow():
    trials = _beads_settings(
        {
            "grid": [
                {
                    "beads.hungarian_match_settings.edge_graph_settings.method": [
                        "full",
                        "radius",
                    ]
                }
            ]
        }
    ).sweep_trials()
    graphs = [t.beads.hungarian_match_settings.edge_graph_settings for t in trials.values()]
    assert [(g.method, g.k, g.radius) for g in graphs] == [
        ("full", None, None),
        ("radius", None, 30.0),
    ]


@pytest.mark.parametrize(
    "grid, message",
    [
        (
            [{"beads.hungarian_match_settings.cost_treshold": [0.1]}],
            "no setting 'cost_treshold'",
        ),
        (
            [{"ants.sobel_filter": [True]}],
            "would be ignored",
        ),  # beads method: ants block unused
        ([{"reference.frame": ["first"]}], "would be ignored"),
        ([{"score_metric": ["correlation"]}], "would be ignored"),
        ([{"transform.seed_from": ["previous_timepoint"]}], "would be ignored"),
        ([{"beads.algorithm": []}], "each with values"),
        ([], "at least 1 item"),
    ],
)
def test_sweep_grid_is_checked_when_the_config_loads(grid, message):
    with pytest.raises(ValueError, match=message):
        _beads_settings({"grid": grid})


def test_sweep_accepts_the_method_block_and_the_transform_type():
    trials = _beads_settings(
        {
            "grid": [
                {
                    "beads.hungarian_match_settings.cost_threshold": [0.05],
                    "transform.type": ["affine"],
                }
            ]
        }
    ).sweep_trials()
    (trial,) = trials.values()
    assert trial.transform.type == "affine"
