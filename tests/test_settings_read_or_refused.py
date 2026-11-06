"""Every setting a config accepts is read: one the run would ignore is refused, saying why."""

import pytest

from pydantic import ValidationError

from biahub.settings import EstimateTransformSettings


def _settings(method="phase_cross_corr", **kwargs):
    kwargs.setdefault("reference", {"frame": "cross", "channel": "Phase3D"})
    return EstimateTransformSettings(moving={"channel": "GFP"}, method=method, **kwargs)


def test_another_methods_block_is_refused():
    with pytest.raises(ValidationError, match="ants block.*method phase_cross_corr"):
        _settings(ants={})
    # a stack's blocks carry their own settings: no top-level block is read
    with pytest.raises(ValidationError, match="belong in the blocks"):
        _settings(method=[{"phase_cross_corr": {}}, {"ants": {}}], phase_cross_corr={})
    _settings(phase_cross_corr={})  # its own block is fine


def test_the_neighbour_window_is_given_with_neighbours_and_only_there():
    # window defaulted to 6 (a hidden choice: how many frames on each side), and was
    # accepted by every frame though only neighbours reads it
    with pytest.raises(ValidationError, match="'neighbours' needs its window"):
        _settings(reference={"frame": "neighbours"})
    with pytest.raises(ValidationError, match="window is read only by frame neighbours"):
        _settings(reference={"frame": "first", "window": 4})
    assert _settings(reference={"frame": "neighbours", "window": 4}).reference.window == 4


def test_propagation_is_refused_for_a_method_that_ignores_where_it_starts():
    # each timepoint would start from the previous result, which these methods do not read
    propagate = {"initial": {"propagate": True}}
    with pytest.raises(ValidationError, match="focus_finding ignores where it starts"):
        _settings(method="focus_finding", reference={"frame": "first"}, transform=propagate)
    _settings(method="ants", transform=propagate)  # ANTs refines from its start
    _settings(method="phase_cross_corr", phase_cross_corr={"local": {}}, transform=propagate)


@pytest.mark.parametrize(
    "method, block",
    [
        ("bead_matching", {}),
        ("ants", {}),
        ("phase_cross_corr", {}),
        ("phase_cross_corr", {"local": {}}),
        ("focus_finding", {}),
        ("nuclei_matching", {}),
        ("manual", {}),
        ("shapes", {"rotate90": 0, "fliplr": False}),
    ],
)
def test_the_settings_know_which_methods_ignore_where_they_start(method, block):
    # the settings cannot build an estimator; their record must match the estimators'
    from biahub.registration.engine.core import build_estimator

    settings = _settings(
        method=method,
        time_indices=0 if method == "manual" else "all",
        transform={"type": "affine" if method == "shapes" else "euclidean"},
        **{method: block},
    )
    estimator, _score = build_estimator(
        settings, (8, 64, 64), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
    )
    assert settings.ignores_initial_transform == (not estimator.uses_initial_transform)


def test_a_transform_type_the_method_cannot_fit_is_refused():
    # the type labelled the file as something the method never fitted (focus finding
    # measures a translation; manual fits euclidean or similarity; shapes rescales)
    with pytest.raises(ValidationError, match="focus_finding fits euclidean only"):
        _settings(
            method="focus_finding", reference={"frame": "first"}, transform={"type": "affine"}
        )
    with pytest.raises(ValidationError, match="manual fits euclidean or similarity"):
        _settings(method="manual", time_indices=0, transform={"type": "affine"})
    with pytest.raises(ValidationError, match="shapes fits affine only"):
        _settings(method="shapes", shapes={"rotate90": 0, "fliplr": False})
    with pytest.raises(ValidationError, match="phase_cross_corr fits euclidean only"):
        _settings(transform={"type": "similarity"})
    _settings(
        phase_cross_corr={"local": {}}, transform={"type": "similarity"}
    )  # local fits it
    # each block of a stack with its own type
    with pytest.raises(ValidationError, match=r"block 1 \(focus_finding\)"):
        _settings(
            method=[{"focus_finding": {}}, {"ants": {}}],
            reference={"frame": "first"},
            transform={"type": "affine"},
        )
