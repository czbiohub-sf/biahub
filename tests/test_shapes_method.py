"""The shapes method: a registration from the two stores' metadata (shapes and voxel
sizes, with quarter turns and a flip), on its own, as a stack's first block, or as the
run's initial transform (`transform.initial: {from: shapes}`)."""

import numpy as np
import pytest

from pydantic import ValidationError

from biahub.core.transform import Transform
from biahub.settings import EstimateTransformSettings

GRIDS = dict(
    mov_shape=(30, 100, 100),
    ref_shape=(20, 120, 120),
    mov_voxel_size=(0.5, 0.2, 0.2),
    ref_voxel_size=(0.6, 0.15, 0.15),
)


def _settings(method, **kwargs):
    return EstimateTransformSettings(
        moving={"channel": "GFP"},
        reference={"frame": "cross", "channel": "Phase3D"},
        method=method,
        **kwargs,
    )


def test_the_shapes_method_reads_only_the_shapes_and_the_voxel_sizes():
    from biahub.registration.methods.manual.shapes import ShapesEstimator

    estimator = ShapesEstimator(
        GRIDS["mov_voxel_size"], GRIDS["ref_voxel_size"], rotate90=-1, fliplr=True
    )
    mov, ref = np.zeros(GRIDS["mov_shape"]), np.ones(GRIDS["ref_shape"])
    mov_s, ref_s = GRIDS["mov_shape"], GRIDS["ref_shape"]
    forward = (
        Transform.from_scaling((0.5 / 0.6, 0.2 / 0.15, 0.2 / 0.15), ref_s, ref_s).matrix
        @ Transform.from_rotation_yx(-90, mov_s, ref_s).matrix
        @ Transform.from_flip_x(mov_s).matrix
    )  # flip first, then rotate, then rescale (moving -> reference)
    transform = estimator.estimate(mov, ref)
    np.testing.assert_allclose(transform.matrix, forward, atol=1e-12)
    # the voxels, not the start, decide it: the initial transform is not used
    assert estimator.uses_initial_transform is False
    start = Transform.from_translation([5.0, 0.0, 0.0])
    from_start = estimator.estimate(mov, ref, initial_transform=start)
    assert from_start.matrix.tolist() == transform.matrix.tolist()


def test_the_initial_transform_from_shapes_is_the_methods_answer():
    from biahub.registration.engine.policies import ShapesInitialTransform
    from biahub.registration.methods.manual.shapes import ShapesEstimator

    policy = ShapesInitialTransform(**GRIDS, rotate90=-1, fliplr=False)
    method = ShapesEstimator(
        GRIDS["mov_voxel_size"], GRIDS["ref_voxel_size"], rotate90=-1, fliplr=False
    )
    answer = method.estimate(np.zeros(GRIDS["mov_shape"]), np.zeros(GRIDS["ref_shape"]))
    assert policy.initial_for(3).matrix.tolist() == answer.matrix.tolist()


def test_method_shapes_needs_its_block_and_builds_with_the_stores_voxel_sizes():
    from biahub.registration.engine.core import build_estimator
    from biahub.registration.methods.manual.shapes import ShapesEstimator

    with pytest.raises(ValidationError, match="shapes: {rotate90"):
        _settings("shapes")
    with pytest.raises(ValidationError, match="fliplr"):
        _settings("shapes", shapes={"rotate90": -1})
    settings = _settings("shapes", shapes={"rotate90": -1, "fliplr": False})
    estimator, _score, _initial = build_estimator(
        settings, GRIDS["mov_shape"], GRIDS["mov_voxel_size"], GRIDS["ref_voxel_size"]
    )
    assert isinstance(estimator, ShapesEstimator)
    assert (estimator.mov_voxel_size, estimator.ref_voxel_size) == (
        GRIDS["mov_voxel_size"],
        GRIDS["ref_voxel_size"],
    )


def test_shapes_starts_a_stack_and_the_next_block_starts_from_it():
    from biahub.registration.engine.core import build_estimator
    from biahub.registration.estimators import ChainedEstimator
    from biahub.registration.methods.manual.shapes import ShapesEstimator

    settings = _settings(
        [{"shapes": {"rotate90": 0, "fliplr": False}}, {"phase_cross_corr": {}}]
    )
    chain, _score, _initial = build_estimator(
        settings, GRIDS["mov_shape"], GRIDS["mov_voxel_size"], GRIDS["ref_voxel_size"]
    )
    assert isinstance(chain, ChainedEstimator)
    assert isinstance(chain.stages[0], ShapesEstimator)
    with pytest.raises(ValueError, match="block 1 \\(shapes\\)"):
        _settings([{"shapes": {}}, {"phase_cross_corr": {}}]).stack_settings()


def test_a_run_with_method_shapes_writes_the_transform_from_the_metadata(tmp_path):
    # both stores on one grid, no turn or flip: identity at every timepoint
    from scipy.ndimage import shift as ndi_shift

    from biahub.settings import load_transform_settings
    from biahub.utils.config import model_to_yaml
    from tests.test_estimate_transform import (
        APPLIED_SHIFT_ZYX,
        SHAPE,
        _run,
        _synthetic_bead_volume,
        _write_plate,
    )

    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    plate = _write_plate(tmp_path / "plate.zarr", [(ref, mov), (ref, mov)])
    config = tmp_path / "shapes.yml"
    settings = _settings(
        "shapes", shapes={"rotate90": 0, "fliplr": False}, transform={"type": "affine"}
    )
    model_to_yaml(settings, config)
    output = tmp_path / "out" / "transforms.yml"
    _run(plate, config, output)

    entries = load_transform_settings(output).transforms
    assert len(entries) == 2
    for entry in entries:
        np.testing.assert_allclose(entry.matrix, np.eye(4), atol=1e-12)


def test_shapes_reads_no_pixels_so_preprocess_blocks_are_refused():
    shapes = {"rotate90": 0, "fliplr": False}
    with pytest.raises(ValidationError, match="shapes reads no pixels"):
        _settings("shapes", shapes=shapes, preprocess=[{"sobel": {}}])
    with pytest.raises(ValidationError, match="shapes reads no pixels"):
        _settings(
            [{"shapes": {**shapes, "preprocess": [{"sobel": {}}]}}, {"phase_cross_corr": {}}]
        )


def test_the_flip_happens_on_the_moving_grid_when_the_widths_differ():
    # The flip comes first, on moving coordinates: it pivots on the moving grid's centre.
    # Pivoting on the reference grid's sent moving x=0 to x=250 of a 200-wide grid.
    from biahub.registration.methods.manual.shapes import ShapesEstimator

    mov_shape, ref_shape = (5, 100, 100), (5, 100, 200)
    flipped = ShapesEstimator((1, 1, 1), (1, 1, 1), rotate90=0, fliplr=True).transform_for(
        mov_shape, ref_shape
    )
    points = np.array([[2.0, 50.0, 50.0], [2.0, 50.0, 0.0], [2.0, 50.0, 100.0]])
    np.testing.assert_allclose(
        flipped.apply_points(points),
        [[2.0, 50.0, 100.0], [2.0, 50.0, 150.0], [2.0, 50.0, 50.0]],
    )


def test_a_run_from_shapes_records_the_methods_forward_answer_once(tmp_path):
    # init_run writes the initial transform into the run's settings for every job: the
    # forward answer as it is (it once wrote the inverse, which each job inverted back)
    from scipy.ndimage import shift as ndi_shift

    from biahub.registration.engine.core import _static_initial
    from biahub.registration.engine.disk import ENGINE_SETTINGS_FILENAME
    from biahub.registration.methods.manual.shapes import ShapesEstimator
    from biahub.utils.config import model_to_yaml, yaml_to_model
    from tests.test_estimate_transform import (
        APPLIED_SHIFT_ZYX,
        SHAPE,
        _run,
        _synthetic_bead_volume,
        _write_plate,
    )

    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    plate = _write_plate(tmp_path / "plate.zarr", [(ref, mov)])
    config = tmp_path / "shapes.yml"
    turn = {"rotate90": 1, "fliplr": False}
    settings = _settings(
        "shapes",
        shapes=turn,
        transform={"type": "affine", "initial": {"from": "shapes", **turn}},
    )
    model_to_yaml(settings, config)
    output = tmp_path / "out" / "transforms.yml"
    _run(plate, config, output)

    resolved = yaml_to_model(
        output.with_suffix("") / ENGINE_SETTINGS_FILENAME, EstimateTransformSettings
    )
    initial = resolved.transform.initial
    expected = ShapesEstimator((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), **turn).transform_for(
        SHAPE, SHAPE
    )
    assert (initial.from_, initial.direction) == ("matrix", "forward")
    np.testing.assert_array_equal(initial.matrix, expected.matrix)
    np.testing.assert_array_equal(_static_initial(resolved).matrix, expected.matrix)
