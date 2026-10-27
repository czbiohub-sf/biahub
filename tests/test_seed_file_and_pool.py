"""A seed from a transforms file (an earlier result) and pooling positions into one transform.

Together they give the plate-then-position flow: estimate on a few positions with
`pool:` (one shared transform), then on every position with `transform.seed_file` set to
that result, each refining from it and never ending worse than it.
"""

import numpy as np
import pytest

from iohub import open_ome_zarr
from pydantic import ValidationError
from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform
from biahub.estimate_transform import estimate_transform
from biahub.settings import (
    DetectPeaksSettings,
    TransformEntry,
    TransformSettings,
    load_transform_settings,
)
from biahub.utils.config import model_to_yaml
from tests.test_estimate_transform import SHAPE, _synthetic_bead_volume, _write_config

# the same yx everywhere, z differs by position (as on DragonFly)
SHIFTS = {"0": (2.0, -3.0, 4.0), "1": (3.0, -3.0, 4.0), "2": (6.0, -3.0, 4.0)}


@pytest.fixture
def three_positions(tmp_path):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    path = tmp_path / "three.zarr"
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]) as p:
        for fov, shift in SHIFTS.items():
            mov = ndi_shift(ref, shift=shift, order=1, mode="constant", cval=0.0)
            p.create_position("A", "1", fov)["0"] = np.stack([ref, mov])[None].astype(
                np.float32
            )
    return [path / "A" / "1" / fov for fov in SHIFTS]


def _config(tmp_path, name="config.yml", **changes):
    path = _write_config(tmp_path)
    from biahub.settings import EstimateTransformSettings
    from biahub.utils.config import yaml_to_model

    settings = yaml_to_model(path, EstimateTransformSettings)
    data = settings.model_dump()
    data["time_indices"] = 0
    for key, value in changes.items():
        section, _, field = key.partition("__")
        if field:
            data[section][field] = value
        else:
            data[section] = value
    out = tmp_path / name
    model_to_yaml(EstimateTransformSettings(**data), out)
    return out


def test_pool_writes_one_shared_transform_and_keeps_each_positions(three_positions, tmp_path):
    output = tmp_path / "plate" / "transforms.yml"
    config = _config(tmp_path, pool={"z": "median"})
    estimate_transform(
        three_positions, config, output, reference_position_dirpaths=three_positions,
        cluster="debug",
    )  # fmt: skip
    pooled = load_transform_settings(output)
    assert not pooled.per_position and len(pooled.transforms) == 1
    entry = pooled.transforms[0]
    assert entry.status == "accepted"
    assert "medoid A/1/1 of 3 positions" in entry.note
    # inverse z at the median of 2, 3, 6; yx as every position measured
    inverse = np.asarray(pooled.matrix_for(0, "inverse"))[:3, 3]
    np.testing.assert_allclose(inverse, (3.0, -3.0, 4.0), atol=0.5)
    # each position's own estimate stays next to it, for evaluation
    per_position = load_transform_settings(output.with_suffix("") / "per_position.yml")
    assert sorted(per_position.positions) == ["A/1/0", "A/1/1", "A/1/2"]


def test_pool_needs_two_positions(three_positions, tmp_path):
    config = _config(tmp_path, pool={"z": "medoid"})
    with pytest.raises(Exception, match="pool needs estimates on two or more positions"):
        estimate_transform(
            three_positions[:1], config, tmp_path / "o.yml",
            reference_position_dirpaths=three_positions[:1], cluster="debug",
        )  # fmt: skip


def _plate_file(tmp_path, inverse_shift):
    path = tmp_path / "plate_transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="inverse",
            moving_channels=["GFP"],
            reference_channel="Phase3D",
            transforms=[
                TransformEntry(matrix=Transform.from_translation(inverse_shift).to_list())
            ],
        ),
        path,
    )
    return path


def test_each_position_refines_from_the_seed_file(three_positions, tmp_path):
    seed_file = _plate_file(tmp_path, (3.0, -3.0, 4.0))
    # the residual score, continuous: the overlap score (6-voxel radius) ties a 1-slice
    # refinement with the seed, and a tie keeps the seed
    config = _config(
        tmp_path, transform__seed_file=str(seed_file), score={"metric": "residual"}
    )
    output = tmp_path / "fovs" / "transforms.yml"
    estimate_transform(
        three_positions, config, output, reference_position_dirpaths=three_positions,
        cluster="debug",
    )  # fmt: skip
    model = load_transform_settings(output)
    for fov, shift in SHIFTS.items():
        inverse = model.matrix_for(0, "inverse", f"A/1/{fov}")[:3, 3]
        np.testing.assert_allclose(inverse, shift, atol=0.5)
        entry = model.entry_for(0, f"A/1/{fov}")
        assert entry.seeded_from in ("seed_file", "seed_file+refined")
        assert entry.status == "accepted"


def test_a_failed_estimate_keeps_the_seed_files_transform(three_positions, tmp_path):
    # no beads detected (threshold above every peak): the estimate fails, the seed file's
    # transform is kept as it is, scored, and says so
    seed_file = _plate_file(tmp_path, SHIFTS["0"])
    peaks = DetectPeaksSettings(
        threshold_abs=1e9, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    config = _config(
        tmp_path,
        transform__seed_file=str(seed_file),
        bead_matching={"source_peaks_settings": peaks, "target_peaks_settings": peaks},
    )
    output = tmp_path / "fov0" / "transforms.yml"
    estimate_transform(
        three_positions[:1], config, output,
        reference_position_dirpaths=three_positions[:1], cluster="debug",
    )  # fmt: skip
    entry = load_transform_settings(output).transforms[0]
    assert entry.seeded_from == "seed_file"
    np.testing.assert_allclose(np.asarray(entry.matrix)[:3, 3], -np.asarray(SHIFTS["0"]))


@pytest.mark.parametrize(
    "transform, message",
    [
        ({"seed_file": "p.yml", "seed": np.diag([2.0, 1, 1, 1]).tolist()}, "seed_file or"),
        ({"seed_file": "p.yml", "seed_from_shapes": True}, "seed_file or"),
        ({"seed_file": "p.yml", "seed_from": "previous_timepoint"}, "previous_timepoint"),
    ],
)
def test_a_seed_file_refuses_another_seed(transform, message):
    from biahub.settings import TransformFitSettings

    with pytest.raises(ValidationError, match=message):
        TransformFitSettings(**transform)


def test_a_changed_seed_file_is_a_different_run(three_positions, tmp_path):
    import click

    seed_file = _plate_file(tmp_path, SHIFTS["0"])
    config = _config(tmp_path, transform__seed_file=str(seed_file))
    output = tmp_path / "fov0" / "transforms.yml"
    run = dict(reference_position_dirpaths=three_positions[:1], cluster="debug")
    estimate_transform(three_positions[:1], config, output, **run)
    _plate_file(tmp_path, SHIFTS["2"])  # same name, other content
    with pytest.raises(click.UsageError, match="changed"):
        estimate_transform(three_positions[:1], config, output, resume=True, **run)


def test_pool_refuses_a_chained_stabilization():
    from biahub.settings import EstimateTransformSettings

    with pytest.raises(ValidationError, match="pool needs one frame"):
        EstimateTransformSettings(
            moving={"channel": "GFP"},
            reference={"frame": "previous"},
            method="phase_cross_corr",
            pool={"z": "medoid"},
        )
