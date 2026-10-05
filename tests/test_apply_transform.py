import numpy as np
import pytest

from click.testing import CliRunner
from iohub import open_ome_zarr

from biahub.apply_transform import (
    apply_transform,
    apply_transform_cli,
    canvas,
    largest_box,
    overlap_slices,
)
from biahub.settings import TransformEntry, TransformSettings
from biahub.utils.config import model_to_yaml


def _translation(dz, dy, dx):
    m = np.eye(4)
    m[:3, 3] = [dz, dy, dx]
    return m.tolist()


def test_overlap_slices_of_a_translation_is_the_shifted_box():
    # inverse: reference voxel r reads moving voxel r + 3 along y, so the last 3 rows of the
    # reference grid read outside the moving volume.
    z, y, x = overlap_slices(
        (20, 40, 40), (20, 40, 40), np.array(_translation(0, 3, 0)), downsample=1
    )
    assert (z, y, x) == (slice(0, 20), slice(0, 37), slice(0, 40))


def test_largest_box_is_exact_on_a_sheared_mask():
    # A full box sheared along x by one voxel per z: the slice-at-middle heuristic keeps
    # the middle rectangle and clips z; the exact box trades a little x for all of z.
    z, y, x = 12, 20, 40
    mask = np.zeros((z, y, x), dtype=bool)
    for k in range(z):
        mask[k, :, k : k + 24] = True
    zs, ys, xs = largest_box(mask)
    assert mask[zs, ys, xs].all()
    assert (zs.stop - zs.start) * (ys.stop - ys.start) * (xs.stop - xs.start) == z * y * (
        24 - (z - 1)
    )
    with pytest.raises(ValueError):
        largest_box(np.zeros((2, 2, 2), dtype=bool))


def test_canvas_intersects_over_timepoints_and_keep_overhang_keeps_the_grid():
    shape = (20, 40, 40)
    matrices = [np.array(_translation(0, 3, 0)), np.array(_translation(0, -3, 5))]
    z, y, x = canvas(shape, shape, matrices, keep_overhang=False, downsample=1)
    assert (z, y, x) == (slice(0, 20), slice(3, 37), slice(0, 35))
    assert canvas(shape, shape, matrices, keep_overhang=True) == tuple(
        slice(0, s) for s in shape
    )
    with pytest.raises(Exception, match="no overlapping region"):
        canvas(
            shape, shape, [np.array(_translation(0, 45, 0))], keep_overhang=False, downsample=1
        )


@pytest.fixture
def structured_plate(tmp_path):
    """One position, 3 timepoints, 2 channels, with a bright block whose position we can track."""
    rng = np.random.default_rng(0)
    data = rng.random((3, 2, 16, 32, 32)).astype(np.float32) * 10
    data[:, :, 6:10, 12:20, 12:20] = 1000.0
    path = tmp_path / "in.zarr"
    with open_ome_zarr(
        path, layout="hcs", mode="w", channel_names=["GFP", "Phase3D"]
    ) as plate:
        plate.create_position("A", "1", "0")["0"] = data
    return path / "A" / "1" / "0", data


def _block_centre(volume):
    idx = np.argwhere(volume > 500)
    return idx.mean(axis=0)


def test_apply_transform_stabilizes_every_channel_with_per_timepoint_matrices(
    structured_plate, tmp_path
):
    position, data = structured_plate
    # Forward translation: move content by +dy per timepoint (inverse = -dy).
    forward = [_translation(0, 0, 0), _translation(0, 2, 0), _translation(0, 4, 0)]
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            transforms=[TransformEntry(t=t, matrix=m) for t, m in enumerate(forward)],
        ),
        config,
    )
    output = tmp_path / "out.zarr"

    apply_transform([position], config, output, cluster="debug")

    with open_ome_zarr(output / "A" / "1" / "0", mode="r") as out:
        assert out.channel_names == ["GFP", "Phase3D"], (
            "stabilization transforms every channel"
        )
        result = np.asarray(out.data)
    assert result.shape[0] == 3 and result.shape[1] == 2
    # Canvas: translations up to 4 along y remove 4 rows; z and x untouched.
    assert result.shape[2:] == (16, 28, 32)
    crop_y0 = 4 if _block_centre(result[0, 0])[1] < _block_centre(data[0, 0])[1] else 0
    for t, dy in enumerate((0, 2, 4)):
        moved = _block_centre(result[t, 0])[1] - (_block_centre(data[t, 0])[1] - crop_y0)
        assert moved == pytest.approx(dy, abs=0.6), (
            f"t={t}: block moved {moved:.2f} rows, expected {dy}"
        )


def test_apply_transform_registers_source_channels_onto_a_target_store(
    structured_plate, tmp_path
):
    position, data = structured_plate
    target = tmp_path / "target.zarr"
    with open_ome_zarr(
        target, layout="hcs", mode="w", channel_names=["Phase3D", "Retardance"]
    ) as plate:
        plate.create_position("A", "1", "0")["0"] = data
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            reference_channel="Phase3D",
            transforms=[TransformEntry(matrix=_translation(0, 0, 0))],  # series-wide
        ),
        config,
    )
    output = tmp_path / "out.zarr"

    apply_transform(
        [position],
        config,
        output,
        reference_position_dirpaths=[target / "A" / "1" / "0"],
        keep_overhang=True,
        cluster="debug",
        channels=["GFP"],
    )

    with open_ome_zarr(output / "A" / "1" / "0", mode="r") as out:
        assert out.channel_names == ["Phase3D", "Retardance", "GFP"], (
            "reference channels copied, moving channel transformed"
        )
        result = np.asarray(out.data)
    assert result.shape == data.shape[:1] + (3,) + data.shape[2:]
    np.testing.assert_allclose(result[:, 0], data[:, 0], atol=1e-3)  # copied target channel
    np.testing.assert_allclose(
        result[:, 2], data[:, 0], atol=1e-2
    )  # identity-transformed source channel


def test_apply_transform_writes_a_channel_both_copied_and_transformed_once(
    structured_plate, tmp_path
):
    # Moving store == reference store: GFP is a reference channel and the transformed one.
    # It must hold the transformed data, written by one job, not also a raw copy.
    position, data = structured_plate
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            reference_channel="Phase3D",
            transforms=[TransformEntry(matrix=_translation(0, 4, 0))],
        ),
        config,
    )
    output = tmp_path / "out.zarr"

    apply_transform(
        [position],
        config,
        output,
        reference_position_dirpaths=[position],
        keep_overhang=True,
        cluster="debug",
        channels=["GFP"],
    )

    with open_ome_zarr(output / "A" / "1" / "0", mode="r") as out:
        assert out.channel_names == ["GFP", "Phase3D"]
        result = np.asarray(out.data)
    np.testing.assert_allclose(result[:, 1], data[:, 1], atol=1e-3)  # copied
    shift = _block_centre(result[0, 0]) - _block_centre(data[0, 0])
    np.testing.assert_allclose(shift, [0, 4, 0], atol=0.5)  # transformed, not copied


def test_apply_transform_time_indices_subset_uses_each_timepoints_own_matrix(
    structured_plate, tmp_path
):
    position, data = structured_plate
    forward = [_translation(0, 0, 0), _translation(0, 2, 0), _translation(0, 4, 0)]
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            transforms=[TransformEntry(t=t, matrix=m) for t, m in enumerate(forward)],
        ),
        config,
    )
    output = tmp_path / "out.zarr"

    apply_transform(
        [position], config, output, time_indices=[0, 2], keep_overhang=True, cluster="debug"
    )

    with open_ome_zarr(output / "A" / "1" / "0", mode="r") as out:
        result = np.asarray(out.data)
    assert result.shape[0] == 2
    # Output t=1 is input t=2 moved by its own matrix (+4 rows), not by matrices[1].
    moved = _block_centre(result[1, 0])[1] - _block_centre(data[2, 0])[1]
    assert moved == pytest.approx(4, abs=0.6)


def test_apply_transform_cli_takes_moving_and_reference_position_paths(
    structured_plate, tmp_path
):
    position, data = structured_plate
    target = tmp_path / "target.zarr"
    with open_ome_zarr(target, layout="hcs", mode="w", channel_names=["Phase3D"]) as plate:
        plate.create_position("A", "1", "0")["0"] = data[:, :1]
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            reference_channel="Phase3D",
            transforms=[TransformEntry(matrix=_translation(0, 0, 0))],  # series-wide
        ),
        config,
    )
    output = tmp_path / "out.zarr"

    result = CliRunner().invoke(
        apply_transform_cli,
        [
            "-m",
            str(position),
            "-r",
            str(target / "A" / "1" / "0"),
            "-c",
            str(config),
            "-o",
            str(output),
            "--keep-overhang",
            "--time-indices",
            "0,2",
            "--cluster",
            "debug",
        ],
    )

    assert result.exit_code == 0, result.output
    with open_ome_zarr(output / "A" / "1" / "0", mode="r") as out:
        assert out.channel_names == ["Phase3D", "GFP"]
        assert out.data.shape[0] == 2  # --time-indices 0,2


def test_transform_settings_matrix_for_uses_own_entry_else_nearest_earlier():
    series = TransformSettings(
        direction="forward",
        moving_channels=["GFP"],
        transforms=[
            TransformEntry(t=0, matrix=_translation(0, 0, 0)),
            TransformEntry(t=5, matrix=_translation(0, 5, 0)),
        ],
    )
    assert series.matrix_for(5, "forward")[1, 3] == 5
    assert series.matrix_for(7, "forward")[1, 3] == 5  # nearest earlier entry
    assert series.matrix_for(5, "inverse")[1, 3] == -5  # inverted on request
    assert len(series.unique_matrices("inverse")) == 2
    with pytest.raises(ValueError, match="unique, increasing"):
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            transforms=[
                TransformEntry(t=5, matrix=_translation(0, 0, 0)),
                TransformEntry(t=0, matrix=_translation(0, 0, 0)),
            ],
        )
    with pytest.raises(ValueError, match="one entry without t"):
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            transforms=[
                TransformEntry(matrix=_translation(0, 0, 0)),
                TransformEntry(t=1, matrix=_translation(0, 0, 0)),
            ],
        )


def test_transform_settings_holds_a_shared_list_or_one_per_position_not_both():
    entry = [TransformEntry(matrix=_translation(0, 0, 0))]
    with pytest.raises(ValueError, match="exactly one of transforms"):
        TransformSettings(direction="forward", moving_channels=["GFP"])
    with pytest.raises(ValueError, match="exactly one of transforms"):
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            transforms=entry,
            positions={"A/1/0": entry},
        )
    with pytest.raises(ValueError, match="row/col/fov"):
        TransformSettings(
            direction="forward", moving_channels=["GFP"], positions={"A1": entry}
        )


def test_transform_settings_per_position_lists_each_with_its_own_time_layout():
    settings = TransformSettings(
        direction="forward",
        moving_channels=["GFP"],
        positions={
            # one FOV: a transform per timepoint; the other: one transform for every t
            "A/1/0": [
                TransformEntry(t=0, matrix=_translation(0, 1, 0)),
                TransformEntry(t=2, matrix=_translation(0, 3, 0)),
            ],
            "A/1/1": [TransformEntry(matrix=_translation(0, 0, 9), estimated_at=4)],
        },
    )
    assert settings.per_position and not settings.series_wide
    assert settings.matrix_for(1, "forward", "A/1/0")[1, 3] == 1  # nearest earlier
    assert settings.matrix_for(2, "forward", "A/1/0")[1, 3] == 3
    assert settings.matrix_for(7, "forward", "A/1/1")[2, 3] == 9  # whole series
    assert settings.timepoints("A/1/0") == [0, 2] and settings.timepoints("A/1/1") is None
    with pytest.raises(ValueError, match="no transforms for position 'B/2/0'"):
        settings.matrix_for(0, "forward", "B/2/0")


def test_transform_entry_estimated_at_is_only_for_a_whole_series_entry():
    TransformEntry(matrix=_translation(0, 0, 0), estimated_at=5)
    with pytest.raises(ValueError, match="estimated_at"):
        TransformEntry(t=5, matrix=_translation(0, 0, 0), estimated_at=5)


@pytest.fixture
def two_position_plate(tmp_path):
    """Two FOVs, 2 timepoints, a bright block at the same place in both."""
    rng = np.random.default_rng(1)
    data = rng.random((2, 1, 16, 32, 32)).astype(np.float32) * 10
    data[:, :, 6:10, 12:20, 12:20] = 1000.0
    path = tmp_path / "two.zarr"
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["GFP"]) as plate:
        for fov in ("0", "1"):
            plate.create_position("A", "1", fov)["0"] = data
    return [path / "A" / "1" / fov for fov in ("0", "1")], data


def test_apply_transform_applies_each_positions_own_list(two_position_plate, tmp_path):
    positions, data = two_position_plate
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            positions={
                "A/1/0": [TransformEntry(matrix=_translation(0, 3, 0))],
                "A/1/1": [TransformEntry(matrix=_translation(0, 0, -4))],
            },
        ),
        config,
    )
    output = tmp_path / "out.zarr"

    apply_transform(positions, config, output, keep_overhang=True, cluster="debug")

    for fov, expected in (("0", [0, 3, 0]), ("1", [0, 0, -4])):
        with open_ome_zarr(output / "A" / "1" / fov, mode="r") as out:
            moved = _block_centre(np.asarray(out.data)[1, 0]) - _block_centre(data[1, 0])
        np.testing.assert_allclose(moved, expected, atol=0.5)


def test_apply_transform_refuses_a_position_missing_from_a_per_position_file(
    two_position_plate, tmp_path
):
    positions, _ = two_position_plate
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            positions={"A/1/0": [TransformEntry(matrix=_translation(0, 0, 0))]},
        ),
        config,
    )
    with pytest.raises(Exception, match=r"no list for positions \['A/1/1'\]"):
        apply_transform(positions, config, tmp_path / "out.zarr", cluster="debug")


def test_reference_positions_are_paired_by_key_not_by_order():
    from pathlib import Path

    from biahub.cli.parsing import pair_reference_positions

    refs = [Path("ref.zarr/A/1/1"), Path("ref.zarr/A/1/0")]
    paired = pair_reference_positions(["A/1/0", "A/1/1"], refs)
    assert paired == {"A/1/0": refs[1], "A/1/1": refs[0]}
    assert pair_reference_positions(["A/1/0", "A/1/1"], refs[:1]) == {
        "A/1/0": refs[0],
        "A/1/1": refs[0],
    }
    with pytest.raises(Exception, match="no reference position"):
        pair_reference_positions(["B/2/0"], refs)


def test_transform_entry_status_defaults_to_accepted_and_a_stand_in_is_never_accepted():
    assert TransformEntry(matrix=_translation(0, 0, 0)).status == "accepted"
    stand_in = TransformEntry(
        t=3, matrix=_translation(0, 0, 0), status="unreliable", filled_from="t=2"
    )
    assert stand_in.filled_from == "t=2"
    with pytest.raises(ValueError, match="stand-in"):
        TransformEntry(t=3, matrix=_translation(0, 0, 0), filled_from="t=2")
    with pytest.raises(ValueError):
        TransformEntry(matrix=_translation(0, 0, 0), status="maybe")


def test_apply_transform_writes_every_timepoint_and_records_the_not_accepted_ones(
    structured_plate, tmp_path
):
    position, data = structured_plate
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            transforms=[
                TransformEntry(t=0, matrix=_translation(0, 0, 0)),
                TransformEntry(
                    t=1, matrix=_translation(0, 0, 0), status="unreliable", filled_from="t=0"
                ),
                TransformEntry(t=2, matrix=_translation(0, 0, 0)),
            ],
        ),
        config,
    )
    output = tmp_path / "out.zarr"

    apply_transform([position], config, output, keep_overhang=True, cluster="debug")

    with open_ome_zarr(output / "A" / "1" / "0", mode="r") as out:
        assert out.data.shape[0] == 3  # the unreliable timepoint is written, not dropped
        recorded = out.zattrs["biahub-apply-transform"]["timepoints_not_accepted"]
    assert recorded == {"unreliable": [1]}


def test_apply_transform_registers_every_moving_channel_by_default(structured_plate, tmp_path):
    # The file was estimated on GFP, but the transform holds for every channel of the
    # moving store (shared optics and stage), so both are registered unless told otherwise.
    position, data = structured_plate
    target = tmp_path / "target.zarr"
    with open_ome_zarr(target, layout="hcs", mode="w", channel_names=["Retardance"]) as plate:
        plate.create_position("A", "1", "0")["0"] = data[:, :1]
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            moving_channels=["GFP"],
            reference_channel="Retardance",
            transforms=[TransformEntry(matrix=_translation(0, 0, 0))],
        ),
        config,
    )

    apply_transform(
        [position],
        config,
        tmp_path / "all.zarr",
        reference_position_dirpaths=[target / "A" / "1" / "0"],
        keep_overhang=True,
        cluster="debug",
    )
    with open_ome_zarr(tmp_path / "all.zarr" / "A" / "1" / "0", mode="r") as out:
        assert out.channel_names == ["Retardance", "GFP", "Phase3D"]

    with pytest.raises(Exception, match="channels not in the moving store"):
        apply_transform(
            [position],
            config,
            tmp_path / "bad.zarr",
            reference_position_dirpaths=[target / "A" / "1" / "0"],
            cluster="debug",
            channels=["DAPI"],
        )


def test_transforms_written_with_the_old_pull_name_load_as_inverse():
    settings = TransformSettings(
        direction="pull",
        moving_channels=["GFP"],
        transforms=[TransformEntry(matrix=_translation(0, 0, 3))],
    )
    assert settings.direction == "inverse"
    np.testing.assert_allclose(settings.matrix_for(0, "forward")[2, 3], -3.0)
