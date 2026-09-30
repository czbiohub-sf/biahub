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
from biahub.settings import TransformSettings
from biahub.utils.config import model_to_yaml


def _translation(dz, dy, dx):
    m = np.eye(4)
    m[:3, 3] = [dz, dy, dx]
    return m.tolist()


def test_overlap_slices_of_a_translation_is_the_shifted_box():
    # pull: reference voxel r reads moving voxel r + 3 along y, so the last 3 rows of the
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
    # Forward translation: move content by +dy per timepoint (pull = -dy).
    forward = [_translation(0, 0, 0), _translation(0, 2, 0), _translation(0, 4, 0)]
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(direction="forward", matrices=forward, source_channels=["GFP"]),
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
            matrices=[_translation(0, 0, 0)],
            source_channels=["GFP"],
            target_channel="Phase3D",
            keep_overhang=True,
        ),
        config,
    )
    output = tmp_path / "out.zarr"

    apply_transform(
        [position],
        config,
        output,
        target_position_dirpaths=[target / "A" / "1" / "0"],
        cluster="debug",
    )

    with open_ome_zarr(output / "A" / "1" / "0", mode="r") as out:
        assert out.channel_names == ["Phase3D", "Retardance", "GFP"], (
            "target channels copied, source channel transformed"
        )
        result = np.asarray(out.data)
    assert result.shape == data.shape[:1] + (3,) + data.shape[2:]
    np.testing.assert_allclose(result[:, 0], data[:, 0], atol=1e-3)  # copied target channel
    np.testing.assert_allclose(
        result[:, 2], data[:, 0], atol=1e-2
    )  # identity-transformed source channel


def test_apply_transform_time_indices_subset_uses_each_timepoints_own_matrix(
    structured_plate, tmp_path
):
    position, data = structured_plate
    forward = [_translation(0, 0, 0), _translation(0, 2, 0), _translation(0, 4, 0)]
    config = tmp_path / "transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="forward",
            matrices=forward,
            source_channels=["GFP"],
            time_indices=[0, 2],
            keep_overhang=True,
        ),
        config,
    )
    output = tmp_path / "out.zarr"

    apply_transform([position], config, output, cluster="debug")

    with open_ome_zarr(output / "A" / "1" / "0", mode="r") as out:
        result = np.asarray(out.data)
    assert result.shape[0] == 2
    # Output t=1 is input t=2 moved by its own matrix (+4 rows), not by matrices[1].
    moved = _block_centre(result[1, 0])[1] - _block_centre(data[2, 0])[1]
    assert moved == pytest.approx(4, abs=0.6)


def test_apply_transform_cli_takes_source_and_target_position_paths(
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
            matrices=[_translation(0, 0, 0)],
            source_channels=["GFP"],
            target_channel="Phase3D",
            keep_overhang=True,
        ),
        config,
    )
    output = tmp_path / "out.zarr"

    result = CliRunner().invoke(
        apply_transform_cli,
        [
            "-s",
            str(position),
            "-t",
            str(target / "A" / "1" / "0"),
            "-c",
            str(config),
            "-o",
            str(output),
            "--cluster",
            "debug",
        ],
    )

    assert result.exit_code == 0, result.output
    with open_ome_zarr(output / "A" / "1" / "0", mode="r") as out:
        assert out.channel_names == ["Phase3D", "GFP"]
