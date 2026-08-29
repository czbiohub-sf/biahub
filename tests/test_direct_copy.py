import json
import os

import numpy as np
import pytest

from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate

from biahub.cli.direct_copy import (
    copy_incompatibilities,
    copy_position_files,
    place_files,
    plan_position_copy,
    read_array_layout,
    summarize_direct_copy,
)

# Geometry small enough to be fast, but with more than one file per (t, c) in
# every spatial axis so a bug in the ZYX grid mapping shows up.
SHAPE = (3, 2, 8, 16, 16)
CHUNKS = (1, 1, 4, 8, 8)
SHARDS_RATIO = (1, 1, 2, 2, 2)
POSITION = ("A", "1", "0")


def _write_plate(
    path,
    channel_names,
    shape=SHAPE,
    chunks=CHUNKS,
    shards_ratio=SHARDS_RATIO,
    version="0.5",
    dtype=np.uint16,
    seed=0,
):
    """An HCS plate with one position holding random data, via create_empty_plate."""
    shape = (shape[0], len(channel_names), *shape[2:])
    create_empty_plate(
        store_path=path,
        position_keys=[POSITION],
        channel_names=list(channel_names),
        shape=shape,
        chunks=chunks,
        shards_ratio=shards_ratio,
        version=version,
        dtype=dtype,
    )
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 1000, size=shape).astype(dtype)
    with open_ome_zarr(path / "/".join(POSITION), layout="fov", mode="r+") as position:
        position.data[:] = data
    return data


def _full_slices(shape=SHAPE):
    return [slice(0, extent) for extent in shape[-3:]]


def _plan(input_plate, output_plate, input_channels, output_channels, **kwargs):
    return plan_position_copy(
        input_position_path=input_plate / "/".join(POSITION),
        output_position_path=output_plate / "/".join(POSITION),
        input_channel_indices=input_channels,
        output_channel_indices=output_channels,
        input_time_indices=kwargs.pop("input_time_indices", list(range(SHAPE[0]))),
        output_time_indices=kwargs.pop("output_time_indices", list(range(SHAPE[0]))),
        zyx_slicing_params=kwargs.pop("zyx_slicing_params", _full_slices()),
        **kwargs,
    )


# -- Layout reading --------------------------------------------------------


def test_read_array_layout_v05_reports_the_shard_as_the_file(tmp_path):
    """For a sharded array the on-disk file is the shard, not the inner chunk."""
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    layout = read_array_layout(tmp_path / "in.zarr" / "/".join(POSITION) / "0")

    assert layout.zarr_format == 3
    expected_shard = tuple(c * r for c, r in zip(CHUNKS, SHARDS_RATIO, strict=True))
    assert layout.file_shape == expected_shard
    assert layout.grid_shape == (3, 1, 1, 1, 1)
    assert layout.file_path((2, 0, 0, 0, 0)).name == "0"
    assert layout.file_path((2, 0, 0, 0, 0)).exists()


def test_read_array_layout_v04(tmp_path):
    _write_plate(tmp_path / "in.zarr", ["GFP"], version="0.4", shards_ratio=None)
    layout = read_array_layout(tmp_path / "in.zarr" / "/".join(POSITION) / "0")

    assert layout.zarr_format == 2
    assert layout.file_shape == CHUNKS
    assert layout.key_prefix == ""


def test_read_array_layout_rejects_a_missing_array(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_array_layout(tmp_path / "nope")


# -- Compatibility --------------------------------------------------------


def test_matching_geometry_is_copyable(tmp_path):
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP", "RFP"])

    plan, reasons = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [1])

    assert reasons == []
    assert plan is not None
    # 3 timepoints x 1 channel x 1 spatial shard cell.
    assert plan.num_files == 3


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"dtype": np.float32}, "data_type"),
        ({"chunks": (1, 1, 2, 8, 8)}, "codecs"),
        ({"shards_ratio": (1, 1, 4, 2, 2)}, "chunk_grid"),
        ({"version": "0.4", "shards_ratio": None}, "Zarr format differs"),
    ],
)
def test_geometry_differences_block_the_copy(tmp_path, kwargs, expected):
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP", "RFP"], **kwargs)

    plan, reasons = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [1])

    assert plan is None
    assert any(expected in reason for reason in reasons), reasons


def test_a_crop_blocks_the_copy(tmp_path):
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP"])

    plan, reasons = _plan(
        tmp_path / "in.zarr",
        tmp_path / "out.zarr",
        [0],
        [0],
        zyx_slicing_params=[slice(0, 8), slice(2, 10), slice(0, 16)],
    )

    assert plan is None
    assert reasons == ["ROI crops Y[2:10]"]


def test_a_differing_zyx_shape_blocks_the_copy(tmp_path):
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP"], shape=(3, 1, 8, 12, 16))

    plan, reasons = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [0])

    assert plan is None
    assert any("ZYX shape differs" in reason for reason in reasons), reasons


def test_a_multi_timepoint_shard_blocks_a_time_subset(tmp_path):
    """A file holding several timepoints cannot be renamed into a subset."""
    ratio = (2, 1, 2, 2, 2)
    _write_plate(tmp_path / "in.zarr", ["GFP"], shards_ratio=ratio)
    _write_plate(tmp_path / "out.zarr", ["GFP"], shape=(2, 1, 8, 16, 16), shards_ratio=ratio)

    plan, reasons = _plan(
        tmp_path / "in.zarr",
        tmp_path / "out.zarr",
        [0],
        [0],
        input_time_indices=[0, 2],
        output_time_indices=[0, 1],
    )

    assert plan is None
    assert any("timepoints" in reason for reason in reasons), reasons


def test_a_multi_timepoint_shard_allows_the_identity(tmp_path):
    ratio = (2, 1, 2, 2, 2)
    _write_plate(tmp_path / "in.zarr", ["GFP"], shards_ratio=ratio)
    _write_plate(tmp_path / "out.zarr", ["GFP", "RFP"], shards_ratio=ratio)

    plan, reasons = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [1])

    assert reasons == []
    # 3 timepoints land in 2 shard cells along T.
    assert plan.num_files == 2


def test_an_unrecognised_codec_blocks_the_copy(tmp_path):
    """A field this module has never heard of must disqualify the copy."""
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP"])
    metadata_path = tmp_path / "out.zarr" / "/".join(POSITION) / "0" / "zarr.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["some_future_field"] = {"name": "quantize"}
    metadata_path.write_text(json.dumps(metadata))

    plan, reasons = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [0])

    assert plan is None
    assert any("some_future_field" in reason for reason in reasons), reasons


def test_attributes_do_not_block_the_copy(tmp_path):
    """Provenance written onto the output array must not disqualify it."""
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP"])
    metadata_path = tmp_path / "out.zarr" / "/".join(POSITION) / "0" / "zarr.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["attributes"] = {"biahub-concatenate": {"anything": True}}
    metadata_path.write_text(json.dumps(metadata))

    plan, reasons = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [0])

    assert reasons == []
    assert plan is not None


def test_a_channel_spanning_shard_blocks_the_copy(tmp_path):
    """Sharding across C means one file holds two channels, so keys cannot remap.

    ``create_empty_plate`` will not produce that geometry, so the metadata is
    edited directly; only the metadata is read, so the files need not agree.
    """
    _write_plate(tmp_path / "in.zarr", ["GFP", "RFP"])
    metadata_path = tmp_path / "in.zarr" / "/".join(POSITION) / "0" / "zarr.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["chunk_grid"]["configuration"]["chunk_shape"][1] = 2
    metadata_path.write_text(json.dumps(metadata))

    layout = read_array_layout(metadata_path.parent)
    reasons = copy_incompatibilities(
        layout, layout, _full_slices(), list(range(SHAPE[0])), list(range(SHAPE[0]))
    )

    assert any("several channels" in reason for reason in reasons), reasons


# -- Placement ------------------------------------------------------------


@pytest.mark.parametrize("mode", ["copy", "link"])
def test_copy_position_files_reproduces_the_read_write_result(tmp_path, mode):
    """The copied channel must read back exactly as the input channel."""
    input_data = _write_plate(tmp_path / "in.zarr", ["GFP", "RFP"])
    _write_plate(tmp_path / "out.zarr", ["DAPI", "GFP", "RFP"], seed=1)

    report = copy_position_files(
        input_position_path=tmp_path / "in.zarr" / "/".join(POSITION),
        output_position_path=tmp_path / "out.zarr" / "/".join(POSITION),
        input_channel_indices=[0, 1],
        output_channel_indices=[1, 2],
        input_time_indices=list(range(SHAPE[0])),
        output_time_indices=list(range(SHAPE[0])),
        zyx_slicing_params=_full_slices(),
        mode=mode,
        num_workers=4,
        extra_metadata={"biahub-concatenate": {"direct_copy": True}},
    )

    assert report.placed == 6  # 3 timepoints x 2 channels
    with open_ome_zarr(tmp_path / "out.zarr" / "/".join(POSITION), layout="fov") as out:
        np.testing.assert_array_equal(out.data[:, 1:3], input_data)
        assert out.zattrs["biahub-concatenate"] == {"direct_copy": True}


def test_link_mode_shares_inodes(tmp_path):
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP"])
    plan, _ = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [0])

    place_files(plan.file_pairs(), mode="link")

    for source, destination in plan.file_pairs():
        assert os.stat(source).st_ino == os.stat(destination).st_ino


def test_a_missing_source_file_clears_a_stale_destination(tmp_path):
    """An unwritten input block must leave the output at its fill value."""
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP"], seed=1)
    plan, _ = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [0])
    pairs = plan.file_pairs()
    pairs[0][0].unlink()

    report = place_files(pairs)

    assert (report.cleared, report.placed) == (1, len(pairs) - 1)
    with open_ome_zarr(tmp_path / "out.zarr" / "/".join(POSITION), layout="fov") as out:
        assert not out.data[0].any()


def test_resume_skips_files_already_placed(tmp_path):
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP"])
    plan, _ = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [0])
    pairs = plan.file_pairs()

    place_files(pairs)
    report = place_files(pairs, resume=True)

    assert (report.reused, report.placed) == (len(pairs), 0)


def test_placement_leaves_no_scratch_files(tmp_path):
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP"])
    plan, _ = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [0])

    place_files(plan.file_pairs(), num_workers=4)

    assert not list((tmp_path / "out.zarr").rglob("*.tmp"))


def test_copy_position_files_refuses_an_incompatible_pair(tmp_path):
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP"], dtype=np.float32)

    with pytest.raises(RuntimeError, match="file level"):
        copy_position_files(
            input_position_path=tmp_path / "in.zarr" / "/".join(POSITION),
            output_position_path=tmp_path / "out.zarr" / "/".join(POSITION),
            input_channel_indices=[0],
            output_channel_indices=[0],
            input_time_indices=list(range(SHAPE[0])),
            output_time_indices=list(range(SHAPE[0])),
            zyx_slicing_params=_full_slices(),
        )


# -- Summary --------------------------------------------------------------


def test_summarize_direct_copy_counts_reasons(tmp_path):
    _write_plate(tmp_path / "in.zarr", ["GFP"])
    _write_plate(tmp_path / "out.zarr", ["GFP"])
    plan, _ = _plan(tmp_path / "in.zarr", tmp_path / "out.zarr", [0], [0])

    summary = summarize_direct_copy(
        [plan, None, None], [[], ["dtype differs"], ["dtype differs"]]
    )

    assert (summary.total, summary.direct, summary.files) == (3, 1, plan.num_files)
    assert summary.all_direct is False
    assert summary.fallback_reasons == {"dtype differs": 2}
