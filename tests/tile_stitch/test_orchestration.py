"""CPU contracts for typed tile-stitch run preparation."""

from pathlib import Path

import numpy as np
import pytest

from iohub.ngff import open_ome_zarr

from biahub.settings import TileStitchReconSettings
from biahub.tile_stitch.orchestration import (
    PreparationError,
    _chunk_shape,
    create_stitch_output,
    prepare_stitch_run,
)


def _config(
    time_indices: str | list[int] = "all", recon_dtype: str | None = None
) -> TileStitchReconSettings:
    settings: dict = {
        "tile_stitch": {
            "tile": {
                "tile_size": {"z": 4, "y": 16, "x": 16},
                "overlap": {"z": 0, "y": 4, "x": 4},
            },
            "blend": {"kind": "uniform_mean"},
            "recon": {
                "input_channel_names": ["phase"],
                "reconstruction_dimension": 3,
                "phase": {},
            },
        },
        "time_indices": time_indices,
    }
    if recon_dtype is not None:
        settings["monarch"] = {"recon_dtype": recon_dtype}
    return TileStitchReconSettings.model_validate(settings)


def _position(path: Path, shape: tuple[int, ...]) -> None:
    data = np.zeros(shape, dtype=np.float32)
    with open_ome_zarr(path, layout="fov", mode="w", channel_names=["phase"]) as position:
        position.create_image("0", data)


def test_prepare_run_separates_static_program_and_work_units(tmp_path: Path):
    input_path = tmp_path / "input.zarr"
    output_path = tmp_path / "result"
    _position(input_path, (2, 1, 4, 24, 24))

    prepared = prepare_stitch_run([input_path], _config(), output_path)

    assert prepared.final_output == output_path.with_suffix(".zarr")
    assert not prepared.final_output.exists()
    assert len(prepared.work_units) == 2
    assert {item.work.timepoint for item in prepared.work_units} == {0, 1}
    assert all(item.work.input_path == str(input_path) for item in prepared.work_units)
    assert all(
        item.work.output_path == str(prepared.final_output) for item in prepared.work_units
    )
    assert prepared.program.monarch == prepared.config.monarch
    assert prepared.program.input_tiles
    assert prepared.program.output_tiles

    create_stitch_output(prepared)
    with open_ome_zarr(prepared.final_output, layout="fov", mode="r") as output:
        assert tuple(output.data.shape) == prepared.full_shape
        assert tuple(output.data.chunks) == prepared.chunk_shape


def test_chunk_shape_caps_production_tiles_near_64_mb():
    itemsize = np.dtype(np.float32).itemsize
    chunk_shape = _chunk_shape((2372, 110, 600), itemsize)

    assert chunk_shape == (1, 1, 242, 110, 600)
    assert int(np.prod(chunk_shape)) * itemsize <= 64_000_000
    assert _chunk_shape((4, 16, 16), itemsize) == (1, 1, 4, 16, 16)
    assert _chunk_shape((10, 10_000, 10_000), itemsize) == (1, 1, 1, 10_000, 10_000)


@pytest.mark.parametrize(
    ("recon_dtype", "expected"),
    [(None, np.float32), ("float32", np.float32), ("float16", np.float16)],
)
def test_output_store_honours_recon_dtype(tmp_path: Path, recon_dtype, expected):
    """``monarch.recon_dtype`` is documented as the STORED dtype, so the store follows it."""
    input_path = tmp_path / "input.zarr"
    _position(input_path, (1, 1, 4, 24, 24))

    prepared = prepare_stitch_run(
        [input_path], _config(recon_dtype=recon_dtype), tmp_path / "result"
    )
    create_stitch_output(prepared)

    with open_ome_zarr(prepared.final_output, layout="fov", mode="r") as output:
        assert output.data.dtype == expected
        assert tuple(output.data.chunks) == prepared.chunk_shape


def test_chunk_shape_scales_leading_extent_with_itemsize():
    """A narrower dtype fits more z per chunk rather than producing half-size chunks."""
    f32 = _chunk_shape((2372, 110, 600), 4)
    f16 = _chunk_shape((2372, 110, 600), 2)

    assert f16[2] == 2 * f32[2]
    assert int(np.prod(f16)) * 2 <= 64_000_000


def test_prepare_run_rejects_empty_timepoint_selection(tmp_path: Path):
    input_path = tmp_path / "input.zarr"
    _position(input_path, (1, 1, 4, 16, 16))

    with pytest.raises(PreparationError, match="at least one timepoint"):
        prepare_stitch_run([input_path], _config(time_indices=[]), tmp_path / "out.zarr")


def test_prepare_run_rejects_mismatched_shapes(tmp_path: Path):
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    _position(first, (1, 1, 4, 16, 16))
    _position(second, (1, 1, 4, 20, 16))

    with pytest.raises(PreparationError, match="share TCZYX shape"):
        prepare_stitch_run([first, second], _config(), tmp_path / "out.zarr")
