import numpy as np
import pytest

from iohub.core.registry import set_default_implementation
from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate

from biahub.utils.config import settings_fingerprint
from biahub.virtual_stain import _write_timepoints

T, C, Z, Y, X = 3, 2, 4, 8, 8


# Production imports cytoland before writing, which switches iohub's default
# implementation to zarr-python (see tests/conftest.py); the rest of biahub
# writes through zarrs-python. Resume has to work under both.
@pytest.fixture(params=["zarrs-python", "zarr-python"])
def implementation(request):
    set_default_implementation(request.param)
    return request.param


def _output_position(tmp_path, shards_ratio=None):
    store = tmp_path / "output.zarr"
    create_empty_plate(
        store_path=store,
        position_keys=[("A", "1", "0")],
        channel_names=["Nuclei", "Membrane"],
        shape=(T, C, Z, Y, X),
        shards_ratio=shards_ratio,
        version="0.5",
    )
    return store / "A" / "1" / "0"


def _predictor(calls, offset=0.0, fail_at=None):
    def predict_timepoint(t):
        if t == fail_at:
            raise RuntimeError(f"interrupted at t={t}")
        calls.append(t)
        return np.full((C, Z, Y, X), t + 1 + offset, dtype=np.float32)

    return predict_timepoint


def _written(position):
    with open_ome_zarr(str(position), mode="r") as dataset:
        return np.asarray(dataset.data[:, 0, 0, 0, 0])


def test_write_timepoints_writes_every_timepoint(tmp_path, implementation):
    position = _output_position(tmp_path)
    calls = []
    _write_timepoints(_predictor(calls), position, T, resume=True, resume_token="a")
    assert calls == [0, 1, 2]
    np.testing.assert_array_equal(_written(position), [1, 2, 3])


def test_write_timepoints_resume_skips_finished(tmp_path, implementation):
    position = _output_position(tmp_path)
    _write_timepoints(_predictor([]), position, T, resume=True, resume_token="a")
    calls = []
    _write_timepoints(_predictor(calls), position, T, resume=True, resume_token="a")
    assert calls == []


def test_write_timepoints_resume_after_interruption(tmp_path, implementation):
    position = _output_position(tmp_path)
    with pytest.raises(RuntimeError):
        _write_timepoints(
            _predictor([], fail_at=1), position, T, resume=True, resume_token="a"
        )
    calls = []
    _write_timepoints(_predictor(calls), position, T, resume=True, resume_token="a")
    assert calls == [1, 2]
    np.testing.assert_array_equal(_written(position), [1, 2, 3])


@pytest.mark.parametrize("resume, token", [(True, "b"), (False, "a")])
def test_write_timepoints_recomputes(tmp_path, implementation, resume, token):
    """A changed config (token), or no --resume, recomputes every timepoint."""
    position = _output_position(tmp_path)
    _write_timepoints(_predictor([]), position, T, resume=True, resume_token="a")
    calls = []
    _write_timepoints(
        _predictor(calls, offset=10), position, T, resume=resume, resume_token=token
    )
    assert calls == [0, 1, 2]
    np.testing.assert_array_equal(_written(position), [11, 12, 13])


def test_write_timepoints_batches_by_time_shard(tmp_path, implementation):
    """With two timepoints per shard, an interruption inside a shard recomputes it whole."""
    position = _output_position(tmp_path, shards_ratio=(2, 1, 1, 1, 1))
    with pytest.raises(RuntimeError):
        _write_timepoints(
            _predictor([], fail_at=1), position, T, resume=True, resume_token="a"
        )
    calls = []
    _write_timepoints(_predictor(calls), position, T, resume=True, resume_token="a")
    assert calls == [0, 1, 2]
    calls = []
    _write_timepoints(_predictor(calls), position, T, resume=True, resume_token="a")
    assert calls == []


def test_settings_fingerprint_of_mapping_ignores_key_order():
    first = {"model": {"lr": 1, "depth": 2}, "ckpt_path": "a.ckpt"}
    second = {"ckpt_path": "a.ckpt", "model": {"depth": 2, "lr": 1}}
    assert settings_fingerprint(first) == settings_fingerprint(second)
    assert settings_fingerprint(first) != settings_fingerprint(
        {**first, "ckpt_path": "b.ckpt"}
    )
