"""Shared fixtures."""

from __future__ import annotations

import pytest

from waveorder.api.tile_stitch import clear_transfer_function_cache


@pytest.fixture(autouse=True)
def _isolate_tf_cache():
    clear_transfer_function_cache()
    yield
    clear_transfer_function_cache()
