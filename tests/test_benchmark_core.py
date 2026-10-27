"""The benchmark's core: many candidates on many positions, one table."""

import numpy as np

from iohub import open_ome_zarr
from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform

SHIFT = (0.0, 3.0, -2.0)


def _plate(tmp_path):
    """Two positions; DAPI is VS shifted by SHIFT voxels."""
    from tests.test_registration_estimators import _synthetic_blob_volume

    path = tmp_path / "plate.zarr"
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["DAPI", "VS"]) as plate:
        for fov in ("0", "1"):
            ref = _synthetic_blob_volume(
                np.random.default_rng(int(fov)), (24, 192, 192), n_blobs=200, sigma=3.0
            )
            ref = ref + 100.0
            mov = ndi_shift(ref, SHIFT, order=1, mode="nearest")
            position = plate.create_position("A", "1", fov)
            position.create_image("0", np.stack([mov, ref])[None].astype(np.float32))
    return [path / "A" / "1" / fov for fov in ("0", "1")]


def test_candidates_on_positions_make_one_table_with_the_unregistered_pair(tmp_path):
    from biahub.registration.benchmark import evaluate_candidates

    positions = _plate(tmp_path)
    exact = Transform.from_translation([-s for s in SHIFT])
    half = Transform.from_translation([-s / 2 for s in SHIFT])
    rows = evaluate_candidates(
        {"exact": exact, "half way": half},
        [(p, p) for p in positions],
        "DAPI",
        "VS",
        tile_px=64,
        crop_px=96,
    )
    assert len(rows) == 6  # (unregistered + 2 candidates) x 2 positions
    for fov in ("A/1/0", "A/1/1"):
        tile = {r["candidate"]: r["tiles_median_um"] for r in rows if r["position"] == fov}
        assert tile["exact"] < tile["half way"] < tile["unregistered"]
    assert {"points_median_um", "z_um", "z_from", "scale_yx"} <= set(rows[0])


def test_each_block_of_a_stack_run_is_a_candidate(tmp_path):
    import json

    from biahub.registration.benchmark import block_candidates
    from biahub.utils.config import model_to_yaml
    from tests.test_estimate_transform import (
        APPLIED_SHIFT_ZYX,
        SHAPE,
        _run,
        _synthetic_bead_volume,
        _write_plate,
    )
    from tests.test_stacked_methods import BEADS_BLOCK, _settings

    ref = _synthetic_bead_volume(np.random.default_rng(11), SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    plate = _write_plate(tmp_path / "beads.zarr", [(ref, mov), (ref, mov)])
    config = tmp_path / "stack.yml"
    model_to_yaml(_settings([{"phase_cross_corr": {}}, BEADS_BLOCK]), config)
    output = tmp_path / "out" / "transforms.yml"
    _run(plate, config, output)

    candidates = block_candidates(output.with_suffix(""))
    assert set(candidates) == {"1:phase_cross_corr", "2:bead_matching"}
    record = json.loads((output.with_suffix("") / "timepoints" / "1.json").read_text())
    np.testing.assert_allclose(
        candidates["2:bead_matching"][1].matrix, record["blocks"][1]["matrix"]
    )


def test_a_blank_position_gets_one_row_saying_which_side_is_empty(tmp_path):
    # A FOV whose moving channel is all zeros gave empty rows without a word.
    from biahub.registration.benchmark import evaluate_candidates

    positions = _plate(tmp_path)
    with open_ome_zarr(positions[1], mode="r+") as position:
        position.data[0, 0] = 0  # DAPI written, but blank
    rows = evaluate_candidates(
        {"exact": Transform.from_translation([-s for s in SHIFT])},
        [(p, p) for p in positions],
        "DAPI",
        "VS",
        tile_px=64,
        crop_px=96,
    )
    blank = [r for r in rows if r["position"] == "A/1/1"]
    assert len(blank) == 1
    assert (blank[0]["candidate"], blank[0]["empty"]) == ("unregistered", "moving")
    assert blank[0]["tiles_median_um"] is None
    assert {r["empty"] for r in rows if r["position"] == "A/1/0"} == {None}
