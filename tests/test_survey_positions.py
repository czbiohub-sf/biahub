"""Survey positions before registering: which are usable, and how they differ."""

import numpy as np

from iohub import open_ome_zarr

UM = (0.2, 0.1, 0.1)


def _plate(tmp_path):
    """Four positions: cells everywhere, cells in one corner, nothing written, all NaN."""
    from tests.test_registration_estimators import _synthetic_blob_volume

    rng = np.random.default_rng(0)
    spread = _synthetic_blob_volume(rng, (24, 256, 256), n_blobs=300, sigma=3.0) + 10
    corner = np.full_like(spread, 10.0)
    corner[:, :80, :80] = spread[:, :80, :80]
    path = tmp_path / "plate.zarr"
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["DAPI", "VS"]) as plate:
        for fov, vol in (("0", spread), ("1", corner), ("3", np.full_like(spread, np.nan))):
            plate.create_position("A", "1", fov)["0"] = np.stack([vol, vol])[None]
        plate.create_position("A", "1", "2").create_zeros(
            "0", shape=(1, 2, *spread.shape), dtype=np.float32
        )
    return [path / "A" / "1" / fov for fov in "0123"]


def test_the_survey_names_unusable_positions_and_how_structure_is_spread(tmp_path):
    from biahub.registration.evaluate import survey_positions

    rows = {r["position"]: r for r in survey_positions(_plate(tmp_path), ["DAPI", "VS"], t=0)}
    assert rows["A/1/2"]["empty"] and rows["A/1/3"]["nan_fraction"] == 1.0
    assert not rows["A/1/0"]["empty"] and rows["A/1/0"]["nan_fraction"] == 0.0
    # structure spread over the field constrains scale and rotation; a corner does not
    assert rows["A/1/0"]["spread"] > 0.8 > 0.3 > rows["A/1/1"]["spread"]
    assert rows["A/1/0"]["foreground"] > rows["A/1/1"]["foreground"]
    # the depth of peak signal, per channel
    assert set(rows["A/1/0"]["peak_z"]) == {"DAPI", "VS"}
