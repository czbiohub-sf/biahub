import numpy as np
import pytest
import yaml

from click.testing import CliRunner
from iohub import open_ome_zarr
from scipy.ndimage import shift as ndi_shift

from biahub.cli.main import cli
from biahub.settings import load_transform_settings
from tests.test_estimate_transform import (
    APPLIED_SHIFT_ZYX,
    SHAPE,
    _synthetic_bead_volume,
    _write_plate,
)

CHANNELS = ("Phase3D", "GFP")
PEAKS = {"threshold_abs": 100, "nms_distance": 4, "min_distance": 0, "block_size": [8, 8, 8]}


@pytest.fixture(autouse=True)
def _in_process_jobs(monkeypatch):
    monkeypatch.setenv("CI", "true")  # get_submitit_cluster -> "debug": jobs run in-process


@pytest.fixture
def beads_plate(tmp_path):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    return _write_plate(tmp_path / "beads.zarr", [(ref, mov), (ref, mov)])


def _legacy_registration_config(path):
    path.write_text(
        yaml.safe_dump(
            {
                "source_channel_name": "GFP",
                "target_channel_name": "Phase3D",
                "estimation_method": "beads",
                "beads_match_settings": {
                    "source_peaks_settings": PEAKS,
                    "target_peaks_settings": PEAKS,
                },
                "affine_transform_settings": {
                    "transform_type": "euclidean",
                    "use_prev_t_transform": False,
                },
            }
        )
    )
    return path


def test_aliases_are_hidden_from_help():
    output = CliRunner().invoke(cli, ["--help"]).output
    hidden = ("estimate-registration", "estimate-stabilization", "register ", "stabilize ")
    for name in (*hidden, "convert-settings"):
        assert name not in output
    assert "estimate-transform" in output and "apply-transform" in output


def test_estimate_registration_alias_runs_estimate_transform_and_warns(beads_plate, tmp_path):
    config = _legacy_registration_config(tmp_path / "estimate-registration.yml")
    output = tmp_path / "out" / "registration_settings.yml"

    result = CliRunner().invoke(
        cli,
        [
            "estimate-registration",
            "-s",
            str(beads_plate),
            "-t",
            str(beads_plate),
            "-c",
            str(config),
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DeprecationWarning" in result.output and "estimate-transform" in result.output
    model = load_transform_settings(output)  # the new transforms file, at the old path
    inverse = [model._as(e.matrix, "inverse")[:3, 3] for e in model.transforms]
    for row in inverse:
        np.testing.assert_allclose(row, APPLIED_SHIFT_ZYX, atol=0.5)


def test_register_alias_applies_a_legacy_registration_config(beads_plate, tmp_path):
    # Source and target are the same store, as in main's register: only the source channels
    # move, the target channel is copied.
    shift_y = np.eye(4)
    shift_y[1, 3] = 4.0
    config = tmp_path / "register.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "source_channel_names": ["GFP"],
                "target_channel_name": "Phase3D",
                "affine_transform_zyx": shift_y.tolist(),
                "keep_overhang": True,
            }
        )
    )
    output = tmp_path / "registered.zarr"

    result = CliRunner().invoke(
        cli,
        [
            "register",
            "-s",
            str(beads_plate),
            "-t",
            str(beads_plate),
            "-c",
            str(config),
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DeprecationWarning" in result.output and "apply-transform" in result.output
    with open_ome_zarr(beads_plate) as source:
        phase, gfp = (
            np.asarray(source.data[0, source.get_channel_index(c)]) for c in CHANNELS
        )
    with open_ome_zarr(output / "A" / "1" / "0") as registered:
        assert registered.channel_names == list(CHANNELS)
        np.testing.assert_array_equal(registered.data[0, 0], phase)
        assert not np.allclose(registered.data[0, 1], gfp)


def test_optimize_registration_points_to_the_replacement():
    result = CliRunner().invoke(cli, ["optimize-registration", "-c", "x.yml"])
    assert result.exit_code != 0 and "estimate-transform" in result.output


@pytest.fixture
def drifting_plate(tmp_path):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    frames = [
        ndi_shift(ref, shift=tuple(t * np.array(APPLIED_SHIFT_ZYX)), order=1, mode="constant")
        for t in range(3)
    ]
    return _write_plate(tmp_path / "drift.zarr", [(f, f) for f in frames])


def test_estimate_stabilization_alias_converts_a_legacy_pcc_config(drifting_plate, tmp_path):
    config = tmp_path / "estimate-stabilization.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "stabilization_estimation_channel": "GFP",
                "stabilization_channels": ["GFP"],
                "stabilization_type": "xyz",
                "stabilization_method": "phase-cross-corr",
                "phase_cross_corr_settings": {
                    "t_reference": "first",
                    "center_crop_xy": [SHAPE[1], SHAPE[2]],
                },
            }
        )
    )
    result = CliRunner().invoke(
        cli,
        [
            "estimate-stabilization",
            "-i",
            str(drifting_plate),
            "-c",
            str(config),
            "-o",
            str(tmp_path / "stab"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DeprecationWarning" in result.output
    model = load_transform_settings(tmp_path / "stab" / "transforms.yml")
    for t, entry in enumerate(model.transforms):
        np.testing.assert_allclose(
            model._as(entry.matrix, "inverse")[:3, 3],
            t * np.array(APPLIED_SHIFT_ZYX),
            atol=0.5,
        )
    assert str(tmp_path / "stab" / "transforms.yml") in result.output

    # The old second step still finds it: legacy wrote per-FOV files (or, for beads, one
    # file) under these names, next to which the alias now writes transforms.yml.
    stab = tmp_path / "stab"
    for old_path in (
        stab / "xyz_stabilization_settings" / "*.yml",
        stab / "xy_stabilization_settings.yml",
    ):
        output = tmp_path / f"stabilized_{old_path.parent.name}.zarr"
        result = CliRunner().invoke(
            cli,
            ["stabilize", "-i", str(drifting_plate), "-c", str(old_path), "-o", str(output)],
        )
        assert result.exit_code == 0, result.output
        assert f"using {stab / 'transforms.yml'}" in result.output
        assert (output / "A" / "1" / "0").exists()

    # Anything else that matches nothing is still an error.
    result = CliRunner().invoke(
        cli,
        ["stabilize", "-i", str(drifting_plate), "-c", str(stab / "other" / "*.yml")]
        + ["-o", str(tmp_path / "x.zarr")],
    )
    assert result.exit_code != 0 and "No files matched" in result.output


def test_stabilize_alias_applies_a_legacy_stabilization_config(drifting_plate, tmp_path):
    config = tmp_path / "stabilize.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "stabilization_estimation_channel": "GFP",
                "stabilization_type": "xyz",
                "stabilization_method": "phase-cross-corr",
                "stabilization_channels": ["GFP"],
                "affine_transform_zyx_list": [np.eye(4).tolist()] * 3,
                "output_voxel_size": [1, 1, 1, 1, 1],
            }
        )
    )
    output = tmp_path / "stabilized.zarr"
    result = CliRunner().invoke(
        cli, ["stabilize", "-i", str(drifting_plate), "-c", str(config), "-o", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert "DeprecationWarning" in result.output and (output / "A" / "1" / "0").exists()


@pytest.fixture
def three_positions(tmp_path):
    """A plate with positions A/1/0 (beads), A/1/1 and A/2/0 (cells)."""
    data = np.zeros((1, 2, 4, 8, 8), dtype=np.float32)
    path = tmp_path / "plate.zarr"
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=list(CHANNELS)) as plate:
        for row, col, fov in (("A", "1", "0"), ("A", "1", "1"), ("A", "2", "0")):
            plate.create_position(row, col, fov)["0"] = data
    return [path / "A" / "1" / "0", path / "A" / "1" / "1", path / "A" / "2" / "0"]


@pytest.fixture
def estimated_positions(monkeypatch):
    """Record the positions each alias hands to estimate-transform, without running it."""
    calls = []

    def record(moving, config, output, reference_position_dirpaths=None, **kwargs):
        calls.append(
            {
                "moving": [str(p) for p in moving],
                "reference": [str(p) for p in reference_position_dirpaths or []],
            }
        )

    monkeypatch.setattr("biahub.registration.legacy.aliases.estimate_transform", record)
    return calls


def test_estimate_registration_alias_estimates_on_the_first_position_only(
    three_positions, estimated_positions, tmp_path
):
    # Legacy estimate-registration read only the first source and target position.
    config = _legacy_registration_config(tmp_path / "estimate-registration.yml")
    paths = [str(p) for p in three_positions]
    result = CliRunner().invoke(
        cli,
        ["estimate-registration", "-s", *paths, "-t", *paths, "-c", str(config)]
        + ["-o", str(tmp_path / "registration.yml")],
    )
    assert result.exit_code == 0, result.output
    assert estimated_positions == [{"moving": paths[:1], "reference": paths[:1]}]


@pytest.mark.parametrize(
    "method, block, expected",
    [
        # beads: the beads FOV only (the first position), one shared transforms file
        ("beads", {}, [0]),
        # the others: every position except skip_beads_fov (a substring of the path)
        (
            "phase-cross-corr",
            {"phase_cross_corr_settings": {"skip_beads_fov": "A/1/0"}},
            [1, 2],
        ),
        ("focus-finding", {"focus_finding_settings": {"skip_beads_fov": "A/1/0"}}, [1, 2]),
        ("focus-finding", {"stack_reg_settings": {"skip_beads_fov": "A/1/0"}}, [1, 2]),
        ("focus-finding", {}, [0, 1, 2]),
    ],
)
def test_estimate_stabilization_alias_picks_positions_as_legacy_did(
    three_positions, estimated_positions, tmp_path, method, block, expected
):
    config = tmp_path / "estimate-stabilization.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "stabilization_estimation_channel": "GFP",
                "stabilization_channels": ["GFP"],
                "stabilization_type": "xyz",
                "stabilization_method": method,
                **block,
            }
        )
    )
    paths = [str(p) for p in three_positions]
    result = CliRunner().invoke(
        cli,
        ["estimate-stabilization", "-i", *paths, "-c", str(config), "-o", str(tmp_path / "s")],
    )
    assert result.exit_code == 0, result.output
    assert estimated_positions[0]["moving"] == [paths[i] for i in expected]
