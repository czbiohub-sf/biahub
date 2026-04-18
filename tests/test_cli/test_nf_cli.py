from unittest.mock import patch

import numpy as np
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli


def test_list_positions(example_plate):
    plate_path, _ = example_plate
    runner = CliRunner()
    result = runner.invoke(cli, ["nf", "list-positions", "-i", str(plate_path)])

    assert result.exit_code == 0
    lines = [line for line in result.output.strip().splitlines() if line]
    assert len(lines) == 3
    assert "A/1/0" in lines
    assert "B/1/0" in lines
    assert "B/2/0" in lines


def test_init_resources(example_plate):
    """init-resources reads input shape and outputs RESOURCES line."""
    plate_path, _ = example_plate
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "init-resources",
            "-i",
            str(plate_path),
            "-r",
            "8",
        ],
    )

    assert result.exit_code == 0
    assert "RESOURCES:" in result.output


def test_init_flat_field(tmp_path, example_plate):
    plate_path, _ = example_plate
    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "ff_config.yml"
    config_path.write_text(yaml.dump({"channel_names": None}))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "init-flat-field",
            "-i",
            str(plate_path),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "RESOURCES:" in result.output


# ---------------------------------------------------------------------------
# rename-channels
# ---------------------------------------------------------------------------


def test_rename_channels_with_prefix(example_plate):
    """Rename channels by adding a prefix — metadata-only, no data copy."""
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "rename-channels",
            "-i",
            str(plate_path),
            "-p",
            "A/1/0",
            "--prefix",
            "raw ",
        ],
    )

    assert result.exit_code == 0, result.output

    with open_ome_zarr(str(plate_path), mode="r") as plate:
        pos = plate["A/1/0"]
        for ch in pos.channel_names:
            assert ch.startswith("raw ")

        other = plate["B/1/0"]
        assert not any(ch.startswith("raw ") for ch in other.channel_names)


def test_rename_channels_with_suffix(example_plate):
    """Rename channels by adding a suffix."""
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "rename-channels",
            "-i",
            str(plate_path),
            "-p",
            "B/1/0",
            "--suffix",
            " decon",
        ],
    )

    assert result.exit_code == 0, result.output

    with open_ome_zarr(str(plate_path), mode="r") as plate:
        pos = plate["B/1/0"]
        for ch in pos.channel_names:
            assert ch.endswith(" decon")


def test_rename_channels_no_prefix_or_suffix_fails(example_plate):
    """Must provide at least --prefix or --suffix."""
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "rename-channels",
            "-i",
            str(plate_path),
            "-p",
            "A/1/0",
        ],
    )

    assert result.exit_code != 0


def test_rename_channels_succeeds(example_plate):
    """Command should succeed with valid prefix."""
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "rename-channels",
            "-i",
            str(plate_path),
            "-p",
            "A/1/0",
            "--prefix",
            "raw ",
        ],
    )

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# init-track / run-track
# ---------------------------------------------------------------------------


def _make_tracking_config(tmp_path, plate_path, blank_frames_csv=None):
    """Write a minimal TrackingSettings YAML for testing."""
    cfg = {
        "mode": "2D",
        "z_range": [-1, -1],
        "target_channel": "GFP",
        "input_images": [
            {
                "path": str(plate_path),
                "channels": {
                    "GFP": [],
                },
            },
            {
                "path": None,
                "channels": {
                    "foreground": [
                        {
                            "function": "np.ones_like",
                            "input_channels": ["GFP"],
                            "kwargs": {},
                        }
                    ],
                    "contour": [
                        {
                            "function": "np.zeros_like",
                            "input_channels": ["GFP"],
                            "kwargs": {},
                        }
                    ],
                },
            },
        ],
        "tracking_config": {},
    }
    config_path = tmp_path / "track_config.yml"
    config_path.write_text(yaml.dump(cfg))
    return config_path


def test_init_track(tmp_path, example_plate):
    """init-track creates output zarr with uint32 dtype, single label channel."""
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    output_path = tmp_path / "track_output.zarr"
    config_path = _make_tracking_config(tmp_path, plate_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "init-track",
            "-i",
            str(plate_path),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "RESOURCES:" in result.output

    with open_ome_zarr(str(output_path), mode="r") as out:
        pos = out["A/1/0"]
        assert pos.data.dtype == np.uint32
        assert len(pos.channel_names) == 1
        assert "label" in pos.channel_names[0].lower() or "GFP" in pos.channel_names[0]


def test_run_track(tmp_path, example_plate):
    """run-track invokes track_one_position with correct args."""
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    output_path = tmp_path / "track_output.zarr"
    config_path = _make_tracking_config(tmp_path, plate_path)

    # First create the output zarr via init-track
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "nf",
            "init-track",
            "-i",
            str(plate_path),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
        ],
    )

    with patch("biahub.track.track_one_position") as mock_track:
        result = runner.invoke(
            cli,
            [
                "nf",
                "run-track",
                "-o",
                str(output_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
            ],
        )

        assert result.exit_code == 0, result.output
        mock_track.assert_called_once()
        call_kwargs = mock_track.call_args
        assert call_kwargs.kwargs["position_key"] == ("A", "1", "0")
        assert call_kwargs.kwargs["output_dirpath"] == output_path


# ---------------------------------------------------------------------------
# estimate-crop / init-concatenate / run-concatenate (assembly)
# ---------------------------------------------------------------------------


def _make_concat_config(tmp_path, plate_paths, channel_names_per_path=None):
    """Write a minimal ConcatenateSettings YAML for testing."""
    if channel_names_per_path is None:
        channel_names_per_path = ["all"] * len(plate_paths)
    cfg = {
        "concat_data_paths": [str(p) + "/*/*/*" for p in plate_paths],
        "channel_names": channel_names_per_path,
        "time_indices": "all",
        "X_slice": "all",
        "Y_slice": "all",
        "Z_slice": "all",
        "output_ome_zarr_version": "0.5",
    }
    config_path = tmp_path / "concat_config.yml"
    config_path.write_text(yaml.dump(cfg))
    return config_path


def test_init_concatenate(tmp_path, example_plate, example_plate_2):
    """init-concatenate creates output zarr with combined channels from all inputs."""
    plate_path_1, ds1 = example_plate
    ds1.close()
    plate_path_2, ds2 = example_plate_2
    ds2.close()

    output_path = tmp_path / "concat_output.zarr"
    config_path = _make_concat_config(
        tmp_path,
        [plate_path_1, plate_path_2],
        channel_names_per_path=[["Phase3D"], ["GFP"]],
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "init-concatenate",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "RESOURCES:" in result.output

    with open_ome_zarr(str(output_path), mode="r") as out:
        pos = out["A/1/0"]
        assert "Phase3D" in pos.channel_names
        assert "GFP" in pos.channel_names


def test_init_concatenate_all_channels(tmp_path, example_plate):
    """init-concatenate with 'all' channels uses all channels from the input plate."""
    plate_path, ds = example_plate
    ds.close()

    output_path = tmp_path / "concat_all_output.zarr"
    config_path = _make_concat_config(tmp_path, [plate_path])

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "init-concatenate",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output

    with open_ome_zarr(str(output_path), mode="r") as out:
        pos = out["A/1/0"]
        assert len(pos.channel_names) == 6


def test_run_concatenate(tmp_path, example_plate):
    """run-concatenate copies data for a single position into the output plate."""
    plate_path, ds = example_plate
    ds.close()

    output_path = tmp_path / "concat_run_output.zarr"
    config_path = _make_concat_config(
        tmp_path,
        [plate_path],
        channel_names_per_path=[["GFP", "RFP"]],
    )

    runner = CliRunner()
    # First, init the output plate
    result = runner.invoke(
        cli,
        ["nf", "init-concatenate", "-c", str(config_path), "-o", str(output_path)],
    )
    assert result.exit_code == 0, result.output

    # Then run concatenate for one position
    result = runner.invoke(
        cli,
        [
            "nf",
            "run-concatenate",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "-p",
            "A/1/0",
        ],
    )

    assert result.exit_code == 0, result.output

    with open_ome_zarr(str(output_path), mode="r") as out:
        pos = out["A/1/0"]
        data = np.asarray(pos[0][:])
        assert data.shape[1] == 2
        assert not np.all(data == 0)


def test_run_concatenate_with_crop(tmp_path, example_plate):
    """run-concatenate applies Z/Y/X slicing when specified in config."""
    plate_path, ds = example_plate
    ds.close()

    cfg = {
        "concat_data_paths": [str(plate_path) + "/*/*/*"],
        "channel_names": [["GFP"]],
        "time_indices": "all",
        "X_slice": [1, 4],
        "Y_slice": [1, 3],
        "Z_slice": [0, 2],
        "output_ome_zarr_version": "0.5",
    }
    config_path = tmp_path / "concat_crop_config.yml"
    config_path.write_text(yaml.dump(cfg))

    output_path = tmp_path / "concat_crop_output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["nf", "init-concatenate", "-c", str(config_path), "-o", str(output_path)],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        cli,
        [
            "nf",
            "run-concatenate",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "-p",
            "A/1/0",
        ],
    )

    assert result.exit_code == 0, result.output

    with open_ome_zarr(str(output_path), mode="r") as out:
        pos = out["A/1/0"]
        data = np.asarray(pos[0][:])
        assert data.shape == (3, 1, 2, 2, 3)


def test_estimate_crop(tmp_path, example_plate, example_plate_2):
    """estimate-crop computes crop slices and writes updated config."""
    plate_path_1, ds1 = example_plate
    ds1.close()
    plate_path_2, ds2 = example_plate_2
    ds2.close()

    cfg = {
        "concat_data_paths": [str(plate_path_1) + "/*/*/*", str(plate_path_2) + "/*/*/*"],
        "channel_names": [["Phase3D"], ["GFP"]],
        "time_indices": "all",
        "X_slice": "all",
        "Y_slice": "all",
        "Z_slice": "all",
        "output_ome_zarr_version": "0.5",
    }
    config_path = tmp_path / "concat_for_crop.yml"
    config_path.write_text(yaml.dump(cfg))

    output_config = tmp_path / "cropped_concat.yml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "estimate-crop",
            "-c",
            str(config_path),
            "-o",
            str(output_config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_config.exists()
    assert "RESOURCES:" in result.output

    with open(output_config) as f:
        updated = yaml.safe_load(f)
    assert (
        updated["Z_slice"] != "all"
        or updated["Y_slice"] != "all"
        or updated["X_slice"] != "all"
    )


# ---------------------------------------------------------------------------
# Stabilization: combine-transforms / init-stabilize / run-stabilize
# ---------------------------------------------------------------------------


def _make_stabilization_config(tmp_path, transforms, filename="stab_config.yml"):
    """Write a minimal StabilizationSettings YAML for testing."""
    cfg = {
        "stabilization_estimation_channel": "GFP",
        "stabilization_type": "xyz",
        "stabilization_method": "focus-finding",
        "stabilization_channels": ["GFP", "RFP"],
        "affine_transform_zyx_list": [t.tolist() if hasattr(t, "tolist") else t for t in transforms],
        "time_indices": "all",
        "output_voxel_size": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    config_path = tmp_path / filename
    config_path.write_text(yaml.dump(cfg))
    return config_path


def _identity_transforms(n_timepoints):
    """Return a list of n identity 4x4 matrices."""
    return [np.eye(4).tolist() for _ in range(n_timepoints)]


def _translation_transforms(n_timepoints, dz=0.0, dy=0.0, dx=0.0):
    """Return a list of n translation 4x4 matrices."""
    transforms = []
    for _ in range(n_timepoints):
        t = np.eye(4)
        t[0, 3] = dz
        t[1, 3] = dy
        t[2, 3] = dx
        transforms.append(t.tolist())
    return transforms


def test_combine_transforms(tmp_path):
    """combine-transforms composes two transform lists: output[t] = A[t] @ B[t]."""
    n_t = 3
    transforms_a = _translation_transforms(n_t, dz=2.0)
    transforms_b = _translation_transforms(n_t, dy=5.0)

    config_a = _make_stabilization_config(tmp_path, transforms_a, "config_a.yml")
    config_b = _make_stabilization_config(tmp_path, transforms_b, "config_b.yml")
    output_config = tmp_path / "combined.yml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "combine-transforms",
            "-a",
            str(config_a),
            "-b",
            str(config_b),
            "-o",
            str(output_config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_config.exists()

    with open(output_config) as f:
        updated = yaml.safe_load(f)

    combined = np.array(updated["affine_transform_zyx_list"])
    assert combined.shape == (n_t, 4, 4)

    for t in range(n_t):
        expected = np.array(transforms_a[t]) @ np.array(transforms_b[t])
        np.testing.assert_allclose(combined[t], expected)


def test_combine_transforms_mismatched_lengths(tmp_path):
    """combine-transforms fails when transform lists have different lengths."""
    config_a = _make_stabilization_config(
        tmp_path, _identity_transforms(3), "config_a.yml"
    )
    config_b = _make_stabilization_config(
        tmp_path, _identity_transforms(5), "config_b.yml"
    )
    output_config = tmp_path / "combined.yml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "combine-transforms",
            "-a",
            str(config_a),
            "-b",
            str(config_b),
            "-o",
            str(output_config),
        ],
    )

    assert result.exit_code != 0


def test_init_stabilize(tmp_path, example_plate):
    """init-stabilize creates output zarr with correct shape and emits RESOURCES."""
    plate_path, ds = example_plate
    ds.close()

    n_t = 3  # example_plate has T=3
    config_path = _make_stabilization_config(
        tmp_path, _identity_transforms(n_t), "stab_config.yml"
    )
    output_path = tmp_path / "stabilized.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "init-stabilize",
            "-i",
            str(plate_path),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "RESOURCES:" in result.output

    with open_ome_zarr(str(output_path), mode="r") as out:
        pos = out["A/1/0"]
        assert pos.data.dtype == np.float32
        assert pos.data.shape[0] == n_t
        assert pos.data.shape[1] == 6  # all channels preserved


def test_run_stabilize(tmp_path, example_plate):
    """run-stabilize invokes process_single_position_v2 with correct args."""
    plate_path, ds = example_plate
    ds.close()

    n_t = 3
    config_path = _make_stabilization_config(
        tmp_path, _identity_transforms(n_t), "stab_config.yml"
    )
    output_path = tmp_path / "stabilized.zarr"

    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "nf",
            "init-stabilize",
            "-i",
            str(plate_path),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
        ],
    )

    with patch("biahub.cli.utils.process_single_position_v2") as mock_process:
        result = runner.invoke(
            cli,
            [
                "nf",
                "run-stabilize",
                "-i",
                str(plate_path),
                "-o",
                str(output_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "-j",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert mock_process.call_count == 6  # one per channel
        call_kwargs = mock_process.call_args_list[0]
        assert call_kwargs.kwargs["num_threads"] == 2
        assert "list_of_shifts" in call_kwargs.kwargs
        assert "output_shape" in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# Estimation: z-focus / xy / pcc / beads
# ---------------------------------------------------------------------------


def _make_estimate_stabilization_config(tmp_path, filename="est_stab_config.yml", **overrides):
    """Write a minimal EstimateStabilizationSettings YAML for testing."""
    cfg = {
        "stabilization_estimation_channel": "GFP",
        "stabilization_channels": ["GFP", "RFP"],
        "stabilization_type": "xyz",
        "stabilization_method": "focus-finding",
        "verbose": False,
    }
    cfg.update(overrides)
    config_path = tmp_path / filename
    config_path.write_text(yaml.dump(cfg))
    return config_path


def test_estimate_stabilization_z_focus(tmp_path, example_plate):
    """estimate-stabilization-z-focus calls estimate_z_focus_per_position."""
    plate_path, ds = example_plate
    ds.close()

    config_path = _make_estimate_stabilization_config(tmp_path, stabilization_type="z")
    output_dir = tmp_path / "z_focus_output"

    with patch(
        "biahub.estimate_stabilization.estimate_z_focus_per_position"
    ) as mock_z_focus:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nf",
                "estimate-stabilization-z-focus",
                "-i",
                str(plate_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        mock_z_focus.assert_called_once()
        call_kwargs = mock_z_focus.call_args.kwargs
        assert "A/1/0" in str(call_kwargs["input_position_dirpath"])


def test_estimate_stabilization_xy(tmp_path, example_plate):
    """estimate-stabilization-xy calls estimate_xy_stabilization_per_position with focus CSV."""
    plate_path, ds = example_plate
    ds.close()

    config_path = _make_estimate_stabilization_config(tmp_path, stabilization_type="xy")

    focus_csv = tmp_path / "positions_focus.csv"
    focus_csv.write_text("position,time_idx,channel,focus_idx\nA/1/0,0,GFP,10\nA/1/0,1,GFP,11\nA/1/0,2,GFP,12\n")

    output_dir = tmp_path / "xy_output"

    with patch(
        "biahub.estimate_stabilization.estimate_xy_stabilization_per_position"
    ) as mock_xy:
        mock_xy.return_value = np.zeros((3, 4, 4))
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nf",
                "estimate-stabilization-xy",
                "-i",
                str(plate_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "--focus-csv",
                str(focus_csv),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        mock_xy.assert_called_once()
        call_kwargs = mock_xy.call_args.kwargs
        assert call_kwargs["df_z_focus_path"] == focus_csv


def test_estimate_stabilization_pcc(tmp_path, example_plate):
    """estimate-stabilization-pcc calls estimate_xyz_stabilization_pcc_per_position."""
    plate_path, ds = example_plate
    ds.close()

    config_path = _make_estimate_stabilization_config(
        tmp_path,
        stabilization_type="xyz",
        stabilization_method="phase-cross-corr",
    )
    output_dir = tmp_path / "pcc_output"

    with patch(
        "biahub.estimate_stabilization.estimate_xyz_stabilization_pcc_per_position"
    ) as mock_pcc:
        mock_pcc.return_value = [np.eye(4).tolist() for _ in range(3)]
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nf",
                "estimate-stabilization-pcc",
                "-i",
                str(plate_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        mock_pcc.assert_called_once()


def test_estimate_stabilization_beads(tmp_path, example_plate):
    """estimate-stabilization-beads calls beads.estimate_tczyx on a single FOV."""
    plate_path, ds = example_plate
    ds.close()

    config_path = _make_estimate_stabilization_config(
        tmp_path,
        stabilization_type="xyz",
        stabilization_method="beads",
    )
    output_dir = tmp_path / "beads_output"

    with patch("biahub.registration.beads.estimate_tczyx") as mock_beads:
        mock_beads.return_value = [np.eye(4).tolist() for _ in range(3)]
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nf",
                "estimate-stabilization-beads",
                "-i",
                str(plate_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        mock_beads.assert_called_once()
        call_kwargs = mock_beads.call_args.kwargs
        assert call_kwargs["mode"] == "stabilization"


# ---------------------------------------------------------------------------
# Deconvolution: estimate-psf / init-deconvolve / run-deconvolve
# ---------------------------------------------------------------------------


def test_estimate_psf(tmp_path, example_plate):
    """estimate-psf calls detect_peaks/extract_beads and writes psf.zarr."""
    plate_path, ds = example_plate
    ds.close()

    output_path = tmp_path / "psf.zarr"
    config = {
        "axis0_patch_size": 101,
        "axis1_patch_size": 101,
        "axis2_patch_size": 101,
    }
    config_path = tmp_path / "psf_config.yml"
    config_path.write_text(yaml.dump(config))

    fake_peaks = np.array([[2, 2, 3]])
    fake_beads = [np.random.randn(4, 5, 6).astype(np.float32)]

    with (
        patch("biahub.characterize_psf.detect_peaks", return_value=fake_peaks) as mock_detect,
        patch(
            "biahub.characterize_psf.extract_beads",
            return_value=(fake_beads, [fake_peaks[0]]),
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nf",
                "estimate-psf",
                "-i",
                str(plate_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "-o",
                str(output_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_path.exists()
        mock_detect.assert_called_once()

    with open_ome_zarr(str(output_path), mode="r") as out:
        pos = out["0/0/0"]
        assert pos.data.shape[0] == 1
        assert pos.data.shape[1] == 1
        assert pos.data.dtype == np.float32


def test_init_deconvolve(tmp_path, example_plate):
    """init-deconvolve creates output plate, computes transfer function, emits RESOURCES."""
    plate_path, ds = example_plate
    ds.close()

    # Create a minimal PSF zarr (single position, single channel)
    psf_path = tmp_path / "psf.zarr"
    psf_ds = open_ome_zarr(psf_path, layout="hcs", mode="w", channel_names=["PSF"])
    pos = psf_ds.create_position("0", "0", "0")
    pos["0"] = np.random.randn(1, 1, 4, 5, 6).astype(np.float32)
    psf_ds.close()

    output_path = tmp_path / "deconv_output.zarr"
    tf_path = tmp_path / "transfer_function.zarr"
    config = {"regularization_strength": 0.001}
    config_path = tmp_path / "deconv_config.yml"
    config_path.write_text(yaml.dump(config))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "init-deconvolve",
            "-i",
            str(plate_path),
            "-o",
            str(output_path),
            "--psf-zarr",
            str(psf_path),
            "--tf-zarr",
            str(tf_path),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert tf_path.exists()
    assert "RESOURCES:" in result.output

    with open_ome_zarr(str(output_path), mode="r") as out:
        pos = out["A/1/0"]
        assert pos.data.shape == (3, 6, 4, 5, 6)

    with open_ome_zarr(str(tf_path), mode="r") as tf:
        assert tf["0"].shape[-3:] == (4, 5, 6)


def test_run_deconvolve(tmp_path, example_plate):
    """run-deconvolve calls process_single_position with deconvolve and correct args."""
    plate_path, ds = example_plate
    ds.close()

    # Create minimal PSF and TF zarrs
    psf_path = tmp_path / "psf.zarr"
    psf_ds = open_ome_zarr(psf_path, layout="hcs", mode="w", channel_names=["PSF"])
    pos = psf_ds.create_position("0", "0", "0")
    pos["0"] = np.random.randn(1, 1, 4, 5, 6).astype(np.float32)
    psf_ds.close()

    output_path = tmp_path / "deconv_output.zarr"
    tf_path = tmp_path / "transfer_function.zarr"
    config = {"regularization_strength": 0.001}
    config_path = tmp_path / "deconv_config.yml"
    config_path.write_text(yaml.dump(config))

    runner = CliRunner()
    # Init first
    runner.invoke(
        cli,
        [
            "nf",
            "init-deconvolve",
            "-i",
            str(plate_path),
            "-o",
            str(output_path),
            "--psf-zarr",
            str(psf_path),
            "--tf-zarr",
            str(tf_path),
            "-c",
            str(config_path),
        ],
    )

    with patch("biahub.cli.nf.process_single_position") as mock_process:
        result = runner.invoke(
            cli,
            [
                "nf",
                "run-deconvolve",
                "-i",
                str(plate_path),
                "-o",
                str(output_path),
                "--tf-zarr",
                str(tf_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "-j",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args
        assert call_kwargs.kwargs["num_processes"] == 2
        assert "transfer_function_store_path" in call_kwargs.kwargs
        assert "regularization_strength" in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------


def test_flip_x(example_plate):
    """flip --x flips data along X axis in-place for a single position."""
    plate_path, ds = example_plate
    ds.close()

    with open_ome_zarr(str(plate_path), mode="r") as plate:
        original = np.asarray(plate["A/1/0"]["0"][0, 0])

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "flip",
            "-i",
            str(plate_path),
            "-p",
            "A/1/0",
            "--x",
        ],
    )

    assert result.exit_code == 0, result.output

    with open_ome_zarr(str(plate_path), mode="r") as plate:
        flipped = np.asarray(plate["A/1/0"]["0"][0, 0])

    np.testing.assert_array_equal(flipped, original[:, :, ::-1])


def test_flip_y(example_plate):
    """flip --y flips data along Y axis in-place for a single position."""
    plate_path, ds = example_plate
    ds.close()

    with open_ome_zarr(str(plate_path), mode="r") as plate:
        original = np.asarray(plate["A/1/0"]["0"][0, 0])

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "flip",
            "-i",
            str(plate_path),
            "-p",
            "A/1/0",
            "--y",
        ],
    )

    assert result.exit_code == 0, result.output

    with open_ome_zarr(str(plate_path), mode="r") as plate:
        flipped = np.asarray(plate["A/1/0"]["0"][0, 0])

    np.testing.assert_array_equal(flipped, original[:, ::-1, :])


def test_flip_no_axis_fails(example_plate):
    """flip without --x or --y should fail."""
    plate_path, ds = example_plate
    ds.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "flip",
            "-i",
            str(plate_path),
            "-p",
            "A/1/0",
        ],
    )

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Registration: estimate-registration / optimize-registration / init-register / run-register
# ---------------------------------------------------------------------------


def _make_estimate_registration_config(tmp_path, filename="est_reg_config.yml", **overrides):
    """Write a minimal EstimateRegistrationSettings YAML for testing."""
    cfg = {
        "target_channel_name": "GFP",
        "source_channel_name": "RFP",
        "estimation_method": "beads",
        "verbose": False,
    }
    cfg.update(overrides)
    config_path = tmp_path / filename
    config_path.write_text(yaml.dump(cfg))
    return config_path


def _make_registration_config(tmp_path, filename="reg_config.yml", **overrides):
    """Write a minimal RegistrationSettings YAML for testing."""
    cfg = {
        "source_channel_names": ["Phase3D"],
        "target_channel_name": "GFP",
        "affine_transform_zyx": np.eye(4).tolist(),
        "keep_overhang": True,
        "time_indices": "all",
    }
    cfg.update(overrides)
    config_path = tmp_path / filename
    config_path.write_text(yaml.dump(cfg))
    return config_path


def test_estimate_registration_beads(tmp_path, example_plate):
    """estimate-registration calls beads.estimate_tczyx and writes output YAML."""
    plate_path, ds = example_plate
    ds.close()

    config_path = _make_estimate_registration_config(tmp_path, estimation_method="beads")
    output_dir = tmp_path / "reg_output"

    with patch("biahub.registration.beads.estimate_tczyx") as mock_beads:
        mock_beads.return_value = [np.eye(4).tolist()]
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nf",
                "estimate-registration",
                "--source-zarr",
                str(plate_path),
                "--target-zarr",
                str(plate_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        mock_beads.assert_called_once()
        call_kwargs = mock_beads.call_args.kwargs
        assert call_kwargs["mode"] == "registration"
        assert (output_dir / "registration_settings.yml").exists()


def test_estimate_registration_ants(tmp_path, example_plate):
    """estimate-registration calls ants.estimate_tczyx when method is ants."""
    plate_path, ds = example_plate
    ds.close()

    config_path = _make_estimate_registration_config(tmp_path, estimation_method="ants")
    output_dir = tmp_path / "reg_output"

    with patch("biahub.registration.ants.estimate_tczyx") as mock_ants:
        mock_ants.return_value = [np.eye(4).tolist()]
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nf",
                "estimate-registration",
                "--source-zarr",
                str(plate_path),
                "--target-zarr",
                str(plate_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        mock_ants.assert_called_once()


def test_optimize_registration(tmp_path, example_plate):
    """optimize-registration calls _optimize_registration and writes updated config."""
    plate_path, ds = example_plate
    ds.close()

    config_path = _make_registration_config(
        tmp_path,
        source_channel_names=["RFP"],
        target_channel_name="GFP",
    )
    output_path = tmp_path / "optimized_config.yml"

    with patch(
        "biahub.optimize_registration._optimize_registration"
    ) as mock_optimize:
        mock_optimize.return_value = np.eye(4)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nf",
                "optimize-registration",
                "--source-zarr",
                str(plate_path),
                "--target-zarr",
                str(plate_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "-o",
                str(output_path),
            ],
        )

        assert result.exit_code == 0, result.output
        mock_optimize.assert_called_once()
        assert output_path.exists()


def test_init_register(tmp_path, example_plate):
    """init-register creates empty output zarr with correct channels."""
    plate_path, ds = example_plate
    ds.close()

    config_path = _make_registration_config(
        tmp_path,
        source_channel_names=["Phase3D"],
        target_channel_name="GFP",
        keep_overhang=True,
    )
    output_path = tmp_path / "registered.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "init-register",
            "--source-zarr",
            str(plate_path),
            "--target-zarr",
            str(plate_path),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "RESOURCES:" in result.output
    assert output_path.exists()

    with open_ome_zarr(str(output_path), mode="r") as out:
        pos = out["A/1/0"]
        ch_names = out.channel_names
        assert "Phase3D" in ch_names
        assert "GFP" in ch_names


def test_run_register(tmp_path, example_plate):
    """run-register calls process_single_position_v2 with apply_affine_transform."""
    plate_path, ds = example_plate
    ds.close()

    config_path = _make_registration_config(
        tmp_path,
        source_channel_names=["Phase3D"],
        target_channel_name="GFP",
        keep_overhang=True,
    )

    output_path = tmp_path / "registered.zarr"
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "nf",
            "init-register",
            "--source-zarr",
            str(plate_path),
            "--target-zarr",
            str(plate_path),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
        ],
    )

    with patch("biahub.cli.utils.process_single_position_v2") as mock_process:
        result = runner.invoke(
            cli,
            [
                "nf",
                "run-register",
                "--source-zarr",
                str(plate_path),
                "--target-zarr",
                str(plate_path),
                "-o",
                str(output_path),
                "-p",
                "A/1/0",
                "-c",
                str(config_path),
                "-j",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert mock_process.call_count >= 1
        first_call = mock_process.call_args_list[0]
        assert first_call.kwargs["num_threads"] == 2


# ---------------------------------------------------------------------------
# Stitch: estimate-stitch / stitch
# ---------------------------------------------------------------------------


def test_estimate_stitch(tmp_path, example_plate):
    """estimate-stitch reads stage positions and writes StitchSettings YAML."""
    plate_path, ds = example_plate
    ds.close()

    output_path = tmp_path / "stitch_settings.yml"

    with patch(
        "biahub.estimate_stitch.extract_stage_position"
    ) as mock_stage:
        mock_stage.return_value = (0.0, 100.0, 200.0)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nf",
                "estimate-stitch",
                "-i",
                str(plate_path),
                "-o",
                str(output_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_path.exists()


def test_stitch(tmp_path, example_plate):
    """stitch per-well creates output and calls write_output_chunk."""
    plate_path, ds = example_plate
    ds.close()

    stitch_cfg = {
        "channels": ["GFP", "RFP"],
        "total_translation": {
            "A/1/0": [0.0, 0.0, 0.0],
        },
    }
    config_path = tmp_path / "stitch_config.yml"
    config_path.write_text(yaml.dump(stitch_cfg))
    output_path = tmp_path / "stitched.zarr"

    with patch("biahub.stitch.write_output_chunk") as mock_write:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nf",
                "stitch",
                "-i",
                str(plate_path),
                "-o",
                str(output_path),
                "--well",
                "A/1",
                "-c",
                str(config_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_path.exists()
