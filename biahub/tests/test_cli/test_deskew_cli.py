import numpy as np


from click.testing import CliRunner

from biahub import deskew
from biahub.cli.main import cli
import pytest


def test_average_n_slices():
    data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]
    )
    # Non-divisible window
    averaged_data1 = deskew._average_n_slices(data, average_window_width=3)
    expected_data1 = np.array([[[5, 6], [7, 8]], [[13, 14], [15, 16]]])
    assert np.array_equal(averaged_data1, expected_data1)
    assert averaged_data1.shape == deskew._get_averaged_shape(data.shape, 3)

    # Divisible window
    averaged_data2 = deskew._average_n_slices(data, average_window_width=2)
    expected_data2 = np.array([[[3, 4], [5, 6]], [[11, 12], [13, 14]]])
    assert np.array_equal(averaged_data2, expected_data2)
    assert averaged_data2.shape == deskew._get_averaged_shape(data.shape, 2)

    # Window = 1
    averaged_data3 = deskew._average_n_slices(data, average_window_width=1)
    assert np.array_equal(averaged_data3, data)
    assert averaged_data3.shape == deskew._get_averaged_shape(data.shape, 1)


def test_deskew_data():
    raw_data = np.random.random((2, 3, 4))
    px_to_scan_ratio = 0.386
    pixel_size_um = 1.0
    ls_angle_deg = 36
    average_n_slices = 1
    keep_overhang = True
    deskewed_data = deskew.deskew(
        raw_data, ls_angle_deg, px_to_scan_ratio, keep_overhang, average_n_slices
    )
    assert deskewed_data.shape[1] == 4
    assert deskewed_data[0, 0, 0] != 0  # indicates incorrect shifting

    assert (
        deskewed_data.shape
        == deskew.get_deskewed_data_shape(
            raw_data.shape,
            ls_angle_deg,
            px_to_scan_ratio,
            keep_overhang,
            pixel_size_um=pixel_size_um,
        )[0]
    )


def test_deskew_cli(tmp_path, example_plate, example_deskew_settings, sbatch_file):
    plate_path, _ = example_plate
    config_path, _ = example_deskew_settings
    output_path = tmp_path / "output.zarr"

    # Test deskew cli
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "deskew",
            "-i",
            str(plate_path) + "/A/1/0",
            str(plate_path) + "/B/1/0",
            str(plate_path) + "/B/2/0",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "--local",
            "--sbatch-filepath",
            sbatch_file,
        ],
    )

    assert output_path.exists()
    assert result.exit_code == 0
    def test_deskew_overhang_only_dataset_error():
        # Parameters that cause only overhang
        shape = (10, 50, 100)
        angle = 36
        ratio = 0.1

        with pytest.raises(ValueError, match="Dataset contains only overhang"):
            deskew.get_deskewed_data_shape(shape, angle, ratio, keep_overhang=False)

        # Should succeed with keep_overhang=True
        out_shape, _ = deskew.get_deskewed_data_shape(shape, angle, ratio, keep_overhang=True)
        assert out_shape[2] > 0

    def test_deskew_function_overhang_only_dataset_error():
        data = np.random.random((10, 50, 100))
        angle = 36
        ratio = 0.1

        with pytest.raises(ValueError, match="Dataset contains only overhang"):
            deskew.deskew(data, angle, ratio, keep_overhang=False)
