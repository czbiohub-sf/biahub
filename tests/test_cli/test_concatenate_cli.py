import numpy as np
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli


def _make_plate(path, channel_names, position_keys, shape=(2, 2, 4, 16, 16), dtype=np.float32):
    """Create a test plate with given channels and positions."""
    plate = open_ome_zarr(path, layout="hcs", mode="w", channel_names=channel_names)
    for row, col, fov in position_keys:
        position = plate.create_position(row, col, fov)
        position["0"] = np.random.uniform(0.1, 1.0, size=shape).astype(dtype)
    plate.close()
    return path


def _make_concat_config(
    config_path,
    concat_data_paths,
    channel_names=None,
    z_slice="all",
    y_slice="all",
    x_slice="all",
):
    """Write a ConcatenateSettings YAML config."""
    if channel_names is None:
        channel_names = ["all"] * len(concat_data_paths)
    config = {
        "concat_data_paths": concat_data_paths,
        "channel_names": channel_names,
        "time_indices": "all",
        "Z_slice": z_slice,
        "Y_slice": y_slice,
        "X_slice": x_slice,
        "output_ome_zarr_version": "0.4",
        "ensure_unique_positions": False,
    }
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return config_path


class TestEstimateCropCli:
    def test_init_mode(self, tmp_path):
        """Test --init emits RESOURCES and POSITION lines."""
        positions = [("B", "3", "000000"), ("B", "3", "000001")]

        lf_plate = _make_plate(
            tmp_path / "deskew.zarr", ["BF"], positions, shape=(2, 1, 4, 16, 16)
        )
        ls_plate = _make_plate(
            tmp_path / "reconstruct.zarr", ["Phase3D"], positions, shape=(2, 1, 4, 16, 16)
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate-crop",
                "--init",
                "--lf-data-path",
                str(lf_plate) + "/*/*/*",
                "--ls-data-path",
                str(ls_plate) + "/*/*/*",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "RESOURCES:" in result.output
        assert "POSITION:" in result.output
        position_lines = [line for line in result.output.splitlines() if line.startswith("POSITION:")]
        assert len(position_lines) == 2

    def test_single_fov_mode(self, tmp_path):
        """Test per-FOV mode emits RANGES."""
        positions = [("B", "3", "000000")]

        lf_plate = _make_plate(
            tmp_path / "deskew.zarr", ["BF"], positions, shape=(1, 1, 4, 16, 16)
        )
        ls_plate = _make_plate(
            tmp_path / "reconstruct.zarr", ["Phase3D"], positions, shape=(1, 1, 4, 16, 16)
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate-crop",
                "--lf-position",
                str(lf_plate / "B" / "3" / "000000"),
                "--ls-position",
                str(ls_plate / "B" / "3" / "000000"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "RANGES:" in result.output
        ranges_line = [line for line in result.output.splitlines() if line.startswith("RANGES:")]
        assert len(ranges_line) == 1
        parts = ranges_line[0].replace("RANGES:", "").strip().split()
        assert len(parts) == 3

    def test_reduce_mode(self, tmp_path):
        """Test --reduce aggregates ranges into config."""
        config_path = _make_concat_config(
            tmp_path / "concat.yml",
            concat_data_paths=["dummy/*/*/*", "dummy2/*/*/*"],
        )

        ranges_file = tmp_path / "ranges.txt"
        ranges_file.write_text(
            "RANGES:0,80 10,900 50,1000\n"
            "RANGES:5,82 15,910 55,1050\n"
            "RANGES:2,78 12,895 48,990\n"
        )

        output_config = tmp_path / "cropped.yml"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate-crop",
                "--reduce",
                "-c",
                str(config_path),
                "-o",
                str(output_config),
                "--ranges-file",
                str(ranges_file),
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_config.exists()

        with open(output_config) as f:
            resolved = yaml.safe_load(f)

        assert resolved["Z_slice"] == [5, 78]
        assert resolved["Y_slice"] == [15, 895]
        assert resolved["X_slice"] == [55, 990]

    def test_reduce_mode_with_concat_data_paths(self, tmp_path):
        """Test --reduce with --concat-data-paths override."""
        config_path = _make_concat_config(
            tmp_path / "concat.yml",
            concat_data_paths=["placeholder/*/*/*"],
        )

        ranges_file = tmp_path / "ranges.txt"
        ranges_file.write_text("RANGES:0,80 10,900 50,1000\n")

        output_config = tmp_path / "cropped.yml"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate-crop",
                "--reduce",
                "-c",
                str(config_path),
                "-o",
                str(output_config),
                "--ranges-file",
                str(ranges_file),
                "--concat-data-paths",
                "real/path1/*/*/*",
                "--concat-data-paths",
                "real/path2/*/*/*",
            ],
        )

        assert result.exit_code == 0, result.output
        with open(output_config) as f:
            resolved = yaml.safe_load(f)
        assert resolved["concat_data_paths"] == ["real/path1/*/*/*", "real/path2/*/*/*"]


class TestConcatenateCli:
    def test_resolve_config_mode(self, tmp_path):
        """Test --resolve-config replaces concat_data_paths."""
        config_path = _make_concat_config(
            tmp_path / "concat.yml",
            concat_data_paths=["placeholder/*/*/*"],
        )
        output_config = tmp_path / "resolved.yml"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "concatenate",
                "--resolve-config",
                "-c",
                str(config_path),
                "-o",
                str(output_config),
                "--concat-data-paths",
                "real/deskew.zarr/*/*/*",
                "--concat-data-paths",
                "real/reconstruct.zarr/*/*/*",
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_config.exists()

        with open(output_config) as f:
            resolved = yaml.safe_load(f)
        assert resolved["concat_data_paths"] == [
            "real/deskew.zarr/*/*/*",
            "real/reconstruct.zarr/*/*/*",
        ]

    def test_init_mode(self, tmp_path):
        """Test --init creates output plate and emits RESOURCES."""
        positions = [("B", "3", "000000"), ("B", "3", "000001")]
        plate1 = _make_plate(
            tmp_path / "source1.zarr", ["ch_a", "ch_b"], positions, shape=(2, 2, 4, 16, 16)
        )
        plate2 = _make_plate(
            tmp_path / "source2.zarr", ["ch_c"], positions, shape=(2, 1, 4, 16, 16)
        )

        config_path = _make_concat_config(
            tmp_path / "concat.yml",
            concat_data_paths=[
                str(plate1) + "/*/*/*",
                str(plate2) + "/*/*/*",
            ],
        )

        output_zarr = tmp_path / "output.zarr"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "concatenate",
                "--init",
                "-c",
                str(config_path),
                "-o",
                str(output_zarr),
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_zarr.exists()
        assert "RESOURCES:" in result.output

        with open_ome_zarr(str(output_zarr / "B" / "3" / "000000"), mode="r") as ds:
            assert set(ds.channel_names) == {"ch_a", "ch_b", "ch_c"}

    def test_cluster_debug_single_position(self, tmp_path):
        """Test --cluster debug -p runs one position in-process."""
        positions = [("B", "3", "000000")]
        plate1 = _make_plate(
            tmp_path / "source1.zarr", ["ch_a"], positions, shape=(2, 1, 4, 16, 16)
        )
        plate2 = _make_plate(
            tmp_path / "source2.zarr", ["ch_b"], positions, shape=(2, 1, 4, 16, 16)
        )

        config_path = _make_concat_config(
            tmp_path / "concat.yml",
            concat_data_paths=[
                str(plate1) + "/*/*/*",
                str(plate2) + "/*/*/*",
            ],
        )

        output_zarr = tmp_path / "output.zarr"

        runner = CliRunner()

        # Step 1: init
        result = runner.invoke(
            cli,
            ["concatenate", "--init", "-c", str(config_path), "-o", str(output_zarr)],
        )
        assert result.exit_code == 0, result.output

        # Step 2: run one position
        result = runner.invoke(
            cli,
            [
                "concatenate",
                "--cluster",
                "debug",
                "-c",
                str(config_path),
                "-o",
                str(output_zarr),
                "-p",
                "B/3/000000",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Concatenation done: B/3/000000" in result.output

        with open_ome_zarr(str(output_zarr / "B" / "3" / "000000"), mode="r") as ds:
            data = ds["0"][:]
            assert data.shape[0] > 0
            assert not np.all(data == 0)

    def test_init_and_run_with_cropping(self, tmp_path):
        """Test init + per-position run with Z/Y/X slicing."""
        positions = [("A", "1", "0")]
        plate = _make_plate(
            tmp_path / "source.zarr", ["ch1", "ch2"], positions, shape=(2, 2, 8, 32, 32)
        )

        config_path = _make_concat_config(
            tmp_path / "concat.yml",
            concat_data_paths=[str(plate) + "/*/*/*"],
            z_slice=[1, 5],
            y_slice=[4, 20],
            x_slice=[4, 20],
        )

        output_zarr = tmp_path / "output.zarr"

        runner = CliRunner()

        # Init
        result = runner.invoke(
            cli,
            ["concatenate", "--init", "-c", str(config_path), "-o", str(output_zarr)],
        )
        assert result.exit_code == 0, result.output

        # Run
        result = runner.invoke(
            cli,
            [
                "concatenate",
                "--cluster",
                "debug",
                "-c",
                str(config_path),
                "-o",
                str(output_zarr),
                "-p",
                "A/1/0",
            ],
        )
        assert result.exit_code == 0, result.output

        with open_ome_zarr(str(output_zarr / "A" / "1" / "0"), mode="r") as ds:
            data = ds["0"][:]
            assert data.shape == (2, 2, 4, 16, 16)
