import numpy as np
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli


def test_resolve_config_blank_concat_data_paths(tmp_path):
    """`--resolve-config` must fill a blank concat_data_paths in the source config.

    Regression guard: the Nextflow config leaves concat_data_paths empty for the
    pipeline to inject, so resolution must apply the override before validating
    (concat_data_paths is a required non-empty list on ConcatenateSettings).
    """
    config_path = tmp_path / "concatenate.yml"
    config_path.write_text(
        "concat_data_paths:\n\n"
        "time_indices: all\n"
        "channel_names:\n- all\n- all\n"
        'output_ome_zarr_version: "0.5"\n'
    )
    resolved = tmp_path / "concatenate_resolved.yml"

    result = CliRunner().invoke(
        cli,
        [
            "concatenate",
            "--resolve-config",
            "-c",
            str(config_path),
            "-o",
            str(resolved),
            "--concat-data-paths",
            "deskew.zarr/*/*/*",
            "--concat-data-paths",
            "reconstruct.zarr/*/*/*",
        ],
    )

    assert result.exit_code == 0, result.output
    out = yaml.safe_load(resolved.read_text())
    assert out["concat_data_paths"] == ["deskew.zarr/*/*/*", "reconstruct.zarr/*/*/*"]


def test_cluster_debug_single_shot(create_custom_plate, tmp_path):
    """`concatenate --cluster debug` (no -p) must run the whole plate in-process
    and block until every position is written.

    Regression guard: submitit's DebugJob executes lazily, so without the CLI
    blocking on the results the command would exit having written nothing.
    """
    position_list = [("A", "1", "0"), ("B", "1", "0")]
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "deskew", position_list=position_list, channel_names=["BF"]
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "reconstruct", position_list=position_list, channel_names=["Phase3D"]
    )

    config_path = tmp_path / "concat.yml"
    config_path.write_text(
        yaml.dump(
            {
                "concat_data_paths": [
                    str(plate_1_path) + "/*/*/*",
                    str(plate_2_path) + "/*/*/*",
                ],
                "channel_names": ["all", "all"],
                "time_indices": "all",
                "Z_slice": "all",
                "Y_slice": "all",
                "X_slice": "all",
                "output_ome_zarr_version": "0.4",
                "ensure_unique_positions": False,
            }
        )
    )

    output_zarr = tmp_path / "output.zarr"

    result = CliRunner().invoke(
        cli,
        [
            "concatenate",
            "--cluster",
            "debug",
            "-c",
            str(config_path),
            "-o",
            str(output_zarr),
        ],
    )

    assert result.exit_code == 0, result.output

    for key in ("A/1/0", "B/1/0"):
        with open_ome_zarr(str(output_zarr / key), mode="r") as ds:
            data = ds["0"][:]
            assert set(ds.channel_names) == {"BF", "Phase3D"}
            assert data.shape[1] == 2
            assert not np.all(data == 0)
