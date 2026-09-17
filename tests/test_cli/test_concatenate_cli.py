import numpy as np
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli

_POSITIONS = [("A", "1", "0"), ("B", "1", "0")]


def _two_sources(create_custom_plate, tmp_path):
    """Two source plates with the same positions and disjoint channels."""
    deskew_path, _ = create_custom_plate(
        tmp_path / "deskew", position_list=_POSITIONS, channel_names=["BF", "GFP"]
    )
    phase_path, _ = create_custom_plate(
        tmp_path / "reconstruct", position_list=_POSITIONS, channel_names=["Phase3D"]
    )
    return deskew_path, phase_path


def _params_only_config(tmp_path, **overrides):
    """A config with NO concat_data_paths: the sources come from -i."""
    config_path = tmp_path / "concat.yml"
    config_path.write_text(
        yaml.dump({"channel_names": "all", "output_ome_zarr_version": "0.4", **overrides})
    )
    return config_path


def _source_args(*groups):
    """`-i <group> -i <group> ...`, each group one source store's positions."""
    args = []
    for group in groups:
        args.append("-i")
        args.extend(str(p) for p in group)
    return args


def _read(output_zarr, key):
    with open_ome_zarr(str(output_zarr / key), mode="r") as ds:
        return ds["0"][:], list(ds.channel_names), dict(ds.zattrs)


def test_init_with_grouped_inputs(create_custom_plate, tmp_path):
    """`--init` with one -i per source scaffolds the plate and prints RESOURCES.

    The config carries no concat_data_paths and a bare `channel_names: all`
    (broadcast to every source), which is what the Nextflow configs look like.
    """
    deskew_path, phase_path = _two_sources(create_custom_plate, tmp_path)
    config_path = _params_only_config(tmp_path)
    output_zarr = tmp_path / "output.zarr"

    result = CliRunner().invoke(
        cli,
        [
            "concatenate",
            "--init",
            *_source_args(sorted(deskew_path.glob("*/*/*")), sorted(phase_path.glob("*/*/*"))),
            "-c",
            str(config_path),
            "-o",
            str(output_zarr),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "RESOURCES:" in result.output
    assert "Initialized" in result.output and "(2 positions)" in result.output
    for key in ("A/1/0", "B/1/0"):
        data, channel_names, zattrs = _read(output_zarr, key)
        assert channel_names == ["BF", "GFP", "Phase3D"]
        assert not data.any()
        assert "biahub-concatenate" in zattrs


def test_per_position_worker_after_init(create_custom_plate, tmp_path):
    """Init, then one position per source with `--cluster debug`: the Nextflow
    worker shape. Only that output position is written, the provenance stamp is
    written once (by init, not by the worker), and running every position
    reproduces a whole-plate single-shot run exactly.
    """
    deskew_path, phase_path = _two_sources(create_custom_plate, tmp_path)
    config_path = _params_only_config(tmp_path)
    output_zarr = tmp_path / "output.zarr"
    all_sources = _source_args(
        sorted(deskew_path.glob("*/*/*")), sorted(phase_path.glob("*/*/*"))
    )
    runner = CliRunner()

    init = runner.invoke(
        cli,
        [
            "concatenate",
            "--init",
            *all_sources,
            "-c",
            str(config_path),
            "-o",
            str(output_zarr),
        ],
    )
    assert init.exit_code == 0, init.output
    _, _, zattrs_before = _read(output_zarr, "A/1/0")

    def run_position(key):
        return runner.invoke(
            cli,
            [
                "concatenate",
                "--cluster",
                "debug",
                *_source_args([deskew_path / key], [phase_path / key]),
                "-c",
                str(config_path),
                "-o",
                str(output_zarr),
            ],
        )

    result = run_position("A/1/0")
    assert result.exit_code == 0, result.output
    assert "Created" not in result.output  # worker scaffolds nothing

    a_data, _, zattrs_after = _read(output_zarr, "A/1/0")
    b_data, _, _ = _read(output_zarr, "B/1/0")
    assert a_data.any() and not b_data.any()
    assert zattrs_after["biahub-concatenate"] == zattrs_before["biahub-concatenate"]

    result = run_position("B/1/0")
    assert result.exit_code == 0, result.output

    # Reference: the whole plate in one in-process run.
    reference_zarr = tmp_path / "reference.zarr"
    single_shot = runner.invoke(
        cli,
        [
            "concatenate",
            "--cluster",
            "debug",
            *all_sources,
            "-c",
            str(config_path),
            "-o",
            str(reference_zarr),
        ],
    )
    assert single_shot.exit_code == 0, single_shot.output
    for key in ("A/1/0", "B/1/0"):
        fanned, channels, _ = _read(output_zarr, key)
        reference, reference_channels, _ = _read(reference_zarr, key)
        assert channels == reference_channels
        np.testing.assert_array_equal(fanned, reference)


def test_cluster_debug_single_shot(create_custom_plate, tmp_path):
    """`concatenate --cluster debug` over whole plates must block until every
    position is written: submitit's DebugJob executes lazily, so without the CLI
    waiting on the results the command would exit having written nothing.
    Sources come from the config here, the pre-`-i` form, which must keep working.
    """
    deskew_path, phase_path = _two_sources(create_custom_plate, tmp_path)
    config_path = tmp_path / "concat.yml"
    config_path.write_text(
        yaml.dump(
            {
                "concat_data_paths": [str(deskew_path) + "/*/*/*", str(phase_path) + "/*/*/*"],
                "channel_names": ["all", "all"],
                "output_ome_zarr_version": "0.4",
            }
        )
    )
    output_zarr = tmp_path / "output.zarr"

    result = CliRunner().invoke(
        cli,
        ["concatenate", "--cluster", "debug", "-c", str(config_path), "-o", str(output_zarr)],
    )

    assert result.exit_code == 0, result.output
    for key in ("A/1/0", "B/1/0"):
        data, channel_names, _ = _read(output_zarr, key)
        assert set(channel_names) == {"BF", "GFP", "Phase3D"}
        assert data.shape[1] == 3
        assert data.any()


def test_per_source_length_mismatch_fails_before_creating_anything(
    create_custom_plate, tmp_path
):
    """A per-source list of the wrong length is caught against the -i count."""
    deskew_path, phase_path = _two_sources(create_custom_plate, tmp_path)
    config_path = _params_only_config(tmp_path, channel_names=["all"])
    output_zarr = tmp_path / "output.zarr"

    result = CliRunner().invoke(
        cli,
        [
            "concatenate",
            "--init",
            *_source_args(sorted(deskew_path.glob("*/*/*")), sorted(phase_path.glob("*/*/*"))),
            "-c",
            str(config_path),
            "-o",
            str(output_zarr),
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "channel_names has 1 entries for 2 sources" in str(result.exception)
    assert not output_zarr.exists()


def test_no_sources_fails(tmp_path):
    """Neither -i nor concat_data_paths: a clear error, not a traceback deep inside."""
    config_path = _params_only_config(tmp_path)
    result = CliRunner().invoke(
        cli, ["concatenate", "--init", "-c", str(config_path), "-o", str(tmp_path / "o.zarr")]
    )
    assert result.exit_code != 0
    assert "No sources" in str(result.exception)
