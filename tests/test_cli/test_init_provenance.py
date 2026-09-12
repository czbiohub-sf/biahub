"""Provenance is stamped by ``--init``, not by the per-position compute.

The Nextflow pipeline scaffolds every step's output store up front, before any
of them hold data (see ``nextflow/mantis-v2.nf``). ``create_empty_plate``
inherits a source store's provenance at plate-creation time and only for
positions it creates, so a step whose own record landed at compute time would
never be inherited: each downstream plate is created while its input store is
still empty. These tests pin the property that makes the chain hold.
"""

import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli


@pytest.fixture()
def flat_field_config(tmp_path):
    config_path = tmp_path / "flat_field.yml"
    config_path.write_text(yaml.dump({"channel_names": None}))
    return config_path


def _init(runner, command, input_zarr, config, output_zarr):
    result = runner.invoke(
        cli,
        [
            command,
            "-i",
            f"{input_zarr}/A/1/0",
            f"{input_zarr}/B/1/0",
            f"{input_zarr}/B/2/0",
            "-c",
            str(config),
            "-o",
            str(output_zarr),
            "--init",
        ],
    )
    assert result.exit_code == 0, result.output
    return result


def _position_zattrs(store_path):
    """Every position's custom (non-OME) zattrs keys, as a list of key sets."""
    with open_ome_zarr(str(store_path), mode="r") as plate:
        return [
            {key for key in dict(position.zattrs) if key != "ome"}
            for _name, position in plate.positions()
        ]


def test_flat_field_init_stamps_own_provenance(tmp_path, example_plate, flat_field_config):
    plate_path, _ = example_plate
    output_path = tmp_path / "flatfield.zarr"

    _init(CliRunner(), "flat-field", plate_path, flat_field_config, output_path)

    for keys in _position_zattrs(output_path):
        assert "biahub-flat_field" in keys


def test_deskew_init_stamps_own_provenance(tmp_path, example_plate, example_deskew_settings):
    plate_path, _ = example_plate
    config_path, _ = example_deskew_settings
    output_path = tmp_path / "deskew.zarr"

    _init(CliRunner(), "deskew", plate_path, config_path, output_path)

    for keys in _position_zattrs(output_path):
        assert "biahub-deskew" in keys


def test_provenance_chains_across_inits_on_empty_stores(
    tmp_path, example_plate, flat_field_config, example_deskew_settings
):
    """Deskew's ``--init`` inherits flat-field's record from a store with no data.

    This is the pipeline's init phase in miniature: nothing is computed between
    the two calls, so the only way ``biahub-flat_field`` can reach the deskewed
    plate is if flat-field's ``--init`` wrote it.
    """
    plate_path, _ = example_plate
    deskew_config, _ = example_deskew_settings
    flat_field_out = tmp_path / "flatfield.zarr"
    deskew_out = tmp_path / "deskew.zarr"

    runner = CliRunner()
    _init(runner, "flat-field", plate_path, flat_field_config, flat_field_out)
    _init(runner, "deskew", flat_field_out, deskew_config, deskew_out)

    for keys in _position_zattrs(deskew_out):
        assert {"biahub-flat_field", "biahub-deskew"} <= keys


def test_init_refreshes_its_own_record_on_rerun(
    tmp_path, example_plate, example_deskew_settings
):
    """A changed config updates the record rather than leaving the first one.

    ``create_empty_plate`` is idempotent and skips the metadata copy for
    positions that already exist, so this only holds because the step's own
    record is written on every position the call names.
    """
    plate_path, _ = example_plate
    _, settings = example_deskew_settings
    output_path = tmp_path / "deskew.zarr"
    runner = CliRunner()

    first = tmp_path / "deskew_first.yml"
    first.write_text(yaml.dump({**settings, "keep_overhang": True}))
    _init(runner, "deskew", plate_path, first, output_path)

    second = tmp_path / "deskew_second.yml"
    second.write_text(yaml.dump({**settings, "keep_overhang": False}))
    _init(runner, "deskew", plate_path, second, output_path)

    with open_ome_zarr(str(output_path), mode="r") as plate:
        for _name, position in plate.positions():
            assert position.zattrs["biahub-deskew"]["keep_overhang"] is False
