from pathlib import Path

import pytest
import yaml

from pydantic import ValidationError

from biahub.settings import AirtableSettings, DcaFinalizeSettings


def test_base_config_must_be_toml():
    with pytest.raises(ValidationError):
        DcaFinalizeSettings(base_config="x.yml")


def test_default_paths_resolve_next_to_plate():
    settings = DcaFinalizeSettings(base_config="base.toml")
    plate = Path("/data/run/plate.zarr")
    assert settings.confirmed_path(plate) == Path("/data/run/plate.dca-confirmed.toml")
    assert settings.metadata_table_path(plate) is None


def test_explicit_paths_win_over_defaults(tmp_path):
    confirmed = tmp_path / "c.toml"
    table = tmp_path / "t.tsv"
    settings = DcaFinalizeSettings(
        base_config="base.toml", confirmed_config=confirmed, metadata_table=table
    )
    plate = Path("/data/run/plate.zarr")
    assert settings.confirmed_path(plate) == confirmed
    assert settings.metadata_table_path(plate) == table


@pytest.mark.parametrize("suffix", ["csv", "tsv"])
def test_default_metadata_table_found_next_to_plate(tmp_path, suffix):
    plate = tmp_path / "plate.zarr"
    table = tmp_path / f"plate.dca-metadata.{suffix}"
    table.write_text("a\n1\n")
    settings = DcaFinalizeSettings(base_config="base.toml")
    assert settings.metadata_table_path(plate) == table


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        DcaFinalizeSettings(base_config="base.toml", typo=1)


def test_airtable_token_env_default():
    airtable = AirtableSettings(base_id="appXXXXXXXXXXXXXX", table="samples")
    assert airtable.token_env == "AIRTABLE_API_KEY"


def test_from_yaml_resolves_relative_paths_against_yaml_dir(tmp_path):
    config = tmp_path / "conf" / "dca_finalize.yml"
    config.parent.mkdir()
    config.write_text(
        yaml.safe_dump({"base_config": "dca_base.toml", "confirmed_config": "sub/c.toml"})
    )
    settings = DcaFinalizeSettings.from_yaml(config)
    assert settings.base_config == config.parent / "dca_base.toml"
    assert settings.confirmed_config == config.parent / "sub" / "c.toml"


def test_from_yaml_keeps_absolute_paths(tmp_path):
    config = tmp_path / "dca_finalize.yml"
    absolute = tmp_path / "elsewhere" / "base.toml"
    config.write_text(yaml.safe_dump({"base_config": str(absolute)}))
    assert DcaFinalizeSettings.from_yaml(config).base_config == absolute


def test_a549_config_validates():
    config = Path("nextflow/configs/a549/dca_finalize.yml")
    settings = DcaFinalizeSettings.from_yaml(config)
    assert settings.base_config == config.parent / "dca_base.toml"
    assert settings.airtable is None
