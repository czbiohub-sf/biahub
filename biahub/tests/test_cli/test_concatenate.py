from biahub.analysis.AnalysisSettings import ConcatenateSettings
from biahub.cli.concatenate import concatenate


def test_concatenate_channels(example_plate, tmp_path, example_concatenate_settings):
    plate_path_1, _ = example_plate
    _, yaml_settings = example_concatenate_settings

    # Update the 'concat_data_paths' key in the settings
    yaml_settings["concat_data_paths"] = [
        str(plate_path_1) + '/*/*/*',
        str(plate_path_1) + '/*/*/*',
    ]

    concatenate(
        settings=ConcatenateSettings(**yaml_settings),
        output_dirpath=tmp_path / "output.zarr",
        local=True,
    )
