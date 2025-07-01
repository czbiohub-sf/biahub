from pathlib import Path
import click
from biahub.cli.parsing import (
    config_filepaths,
    output_dirpath,
)
from biahub.cli.utils import (
    yaml_to_model,
)
from biahub.settings import DatasetSettings


def create_dataset(
    dataset_name: str,
    raw_data_dirpath: str,
    output_dirpath: str,
    config_filepaths: list[str],
    ):
    """
    Create a dataset from raw data.

    
    Example:
    >> biahub create-dataset
        -i ./raw_data/0/0/0               # Input raw data
        -o ./dataset.zarr                 # Output directory for dataset
        -c ./dataset_settings.yml         # Configuration file with dataset settings
        -v                                # Verbose mode for detailed logs (default: False)
        --local                           # Run locally instead of submitting to SLURM (default: False)   

    """

    # Single config file for all FOVs
    if len(config_filepaths) == 1:
        config_filepath = Path(config_filepaths[0])
    else:
        config_filepath = None

    settings = yaml_to_model(config_filepaths[0], DatasetSettings)

    output_dirpath = Path(output_dirpath)
  

    

   
@click.command("create-dataset")
@click.option("--dataset-name", type=str, required=True)
@click.option("--raw-data-dirpath", type=str, required=True)
@config_filepaths()
@output_dirpath()
def create_dataset_cli(
    dataset_name: str,
    raw_data_dirpath: str,
    output_dirpath: str,
    config_filepaths: list[str],
):
    create_dataset(
        dataset_name=dataset_name,
        raw_data_dirpath=raw_data_dirpath,
        output_dirpath=output_dirpath,
        config_filepaths=config_filepaths,
    )
    

if __name__ == "__main__":
    create_dataset_cli()
