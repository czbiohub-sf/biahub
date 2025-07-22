
from pathlib import Path
from typing import List

import ants
import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as R

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    estimate_resources,
    yaml_to_model,
)
from biahub.register import convert_transform_to_ants
from biahub.settings import StabilizationSettings
from biahub.cli.main import cli
from biahub.settings import ProcessingDataset



           
from pathlib import Path

def create_experiment(config_path: Path):
    settings = yaml_to_model(config_path, ProcessingDataset)
    print(settings)
    create_experiment_folders(settings)


def create_experiment_folders(config: ProcessingDataset):
    base_path = Path(config.path)
    dataset = config.dataset_name
    pipeline = config.pipeline

    root = base_path / dataset
    root.mkdir(parents=True, exist_ok=True)

    for key in pipeline:
        folder_path = root / key
        folder_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    
    create_experiment(Path("/hpc/mydata/taylla.theodoro/repo/biahub/settings/example_create_experiment.yml"))
