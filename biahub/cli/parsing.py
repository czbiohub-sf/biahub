from pathlib import Path
from typing import Callable

import click

from iohub.ngff import Plate, open_ome_zarr
from natsort import natsorted

from biahub.cli.option_eat_all import OptionEatAll


def _validate_and_process_paths(
    ctx: click.Context, opt: click.Option, value: str
) -> list[Path]:
    # Sort and validate the input paths
    input_paths = [Path(path) for path in natsorted(value)]
    with open_ome_zarr(input_paths[0], mode='r') as dataset:
        if isinstance(dataset, Plate):
            raise ValueError(
                "Please supply a single position instead of an HCS plate. Likely fix: replace 'input.zarr' with 'input.zarr/0/0/0'"
            )
    return input_paths


def _str_to_path(ctx: click.Context, opt: click.Option, value: str) -> Path:
    return Path(value)


def input_position_dirpaths() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--input-position-dirpaths",
            "-i",
            required=True,
            cls=OptionEatAll,
            type=tuple,
            callback=_validate_and_process_paths,
            help='Paths to input positions, for example: "input.zarr/0/0/0" or "input.zarr/*/*/*"',
        )(f)

    return decorator


def source_position_dirpaths() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--source-position-dirpaths",
            "-s",
            required=True,
            cls=OptionEatAll,
            type=tuple,
            callback=_validate_and_process_paths,
            help='Paths to source positions, for example: "source.zarr/0/0/0" or "source.zarr/*/*/*"',
        )(f)

    return decorator


def target_position_dirpaths() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--target-position-dirpaths",
            "-t",
            required=True,
            cls=OptionEatAll,
            type=tuple,
            callback=_validate_and_process_paths,
            help='Paths to target positions, for example: "target.zarr/0/0/0" or "target.zarr/*/*/*"',
        )(f)

    return decorator


def config_filepath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--config-filepath",
            "-c",
            required=True,
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
            callback=_str_to_path,
            help="Path to YAML configuration file.",
        )(f)

    return decorator


def output_dirpath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--output-dirpath",
            "-o",
            required=True,
            type=click.Path(exists=False, file_okay=False, dir_okay=True),
            help="Path to output directory",
            callback=_str_to_path,
        )(f)

    return decorator


def output_filepath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--output-filepath",
            "-o",
            required=True,
            type=click.Path(exists=False, file_okay=True, dir_okay=False),
            callback=_str_to_path,
            help="Path to output file",
        )(f)

    return decorator


def sbatch_filepath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--sbatch-filepath",
            "-sb",
            default=None,
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
            help="SBATCH filepath that contains slurm parameters to overwrite defaults. "
            "For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU.",
        )(f)

    return decorator


def sbatch_filepath_preprocess() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--sbatch-filepath-preprocess",
            "-sb-preprocess",
            default=None,
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
            help="SBATCH filepath that contains slurm parameters to overwrite defaults. "
            "For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU.",
        )(f)

    return decorator


def sbatch_filepath_predict() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--sbatch-filepath-predict",
            "-sb-predict",
            default=None,
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
            help="SBATCH filepath that contains slurm parameters to overwrite defaults. "
            "For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU.",
        )(f)

    return decorator


def sbatch_to_submitit(filepath: str) -> dict:
    """Reads a text configuration file and returns a dictionary of parameters
    which can be passed to the submitit executor. This file can contain parameters
    starting with #SBATCH to configure SLURM jobs or parameters starting with #LOCAL
    to configure local jobs. The submitit executor will only apply valid parameters
    and will, for example, ignore local parameters when running on SLURM.

    Parameters
    ----------
    value : Path
        Path to sbatch file

    Returns
    -------
    dict
        Dictionary of slurm parameters

    Example:

    --- sbatch_file.sh ---
    #SBATCH --mem-per-cpu=16G
    #SBATCH --time=1:00:00
    #LOCAL --cpus-per-task=1
    ---

    >>> dict = sbatch_to_submitit(Path("sbatch_file.sh"))
    >>> print(dict)
    {'slurm_mem_per_cpu': '16G', 'slurm_time': '1:00:00', 'cpus_per_task': 1}
    """

    with open(filepath, "r") as f:
        sbatch_file = f.readlines()

    keywords = ["SBATCH", "LOCAL"]
    sbatch_dict = {}
    for line in sbatch_file:
        for keyword in keywords:
            if line.startswith(f"#{keyword} --"):
                line = line.strip(f"#{keyword} --").strip()
                key, value = line.split("=", 1)
                key = key.replace("-", "_").strip()
                try:
                    value = int(value.strip())
                except ValueError:
                    # If conversion to int fails, keep it as a string
                    value = value.strip()
                if keyword == "SBATCH":
                    sbatch_dict["slurm_" + key] = value
                elif keyword == "LOCAL":
                    sbatch_dict[key] = value

    return sbatch_dict


def local() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--local",
            "-l",
            is_flag=True,
            default=False,
            help="Run jobs locally instead of submitting to SLURM.",
        )(f)

    return decorator


def monitor() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--monitor",
            "-m",
            is_flag=True,
            default=False,
            help="Monitor of submitted SLURM jobs.",
        )(f)

    return decorator


def num_processes() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--num-processes",
            "-j",
            default=1,
            help="Number of parallel processes",
            required=False,
            type=int,
        )(f)

    return decorator
