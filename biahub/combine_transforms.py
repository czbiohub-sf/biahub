from pathlib import Path

import click
import numpy as np

from biahub.cli.utils import model_to_yaml, yaml_to_model
from biahub.settings import StabilizationSettings


@click.command("combine-transforms")
@click.option("--config-a", "-a", required=True, type=click.Path(exists=True))
@click.option("--config-b", "-b", required=True, type=click.Path(exists=True))
@click.option("--output-config", "-o", required=True, type=click.Path())
def combine_transforms_cli(config_a: str, config_b: str, output_config: str):
    """Compose two per-FOV transform lists: output[t] = A[t] @ B[t].

    Used in the xyz focus-finding workflow to combine z and xy transforms.
    """
    settings_a = yaml_to_model(Path(config_a), StabilizationSettings)
    settings_b = yaml_to_model(Path(config_b), StabilizationSettings)

    transforms_a = np.array(settings_a.affine_transform_zyx_list)
    transforms_b = np.array(settings_b.affine_transform_zyx_list)

    if len(transforms_a) != len(transforms_b):
        raise click.ClickException(
            f"Transform count mismatch: {len(transforms_a)} vs {len(transforms_b)}"
        )

    composed = np.array([a @ b for a, b in zip(transforms_a, transforms_b, strict=True)])

    output_model = settings_a.model_copy()
    output_model.affine_transform_zyx_list = composed.tolist()

    output_path = Path(output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_to_yaml(output_model, output_path)

    click.echo(f"Combined transforms written to {output_config}")
