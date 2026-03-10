from pathlib import Path
import numpy as np
from copy import deepcopy
from biahub.settings import StabilizationSettings
from biahub.cli.utils import model_to_yaml, yaml_to_model

# Load full list of registration matrices

# Folders with per-FOV Z and XY transform configs
z_dir = Path("z_stabilization_settings")
xy_dir = Path("../../../label-free/2-stabilize/xy_stabilization_settings")
output_dir = Path("./combined_stabilization_settings")
output_dir.mkdir(exist_ok=True)

for z_file in sorted(z_dir.glob("*.yml")):
    fov_name = z_file.stem
    xy_file = xy_dir / f"{fov_name}.yml"
    if not xy_file.exists():
        raise FileNotFoundError(f"No matching XY transform for FOV {fov_name}")

    z_settings = yaml_to_model(z_file, StabilizationSettings)
    xy_settings = yaml_to_model(xy_file, StabilizationSettings)

    z_list = np.array(z_settings.affine_transform_zyx_list)
    xy_list = np.array(xy_settings.affine_transform_zyx_list)

    if not (len(z_list) == len(xy_list)):
        raise ValueError(f"Transform list lengths mismatch for FOV {fov_name}")

    combined_list = [
        (z @ xy).tolist()
        for z, xy in zip(z_list, xy_list)
    ]

    combined_settings = deepcopy(z_settings)  # copy metadata + non-transform fields
    combined_settings.affine_transform_zyx_list = combined_list

    model_to_yaml(combined_settings, output_dir / f"{fov_name}.yml")

