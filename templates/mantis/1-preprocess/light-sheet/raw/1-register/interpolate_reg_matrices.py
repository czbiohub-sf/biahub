# %%
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from biahub.analysis.AnalysisSettings import RegistrationSettings, StabilizationSettings
from biahub.cli.utils import model_to_yaml, yaml_to_model
from scipy.interpolate import CubicSpline, interp1d

files = os.listdir(".")
reg_files = [f for f in files if "registration_settings.yml" in f]
# %%

t_idx = [int(t.split("_")[0][1:]) for t in reg_files]

t_idx = np.array(t_idx)

matrices = []
for file in reg_files:
    model = yaml_to_model(file, RegistrationSettings)
    matrices.append(model.affine_transform_zyx)
matrices = np.stack(matrices)

# %%
f = interp1d(t_idx, matrices, axis=0)

output_model = StabilizationSettings(
    stabilization_estimation_channel="",
    stabilization_type="xyz",
    stabilization_channels=model.source_channel_names,
    affine_transform_zyx_list=[],
    time_indices="all",
)

for t in range(t_idx[-1] + 1):
    matrix = f(t)
    matrix[0, 3] = np.round(matrix[0, 3])
    output_model.affine_transform_zyx_list.append(matrix.tolist())

# %%
model_to_yaml(output_model, "timelapse_registration_settings.yml")
# %%
