# %%
import ants
import os
import yaml
import numpy as np
from pathlib import Path
from iohub import open_ome_zarr
import napari
from biahub.cli.register import find_overlapping_volume, find_overlapping_volume
from biahub.analysis.register import convert_transform_to_ants, convert_transform_to_numpy
from skimage import exposure, filters
from biahub.cli.utils import model_to_yaml, yaml_to_model
from biahub.analysis.AnalysisSettings import StabilizationSettings

os.environ['DISPLAY'] = ':1'

# %%
dataset = "2025_05_01_A549_DENV_sensor_DENV"
fov = "B/2/001001"
root = Path("/hpc/projects/intracellular_dashboard/viral-sensor") / dataset
source_path = root / f"1-preprocess/light-sheet/raw/0-deskew/{dataset}.zarr/" / fov
target_path = root / f"1-preprocess/label-free/2-stabilize/phase/{dataset}.zarr/" / fov
registration_settings_path = (
    root / "1-preprocess/light-sheet/raw/1-register/estimate-registration-beads.yml"
)

# source_channel = "Cy5 EX639 EM698-70"
# target_chahnel = "nuclei_prediction"

# source_channel = "mCherry EX561 EM600-37"
target_chahnel = "Phase3D"

t_idx = 13

# approx_tform = np.array([
#     [1.0, 0.0, 0.0, -5],
#     [0.0, 0.046, -1.287, 2100], # 1972
#     [0.0, 1.287, 0.046, -480],
#     [0.0, 0.0, 0.0, 1.0],
# ])
with open(registration_settings_path, mode='r') as fp:
    settings = yaml.safe_load(fp)
    approx_tform = np.asarray(settings["approx_affine_transform"])

# %%
print("Loading data...")
with open_ome_zarr(source_path) as source:
    data_source_0 = source.data[t_idx, 0]

with open_ome_zarr(target_path) as target:
    target_channel_idx = target.channel_names.index(target_chahnel)
    data_target = target.data[t_idx, target_channel_idx]

# %%
print("Transforming data")
source_ants_0 = ants.from_numpy(data_source_0)
target_ants = ants.from_numpy(data_target)

tform_ants = convert_transform_to_ants(approx_tform)
source_pre_opt_ants_0 = tform_ants.apply_to_image(source_ants_0, reference=target_ants)
source_pre_opt_0 = source_pre_opt_ants_0.numpy()

# %% Try 2D registration on non-zero crop
# z_idx = 35
z_slice = slice(5, 85)
y_slice = slice(400, 1300)
x_slice = slice(200, -200)

source_pre_opt_0_cropped = source_pre_opt_0[z_slice, y_slice, x_slice]
target_cropped = data_target[z_slice, y_slice, x_slice]

print(np.sum(target_cropped == 0))
print(np.sum(source_pre_opt_0_cropped == 0))

# viewer = napari.Viewer()
# viewer.add_image(target_cropped, name="target", colormap="gray")
# viewer.add_image(source_pre_opt_1_cropped, name="source 1", colormap="green", blending="additive")
# viewer.add_image(source_pre_opt_0_cropped, name="source 0", colormap="magenta", blending="additive")

# %%
print("Filtering data")
source_0_edge_filter = filters.sobel(
    np.clip(source_pre_opt_0_cropped, 110, np.quantile(source_pre_opt_0_cropped, 0.99))
)
source_filter_sum = source_0_edge_filter

# viewer = napari.Viewer()
# viewer.add_image(source_pre_opt_0_cropped, name="source pre-opt 0", colormap="gray")
# viewer.add_image(source_pre_opt_1_cropped, name="source pre-opt 1", colormap="gray", blending="additive")
# viewer.add_image(source_0_edge_filter, name="source 0 edge", colormap="bop orange", blending="additive")
# viewer.add_image(source_1_edge_filter, name="source 1 edge", colormap="bop blue", blending="additive")
# viewer.add_image(source_0_edge_filter + source_1_edge_filter, name="edge sum", colormap="green", blending="additive")

# %%
target_edge_filter = filters.sobel(np.clip(target_cropped, 0, 0.5))

# viewer = napari.Viewer()
# # viewer.add_image(target_cropped, name="target", colormap="gray")
# # viewer.add_image(target_filtered, name="target filtered", colormap="gray")
# viewer.add_image(target_edge_filter, name="target edge", colormap="green", blending="additive")
# # viewer.add_image(target_filter_smooth, name="target filter smooth", colormap="magenta", blending="additive")
# viewer.add_image(source_0_edge_filter + source_1_edge_filter, name="edge sum", colormap="magenta", blending="additive")


# %%
# Select z-idx from source_pre_opt_ants - works well
_source = ants.from_numpy(source_filter_sum)
_target = ants.from_numpy(target_edge_filter)

reg = ants.registration(
    fixed=_target,
    moving=_source,
    type_of_transform="Similarity",
    aff_shrink_factors=(6, 3, 1),
    aff_iterations=(2100, 1200, 50),
    aff_smoothing_sigmas=(2, 1, 0),
    verbose=True,
)
tform_opt = ants.read_transform(reg["fwdtransforms"][0])

source_opt_ants = tform_opt.apply_to_image(_source, reference=_target).numpy()
source_ch0_opt = tform_opt.apply_to_image(
    ants.from_numpy(source_pre_opt_0_cropped), reference=_target
).numpy()

print("Display results in napari")
viewer = napari.Viewer()
viewer.add_image(target_cropped, name="target", colormap="gray")
viewer.add_image(source_ch0_opt, name="nuclei", colormap="bop blue", blending="additive")
# viewer.add_image(target_edge_filter, name="target edges", colormap="green", blending="additive")
# viewer.add_image(source_opt_ants, name="source edges", colormap="magenta", blending="additive")

# %%
tform_opt_np = convert_transform_to_numpy(tform_opt)
composed_matrix = approx_tform @ tform_opt_np

settings_path = registration_settings_path
in_model = yaml_to_model(settings_path, StabilizationSettings)

out_model = in_model.model_copy()
out_model.affine_transform_zyx_list[t_idx] = composed_matrix.tolist()

model_to_yaml(out_model, settings_path.with_name("sobel_registration.yml"))

# %%
