from pathlib import Path

import numpy as np
import torch
from biahub.deskew import deskew_zyx, fast_deskew_zyx, get_deskewed_data_shape
from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta
import time

LS_SCAN_ANGLE = 30.0
PX_SIZE_UM = 0.1133
SCAN_STEP = 0.15
PX_TO_SCAN_RATIO = PX_SIZE_UM / SCAN_STEP
KEEP_OVERHANG = True
AVERAGE_N_SLICES = 3

print("Loading data...")
data_path = Path("/tmp/fish-0_neuromast-0.zarr")
with open_ome_zarr(data_path) as ds:
    data = ds.data.numpy()[0, 0]

deskewed_data_shape, deskewed_voxel_size = get_deskewed_data_shape(
    raw_data_shape=data.shape,
    ls_angle_deg=LS_SCAN_ANGLE,
    px_to_scan_ratio=PX_TO_SCAN_RATIO,
    keep_overhang=KEEP_OVERHANG,
    average_n_slices=AVERAGE_N_SLICES,
    pixel_size_um=PX_SIZE_UM,
)

print("Deskewing data...")
start_time = time.time()
deskewed_data = deskew_zyx(
    raw_data=data,
    ls_angle_deg=LS_SCAN_ANGLE,
    px_to_scan_ratio=PX_TO_SCAN_RATIO,
    keep_overhang=KEEP_OVERHANG,
    average_n_slices=AVERAGE_N_SLICES,
    device='cuda'
)
end_time = time.time()
print(f"Deskewing completed in {end_time - start_time:.2f} seconds")

print(f"Running fast_deskew_zyx...")
start_time = time.time()
print(f"Transferring data to GPU...")
data_tensor = torch.from_numpy(data.astype(np.float32)).to('cuda')
t1 = time.time()
print(f"Data transfer completed in {t1 - start_time:.2f} seconds")
fast_deskewed_tensor = fast_deskew_zyx(
    raw_data=data_tensor,
    ls_angle_deg=LS_SCAN_ANGLE,
    px_to_scan_ratio=PX_TO_SCAN_RATIO,
    keep_overhang=KEEP_OVERHANG,
    average_n_slices=AVERAGE_N_SLICES,
)
torch.cuda.synchronize()
end_time = time.time()
fast_deskewed_data = fast_deskewed_tensor.cpu().numpy()
print(f"fast deskew completed in {end_time - t1:.2f} seconds (excluding data transfer)")
print(f"Total time for fast deskew (including data transfer): {end_time - start_time:.2f} seconds")

with open_ome_zarr(
    "/tmp/fish-0_neuromast-0_deskewed.ome.zarr",
    mode='w',
    layout='fov',
    channel_names=['deskewed'],
    version="0.5"
    ) as ds2:
    im = ds2.create_image(
        name="0",
        data=deskewed_data[None, None],
        chunks=(1, 1) + deskewed_data_shape,
        transform=[TransformationMeta(type="scale", scale=(1, 1)+deskewed_voxel_size)]
    )

with open_ome_zarr(
    "/tmp/fish-0_neuromast-0_fast_deskewed.ome.zarr",
    mode='w',
    layout='fov',
    channel_names=['fast_deskewed'],
    version="0.5"
    ) as ds3:
    im = ds3.create_image(
        name="0",
        data=fast_deskewed_data[None, None],
        chunks=(1, 1) + deskewed_data_shape,
        transform=[TransformationMeta(type="scale", scale=(1, 1)+deskewed_voxel_size)]
    )
