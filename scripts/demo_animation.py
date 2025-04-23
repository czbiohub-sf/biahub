#%%
import napari
import numpy as np
from pathlib import Path
from biahub.visualize.animation_utils import (
    add_scale_bar,
    add_text_overlay,
    ElementPosition,
    simple_recording,
)
import os
np.random.seed(42)

os.environ['DISPLAY'] = ':1'

# Create a random 4D image (T, Z, Y, X)
data = np.random.randint(0, 65535, size=(10, 20, 512, 512), dtype=np.uint16)
data = data.astype(np.float32) / 65535.0  # Normalize to [0, 1]

# Create viewer and add the image
viewer = napari.Viewer()
layer = viewer.add_image(
    data,
    name='random_image',
    scale=(5.0, 0.25, 0.108, 0.108),  # (T:min, Z:µm, Y:µm, X:µm)
    contrast_limits=(0, 1),
)

# Add a 10µm scale bar
add_scale_bar(
    viewer,
    scale_bar_length=10,  # 10 µm
    position=ElementPosition.BOTTOM_RIGHT,
    line_width=5,
    text_size=20,
    color='white'
)

# Add time and z position overlay using axis indices
add_text_overlay(
    viewer,
    time_axis=0,  # First dimension is time
    z_axis=1,     # Second dimension is Z
    position=ElementPosition.TOP_LEFT,
    text_size=30,
    color='black'
)

# Record animation looping through T and Z axes
# Each axis will take 3 seconds
simple_recording(
    viewer,
    output_path=Path('animation.mp4'),
    loop_axes=[
        (0, (None, None), 5),  # Time axis
        (1, (None, None), 3),  # Z axis
    ],
    fps=30
)

napari.run() 
# %%
