import numpy as np
from iohub import open_ome_zarr
from pathlib import Path

def find_stable_crop(path1, path2, intensity_threshold=0.05, margin=10):
    """Find the largest stable crop for two OME-Zarr datasets."""
    # Open datasets
    dataset1 = open_ome_zarr(path1)
    dataset2 = open_ome_zarr(path2)

    # Assume first channel
    data1 = dataset1.data[:, 0]  # (T, Z, Y, X)
    data2 = dataset2.data[:, 0]

    # Load small subset to memory
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    T, Z, Y, X = data1.shape

    # Make masks: pixels above threshold
    mask1 = data1 > (intensity_threshold * np.max(data1, axis=(2, 3), keepdims=True))
    mask2 = data2 > (intensity_threshold * np.max(data2, axis=(2, 3), keepdims=True))

    # Combine masks across time and Z
    valid_mask = np.all(mask1 & mask2, axis=(0, 1))  # (Y, X)

    # Find bounding box
    ys, xs = np.where(valid_mask)
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("No stable overlap found.")

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Add margins
    y_min = max(y_min - margin, 0)
    y_max = min(y_max + margin, Y - 1)
    x_min = max(x_min - margin, 0)
    x_max = min(x_max + margin, X - 1)

    # Print the slices
    print(f"Recommended Y_slice: [{y_min}, {y_max}]")
    print(f"Recommended X_slice: [{x_min}, {x_max}]")

    return [y_min, y_max], [x_min, x_max]

# Example usage:
path1 = Path("/path/to/label_free.zarr/0/0/0")
path2 = Path("/path/to/light_sheet.zarr/0/0/0")

Y_slice, X_slice = find_stable_crop(path1, path2)
