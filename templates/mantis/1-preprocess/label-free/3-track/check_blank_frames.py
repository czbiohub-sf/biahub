import os
from pathlib import Path
import numpy as np
import pandas as pd
from natsort import natsorted
import glob
from iohub import open_ome_zarr
from tqdm import tqdm

dataset = os.environ.get("DATASET")
if dataset is None:
    print("DATASET environmental variable is not set")
    exit()

def _check_nan_n_zeros(input_array):
    """
    Checks if an array is entirely zeros or NaNs and returns indices of such slices.

    Args:
        input_array: Input array (2D, 3D, or 4D).

    Returns:
        List[Tuple[int, List[int]]]:
            - For 2D arrays: Returns True if the array is zeros or NaNs, False otherwise.
            - For 3D arrays: Returns a list of Z indices that are entirely zeros or NaNs.
            - For 4D arrays: Returns a list of tuples, where each tuple contains:
              (channel_index, list_of_empty_z_indices).
    """
    indices = []

    if len(input_array.shape) == 2:  # 2D array (e.g., Y, X)
        # Return True if entirely zeros or NaNs
        return np.all(input_array == 0) or np.all(np.isnan(input_array))

    elif len(input_array.shape) == 3:  # 3D array (e.g., Z, Y, X)
        for z in range(input_array.shape[0]):
            if np.all(input_array[z, :, :] == 0) or np.all(np.isnan(input_array[z, :, :])):
                indices.append(z)  # Add Z index if it's empty
        return indices

    elif len(input_array.shape) == 4:  # 4D array (e.g., C, Z, Y, X)
        for c in range(input_array.shape[0]):  # Iterate over channels
            z_indices = _check_nan_n_zeros(
                input_array[c, :, :, :]
            )  # Check Z slices in each channel
            if z_indices:  # If there are empty Z slices, add them
                indices.append((c, z_indices))  # Add (channel, empty_z_indices)
        return indices

    else:
        raise ValueError("Input array must be 2D, 3D, or 4D.")


def main():
    im_path = f"../2-stabilize/phase/{dataset}.zarr/"

    input_position_dirpaths = [Path(p) for p in natsorted(glob.glob(f"{im_path}/*/*/*"))]
    im_ds = open_ome_zarr(Path(im_path))
    frames = {}
    for position_dirpath in tqdm(input_position_dirpaths):
        position = position_dirpath.parts[-3:]
        fov = f"{position[0]}/{position[1]}/{position[2]}"

        print(f"Processing FOV: {fov}")
        
        T, C, Z, Y, X = im_ds[fov].data.shape
        im_arr = im_ds[fov].data[:, 0, int(Z/2), :, :]
        empty_frames_idx = _check_nan_n_zeros(im_arr)

        print("Empty frames:", empty_frames_idx)
        if empty_frames_idx==[]:
            frames[f"{fov}"] = {"Count": 0, "t": 0}
        else:
            frames[f"{fov}"] = {"Count": len(empty_frames_idx), "t": empty_frames_idx}


    df = pd.DataFrame.from_dict(frames, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "FOV"}, inplace=True)
    df.sort_values(by=["FOV"], inplace=True)
    df.to_csv("blank_frames.csv")
    print(df)


if __name__ == "__main__":
    main()
