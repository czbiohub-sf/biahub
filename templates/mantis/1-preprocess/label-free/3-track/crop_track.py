# %%
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from biahub.cli.utils import create_empty_hcs_zarr
from iohub import open_ome_zarr
from tqdm import tqdm

dataset = os.environ.get("DATASET")
if dataset is None:
    print("DATASET environmental variable is not set")
    exit()


def main():

    track_path = Path(f"{dataset}.zarr")

    blank_frames_path = f"blank_frames.csv"
    blank_frames_df = pd.read_csv(blank_frames_path, index_col=0)
    keys = blank_frames_df["FOV"].to_list()

    path_concatenate_crop = Path("../../../2-assemble/concatenate_cropped.yml")
    with open(path_concatenate_crop, "r") as file:
        config = yaml.safe_load(file)  # Load YAML file

    x_slices_list = config["X_slice"]
    y_slices_list = config["Y_slice"]
    x_slices = slice(x_slices_list[0], x_slices_list[1])
    y_slices = slice(y_slices_list[0], y_slices_list[1])

    track_seg = open_ome_zarr(track_path)

    positions = blank_frames_df["FOV"].to_list()
    positions_temp = []
    for position in positions:
        positions_temp.append(tuple(position.split("/")))
    position_list = tuple(positions_temp)

    # # Store the output in OME-Zarr
    store_path = f"{dataset}_cropped.zarr"
    print("Zarr store path", store_path)

    key = keys[0]
    segments = track_seg[key].data[:, 0, 0, :, :]

    cropped_segments = segments[:, y_slices, x_slices]
    T, Y, X = cropped_segments.shape

    channel_names = track_seg.channel_names
    print(f"Channel names: {channel_names}")

    processing_channels = [f"{channel_names[0]}_labels"]
    output_metadata = {
        "shape": (T, len(processing_channels), 1, Y, X),
        "chunks": None,
        "scale": track_seg[key].scale,
        "channel_names": processing_channels,
        "dtype": np.uint32,
    }

    create_empty_hcs_zarr(
        store_path=store_path, position_keys=position_list, **output_metadata
    )

    for key in tqdm(keys):
        w1, w2, f = key.split("/")
        empty_frames_idx = blank_frames_df.loc[
            blank_frames_df["FOV"] == key, "t"
        ].values[0]

        print(empty_frames_idx)
        empty_frames_idx = blank_frames_df.loc[blank_frames_df["FOV"] == key, "t"].values[0]

        # Ensure `empty_frames_idx` is a list of integers
        if isinstance(empty_frames_idx, str):
            empty_frames_idx = empty_frames_idx.strip("[]").replace("'", "").split(",")
            empty_frames_idx = [int(i.strip()) for i in empty_frames_idx if i.strip()]
        elif isinstance(empty_frames_idx, (int, np.integer)):
            empty_frames_idx = [int(empty_frames_idx)]
        elif isinstance(empty_frames_idx, list):
            empty_frames_idx = [int(i) for i in empty_frames_idx]
        else:
            raise TypeError(f"Unexpected type for empty_frames_idx: {type(empty_frames_idx)}")

        empty_frames_idx.sort()
        print(empty_frames_idx)
        key = f"{w1}/{w2}/{f}"
        print(key)

        segments = track_seg[key].data[:, 0, 0, :, :]
        cropped_segments = segments[:, y_slices, x_slices]
        df_path = track_path / f"{w1}" / f"{w2}" / f"{f}" / f"tracks_{w1}_{w2}_{f}.csv"
        track_df = pd.read_csv(df_path)
        # remove tracks that are not in the cropped segments
        # drop tracks that x and y are not in the cropped segments
        # list of track ids that are not in the cropped segments

        track_df = track_df[track_df["x"] > x_slices.start]
        track_df = track_df[track_df["x"] < x_slices.stop]
        track_df = track_df[track_df["y"] > y_slices.start]
        track_df = track_df[track_df["y"] < y_slices.stop]

        # shift the x and y coordinates
        track_df["x"] = track_df["x"] - x_slices.start
        track_df["y"] = track_df["y"] - y_slices.start
        # remove tracks with t in empty_frames_idx
        if len(empty_frames_idx) > 0 and empty_frames_idx[0] != 0:
            print("Removing tracks with t in empty frames")
            track_df = track_df[~track_df["t"].isin(empty_frames_idx)]

        # if parent track not in track if, change the value to -1
        track_df["parent_track_id"] = track_df["parent_track_id"].apply(
            lambda x: x if x in track_df["track_id"].values else -1
        )
        # #save the cropped tracks
        csv_path = store_path / Path(key) / f"tracks_{w1}_{w2}_{f}.csv"
        track_df.to_csv(csv_path, index=False)

        # Save the cropped tracks
        with open_ome_zarr(store_path / Path(key), mode="r+") as output_dataset:
            output_dataset[0][:, 0, 0] = np.array(cropped_segments)

            print(f"Saved tracks and labels to: {store_path / Path(key)}")


if __name__ == "__main__":
    main()
