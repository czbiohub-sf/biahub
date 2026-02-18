import os
from pathlib import Path

from iohub import open_ome_zarr

dataset = os.environ.get("DATASET")
if dataset is None:
    print("DATASET environmental variable is not set")
    exit()
dataset_path = f"{dataset}.zarr"
# Check if the dataset exists
if not Path(dataset_path).exists():
    raise FileNotFoundError(f"Dataset {dataset_path} does not exist.")

# Open the dataset and rename channels
with open_ome_zarr(dataset_path, mode="a") as ds:
    for pos_name, pos in ds.positions():
        print(f"Processing position: {pos_name}")
        for channel_name in ds.channel_names:
            if "raw" in channel_name:
                print(f"Channel '{channel_name}' already has 'raw' prefix. Skipping.")
                continue
            # Rename the channel by prefixing it with 'raw'
            new_channel_name = f"raw {channel_name}"
            print(f"Renaming channel '{channel_name}' to '{new_channel_name}'")
            pos.rename_channel(channel_name, new_channel_name)
