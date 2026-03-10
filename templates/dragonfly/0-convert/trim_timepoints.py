# %%
import json
import os
from pathlib import Path

from iohub import open_ome_zarr

num_timepoints = 33

dataset = os.environ.get("DATASET")
if dataset is None:
    print("$DATASET environmental variable is not set")

modalities = ["labelfree", "lightsheet"]

# %%
position_names = []
with open_ome_zarr(
    Path(f"{dataset}_symlink", f"{dataset}_{modalities[0]}_1.zarr")
) as ds:
    for pos_name, pos in ds.positions():
        position_names.append(pos_name)

# %%
for modality in modalities:
    for position in position_names:
        with open(
            Path(
                f"{dataset}_symlink",
                f"{dataset}_{modality}_1.zarr",
                position,
                "0/.zarray",
            ),
            mode="r+",
        ) as f:
            zarray = json.load(f)
            zarray["shape"] = [num_timepoints] + zarray["shape"][1:]
            f.seek(0)
            json.dump(zarray, f, indent=4)


# %%
