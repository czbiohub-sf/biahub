"""Rename raw instrument channel labels on the assembled plate.

Run ONCE, after 5-assemble completes, from the directory holding the store:

    cd <OUTPUT>/5-assemble
    DATASET=2026_07_14_A549_MAP4_ZIKV uv run --project <BIAHUB> python rename_channels.py

`BF - Oblique` becomes `BF`, and each acquired fluorescence channel gets a
`raw ` prefix so it is distinguishable from the virtual-stain predictions
(`nuclei_prediction`, `membrane_prediction`), which are left untouched.

CHECK THE CHANNEL LIST FIRST — it is dataset-dependent. Print the store's
channel names and extend `RAW_CHANNELS` before running; a channel that is not
listed here is silently left alone.

NOT IDEMPOTENT in the prefix direction: running twice yields
`raw raw mCherry ...`. Inspect the current names before a re-run.

Adapted from
/hpc/projects/intracellular_dashboard/organelle_dynamics/
  2026_05_27_A549_SEC61B_TOMM20_G3BP1_ZIKV/1-preprocess/5-assemble/rename_channels.py
"""

import os
import sys

from pathlib import Path

from iohub import open_ome_zarr

# Channels to prefix with 'raw'. Extend per dataset.
RAW_CHANNELS = [
    "mCherry EX561 EM600-37",
    "GFP EX488 EM525-45",
    "Cy5 EX639 EM698-70",
]

# Exact renames.
RENAMES = {
    "BF - Oblique": "BF",
}

dataset = os.environ.get("DATASET")
if dataset is None:
    sys.exit("DATASET environment variable is not set")

dataset_path = Path(f"{dataset}.zarr")
if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset {dataset_path} does not exist.")

with open_ome_zarr(dataset_path, mode="a") as ds:
    print(f"channels before: {ds.channel_names}")

    already_prefixed = [c for c in ds.channel_names if c.startswith("raw ")]
    if already_prefixed:
        sys.exit(
            f"Channels already carry a 'raw ' prefix ({already_prefixed}) — "
            "this store looks renamed. Refusing to double-prefix."
        )

    for pos_name, pos in ds.positions():
        print(f"Processing position: {pos_name}")
        for channel_name in ds.channel_names:
            if channel_name in RENAMES:
                new_name = RENAMES[channel_name]
            elif channel_name in RAW_CHANNELS:
                new_name = f"raw {channel_name}"
            else:
                continue
            print(f"  '{channel_name}' -> '{new_name}'")
            pos.rename_channel(channel_name, new_name)

with open_ome_zarr(dataset_path, mode="r") as ds:
    print(f"channels after: {ds.channel_names}")
