"""Normalize channel names on the assembled plate.

Implements the convention in czbiohub-sf/biahub#291. Run ONCE, after
``5-assemble`` completes, from the directory holding the store::

    cd <OUTPUT>/5-assemble
    DATASET=2026_07_14_A549_MAP4_ZIKV uv run --project <BIAHUB> python rename_channels.py

Mapping (first match wins):

===========================================  ==========================
incoming                                     renamed to
===========================================  ==========================
``BF - Oblique``                             ``BF``
``Phase3D*``                                 ``Phase3D``
``nuclei``                                   ``nuclei_prediction``
``membrane``                                 ``membrane_prediction``
already canonical, or already ``raw ``-fixed  left alone
anything else (e.g. ``mCherry EX561 ...``)   ``raw <original>``
===========================================  ==========================

The ``raw`` prefix marks unprocessed detector output, so raw and derived
channels are distinguishable at a glance. The rule is **idempotent** — the
passthrough branch means re-running never yields ``raw raw mCherry ...``.

``nuclei`` -> ``nuclei_prediction`` exists because ``biahub virtual-stain``
names its outputs verbatim from ``target_channel``, dropping the ``_prediction``
suffix that viscy's ``HCSPredictionWriter`` used to append (biahub#288), while
the rest of biahub still keys off the suffix.

This script is a stopgap: biahub#291 proposes doing the rename inside
``concatenate`` (it already resolves the output channel list, so the assembled
plate would be correct the first time), and PRs #260 / #250 carry a
``rename-channels`` CLI and Nextflow subworkflow. Neither has landed on main —
check before running this by hand.

ORDERING: this runs after the whole Nextflow pipeline, so ``4-track`` reads the
assembled plate under its PRE-rename channel names. Any channel a track config
references must exist under the un-renamed name at track time.
"""

import os
import sys

from fnmatch import fnmatch
from pathlib import Path

from iohub import open_ome_zarr

# fnmatch patterns -> canonical name. First match wins, in declaration order.
RENAMES = {
    "BF - Oblique": "BF",
    "Phase3D*": "Phase3D",
    "nuclei": "nuclei_prediction",
    "membrane": "membrane_prediction",
}

# Names that are already canonical and must be left untouched.
PASSTHROUGH = [
    "BF",
    "Phase3D",
    "nuclei_prediction",
    "membrane_prediction",
]

# Prefix applied to every channel matching neither RENAMES nor PASSTHROUGH
# (the raw detector channels). Set to None to leave them alone.
UNMATCHED_PREFIX = "raw "


def canonical_name(channel_name: str) -> str | None:
    """Return the canonical name for a channel, or None to leave it unchanged.

    Parameters
    ----------
    channel_name : str
        The channel name as it appears in the assembled store.

    Returns
    -------
    str or None
        The new name, or None if the channel is already canonical.
    """
    for pattern, new_name in RENAMES.items():
        if fnmatch(channel_name, pattern):
            return None if new_name == channel_name else new_name
    if channel_name in PASSTHROUGH:
        return None
    if UNMATCHED_PREFIX and not channel_name.startswith(UNMATCHED_PREFIX):
        return f"{UNMATCHED_PREFIX}{channel_name}"
    return None


def main() -> None:
    """Rename every channel of the assembled plate named by ``$DATASET``."""
    dataset = os.environ.get("DATASET")
    if dataset is None:
        sys.exit("DATASET environment variable is not set")

    dataset_path = Path(f"{dataset}.zarr")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_path} does not exist.")

    with open_ome_zarr(dataset_path, mode="a") as ds:
        print(f"channels before: {ds.channel_names}")
        plan = {c: canonical_name(c) for c in ds.channel_names}
        plan = {old: new for old, new in plan.items() if new is not None}

        if not plan:
            print("all channels already canonical -- nothing to do")
            return
        for old, new in plan.items():
            print(f"  '{old}' -> '{new}'")

        for pos_name, pos in ds.positions():
            print(f"Processing position: {pos_name}")
            for old, new in plan.items():
                pos.rename_channel(old, new)

    with open_ome_zarr(dataset_path, mode="r") as ds:
        print(f"channels after: {ds.channel_names}")


if __name__ == "__main__":
    main()
