"""Generate template annotation CSVs for a dataset before running the pipeline.

Discovers all FOVs in the zarr, reads T from each, and creates:
  1. <output_dir>/annotations.csv — one row per FOV with columns:
       dataset, fov, exclude, beads, comments
  2. <output_dir>/per_fov/<fov_name>/annotation.csv — one row per timepoint:
       t, exclude, comments

The user fills these in (e.g. mark exclude=1 for bad FOVs or timepoints),
then passes the annotations directory to the pipeline.

Usage:
    python dynacell_init_annotations.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from iohub import open_ome_zarr


def init_annotations(
    root_path: Path,
    dataset: str,
    output_dir: Path | None = None,
    beads_fov: str | None = None,
    lf_zarr_override: Path | None = None,
):
    """Create template annotation CSVs for all FOVs in the dataset.

    Parameters
    ----------
    root_path : Path
        Root path for organelle_dynamics datasets.
    dataset : str
        Dataset name.
    output_dir : Path or None
        Where to write annotations. Defaults to <root_path>/<dataset>/dynacell/annotations/.
    beads_fov : str or None
        FOV key for beads (e.g. "C/1/000000"). Will be pre-filled with beads=1.
    lf_zarr_override : Path or None
        Override path for the LF zarr.
    """
    # Resolve zarr path
    if lf_zarr_override is not None:
        lf_zarr = lf_zarr_override
    else:
        lf_zarr = (
            root_path / dataset / "1-preprocess" / "label-free"
            / "0-reconstruct" / f"{dataset}.zarr"
        )

    # Discover FOVs
    position_dirpaths = sorted(Path(p) for p in glob(str(lf_zarr / "*" / "*" / "*")))
    position_keys = ["/".join(p.parts[-3:]) for p in position_dirpaths]
    print(f"Found {len(position_keys)} FOVs in {lf_zarr}")

    # Output directory
    if output_dir is None:
        output_dir = root_path / dataset / "dynacell" / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_fov_dir = output_dir / "per_fov"
    per_fov_dir.mkdir(parents=True, exist_ok=True)

    # Read T from each FOV and generate per-FOV CSVs
    dataset_rows = []
    for fov_key in position_keys:
        fov_name = "_".join(fov_key.split("/"))
        fov_path = lf_zarr / fov_key

        with open_ome_zarr(fov_path) as ds:
            T = ds.data.shape[0]

        # Per-FOV annotation: one row per timepoint
        fov_out_dir = per_fov_dir / fov_name
        fov_out_dir.mkdir(parents=True, exist_ok=True)
        fov_annot_path = fov_out_dir / "annotation.csv"

        if not fov_annot_path.exists():
            fov_df = pd.DataFrame({
                "t": np.arange(T),
                "exclude": 0,
                "comments": "",
            })
            with open(fov_annot_path, "w") as f:
                f.write(f"# dataset: {dataset}\n# fov: {fov_key}\n")
                fov_df.to_csv(f, index=False)
            print(f"  Created {fov_annot_path} ({T} timepoints)")
        else:
            print(f"  Skipped {fov_annot_path} (already exists)")

        # Dataset-level row
        is_beads = 1 if beads_fov and fov_key == beads_fov else 0
        dataset_rows.append({
            "dataset": dataset,
            "fov": fov_key,
            "T": T,
            "exclude": 0,
            "beads": is_beads,
            "comments": "beads" if is_beads else "",
        })

    # Dataset-level annotation CSV
    ds_annot_path = output_dir / "annotations.csv"
    if not ds_annot_path.exists():
        ds_df = pd.DataFrame(dataset_rows)
        ds_df.to_csv(ds_annot_path, index=False)
        print(f"\nDataset annotations: {ds_annot_path} ({len(dataset_rows)} FOVs)")
    else:
        print(f"\nSkipped {ds_annot_path} (already exists)")

    print(f"\nEdit the CSVs to mark FOVs/timepoints to exclude, then run the pipeline.")
    return output_dir


if __name__ == "__main__":
    root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")

    # 2024_11_05_A549_TOMM20_ZIKV_DENV
    init_annotations(
        root_path=root_path,
        dataset="2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV",
        beads_fov="A/3/000000",
    )
