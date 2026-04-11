"""Re-crop dynacell plate with manual frame/FOV exclusions.

Workflow:
1. Run the main pipeline with drop_frames=False to get a full plate
2. Visually inspect in napari
3. Create a manual_drops.csv:
   - Rows with 'fov' only: exclude entire FOV
   - Rows with 't' only: drop timepoint from all FOVs
4. Run this script to produce a cleaned plate

CSV format (manual_drops.csv):
    fov,t,reason
    A/1/000001,,bad_sample
    ,5,blurry
    ,23,artifact

Usage:
    python dynacell_manual_crop.py \\
        --run-dir /path/to/run_YYYYMMDD_HHMMSS \\
        --manual-drops /path/to/manual_drops.csv
"""

import argparse
import yaml
import time
import numpy as np
import pandas as pd
import submitit
from datetime import datetime
from pathlib import Path
from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate

from dynacell.stage2 import crop_fov


def _resolve_zarr_paths(root_path, dataset, params):
    """Resolve zarr paths from run log parameters."""
    lf_override = params.get("lf_zarr_override")
    if lf_override:
        lf_zarr = Path(lf_override)
    else:
        lf_zarr = root_path / dataset / "1-preprocess" / "label-free" / "0-reconstruct" / f"{dataset}.zarr"
    ls_zarr = root_path / dataset / "1-preprocess" / "light-sheet" / "raw" / "1-register" / f"{dataset}.zarr"
    ls_deconvolved_zarr = root_path / dataset / "1-preprocess" / "light-sheet" / "deconvolved" / "2-register" / f"{dataset}.zarr"
    bf_zarr = root_path / dataset / "0-convert" / f"{dataset}_symlink" / f"{dataset}_labelfree_1.zarr"
    return [lf_zarr, ls_zarr, ls_deconvolved_zarr, bf_zarr]


def parse_manual_drops(csv_path):
    """Parse manual drops CSV into excluded FOVs and dropped timepoints.

    Returns (exclude_fovs: set[str], drop_timepoints: set[int]).
    """
    df = pd.read_csv(csv_path)
    exclude_fovs = set()
    drop_timepoints = set()

    for _, row in df.iterrows():
        fov_val = row.get("fov", "")
        t_val = row.get("t", "")
        has_fov = pd.notna(fov_val) and str(fov_val).strip() != ""
        has_t = pd.notna(t_val) and str(t_val).strip() != ""

        if has_fov and not has_t:
            exclude_fovs.add(str(fov_val).strip())
        if has_t:
            drop_timepoints.add(int(float(t_val)))

    return exclude_fovs, drop_timepoints


def manual_crop(
    run_dir: Path,
    manual_drops_csv: Path,
    output_dir: Path | None = None,
    include_auto_drops: bool = False,
    local: bool = True,
):
    """Re-crop from source zarrs with manual frame/FOV exclusions.

    Parameters
    ----------
    run_dir : Path
        Existing run directory with stage 1 results.
    manual_drops_csv : Path
        CSV with columns: fov, t, reason.
        Rows with fov only → exclude entire FOV.
        Rows with t → drop timepoint from all FOVs.
    output_dir : Path or None
        Output directory. If None, creates a new run dir.
    include_auto_drops : bool
        If True, also apply automatic drops from stage 1's drop_list_all_fovs.csv.
    local : bool
        If True, run locally. If False, use SLURM.
    """
    # Read run log for original paths and parameters
    with open(run_dir / "run_log.yaml") as f:
        run_log = yaml.safe_load(f)
    params = run_log["parameters"]

    root_path = Path(params["root_path"])
    dataset = params["dataset"]
    z_final = params["z_final"]

    all_zarrs = _resolve_zarr_paths(root_path, dataset, params)
    plots_dir = run_dir / "per_fov_analysis"

    # Read stage 1 results
    summary_df = pd.read_csv(run_dir / "global_summary.csv")
    all_fovs = summary_df["fov"].tolist()

    # Parse manual drops
    exclude_fovs, drop_timepoints = parse_manual_drops(manual_drops_csv)
    print(f"Manual drops: {len(exclude_fovs)} FOVs excluded, {len(drop_timepoints)} timepoints dropped")
    if exclude_fovs:
        print(f"  Excluded FOVs: {sorted(exclude_fovs)}")
    if drop_timepoints:
        print(f"  Dropped timepoints: {sorted(drop_timepoints)}")

    # Optionally include auto drops from stage 1
    if include_auto_drops:
        auto_drop_csv = run_dir / "drop_list_all_fovs.csv"
        if auto_drop_csv.exists() and auto_drop_csv.stat().st_size > 0:
            auto_df = pd.read_csv(auto_drop_csv)
            auto_ts = set(int(t) for t in auto_df["t"].unique())
            n_new = len(auto_ts - drop_timepoints)
            drop_timepoints.update(auto_ts)
            print(f"  + auto drops from stage 1: {len(auto_ts)} timepoints ({n_new} new)")

    # Filter FOVs (support both "A/1/000001" and "A_1_000001" formats)
    exclude_normalized = set()
    for f in exclude_fovs:
        exclude_normalized.add(f)
        exclude_normalized.add("_".join(f.split("/")))
        exclude_normalized.add("/".join(f.split("_")))

    kept_fovs = [f for f in all_fovs if f not in exclude_normalized
                 and "_".join(f.split("/")) not in exclude_normalized]

    if not kept_fovs:
        print("ERROR: No FOVs remaining after manual exclusions")
        return

    # Get T_total from first FOV's z_focus
    first_fov_name = "_".join(kept_fovs[0].split("/"))
    z_focus_df = pd.read_csv(plots_dir / first_fov_name / "z_focus.csv")
    T_total = len(z_focus_df)

    # Compute kept timepoints and plate dims
    keep_indices = sorted(set(range(T_total)) - drop_timepoints)
    T_out = len(keep_indices)
    kept_summary = summary_df[summary_df["fov"].isin(kept_fovs)]
    Y_out = int(kept_summary["Y_crop"].min())
    X_out = int(kept_summary["X_crop"].min())

    # Get channel names and scale from first FOV
    all_channel_names = []
    scale = None
    for zarr_path in all_zarrs:
        with open_ome_zarr(zarr_path / kept_fovs[0]) as ds:
            all_channel_names.extend(list(ds.channel_names))
            if scale is None:
                scale = list(ds.scale)

    C_out = len(all_channel_names)
    Z_out = z_final

    print(f"\n=== Manual crop summary ===")
    print(f"  Source run: {run_dir}")
    print(f"  Kept FOVs: {len(kept_fovs)} / {len(all_fovs)}")
    print(f"  Kept timepoints: {T_out} / {T_total}")
    print(f"  Dropped timepoints: {sorted(drop_timepoints)}")
    print(f"  Output shape: ({T_out}, {C_out}, {Z_out}, {Y_out}, {X_out})")

    # Setup output directory
    if output_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = root_path / dataset / "dynacell" / f"run_{run_id}_manual"
    output_zarr = output_dir / f"{dataset}.zarr"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Write the combined drop list CSV for crop_fov to read
    drop_rows = []
    for t in sorted(drop_timepoints):
        drop_rows.append({"fov": "manual", "t": int(t), "reason": "manual_drop"})
    if include_auto_drops:
        auto_drop_csv = run_dir / "drop_list_all_fovs.csv"
        if auto_drop_csv.exists() and auto_drop_csv.stat().st_size > 0:
            auto_df = pd.read_csv(auto_drop_csv)
            for _, row in auto_df.iterrows():
                drop_rows.append({
                    "fov": row["fov"], "t": int(row["t"]), "reason": row["reason"],
                })
    combined_drop_csv = output_dir / "drop_list_all_fovs.csv"
    pd.DataFrame(drop_rows, columns=["fov", "t", "reason"]).to_csv(
        combined_drop_csv, index=False
    )
    print(f"Saved combined drop list ({len(drop_rows)} entries) to {combined_drop_csv}")

    # Save crop log
    crop_log = {
        "timestamp": datetime.now().isoformat(),
        "source_run": str(run_dir),
        "manual_drops_csv": str(manual_drops_csv),
        "include_auto_drops": include_auto_drops,
        "excluded_fovs": sorted(exclude_fovs),
        "dropped_timepoints": sorted(drop_timepoints),
        "kept_fovs": kept_fovs,
        "T_out": T_out,
        "T_total": T_total,
        "Y_out": Y_out,
        "X_out": X_out,
    }
    with open(output_dir / "manual_crop_log.yaml", "w") as f:
        yaml.dump(crop_log, f, default_flow_style=False, sort_keys=False)

    # Create output plate
    ok_position_keys = [tuple(fov.split("/")) for fov in kept_fovs]
    create_empty_plate(
        store_path=output_zarr,
        position_keys=ok_position_keys,
        shape=(T_out, C_out, Z_out, Y_out, X_out),
        chunks=(1, 1, Z_out, Y_out, X_out),
        scale=scale,
        channel_names=all_channel_names,
        dtype=np.float32,
        version='0.5',
    )
    print(f"Created output plate at {output_zarr}")

    # Submit crop jobs
    _start = time.time()
    cluster = "local" if local else "slurm"
    slurm_out_path = output_dir / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    # Estimate resources
    total_elements = 0
    for zarr_path in all_zarrs:
        with open_ome_zarr(zarr_path / kept_fovs[0]) as ds:
            _, C_i, Z_i, Y_i, X_i = ds.data.shape
            total_elements += C_i * Z_i * Y_i * X_i
    gb_per_t = total_elements * np.dtype(np.float32).itemsize / 1e9
    gb_ram_per_cpu = max(4, int(np.ceil(gb_per_t * 4)))

    slurm_args = {
        "slurm_job_name": "dynacell_manual_crop",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": 4,
        "slurm_array_parallelism": 100,
        "slurm_time": 120,
        "slurm_partition": "gpu",
    }

    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for fov in kept_fovs:
            fov_name = "_".join(fov.split("/"))
            input_zarr_paths = [z / fov for z in all_zarrs]

            job = executor.submit(
                crop_fov,
                input_zarr_paths=input_zarr_paths,
                output_zarr=output_zarr,
                output_plots_dir=plots_dir / fov_name,
                fov=fov,
                global_drop_csv=combined_drop_csv,
                z_final=z_final,
                T_out=T_out,
                Y_out=Y_out,
                X_out=X_out,
                drop_frames=True,
            )
            jobs.append(job)

    print(f"Submitted {len(jobs)} crop jobs ({cluster})")
    print("Waiting for completion...")
    for job in jobs:
        job.result()

    elapsed = time.time() - _start
    print(f"\nDone! Output: {output_zarr}")
    print(f"Time: {int(elapsed // 60)}m {int(elapsed % 60)}s")


if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Re-crop dynacell plate with manual frame/FOV exclusions.",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# CSV format (manual_drops.csv):
#     fov,t,reason
#     A/1/000001,,bad_sample       # exclude entire FOV
#     ,5,blurry                    # drop timepoint 5 from all FOVs
#     ,23,artifact                 # drop timepoint 23 from all FOVs
# """,
#     )
#     parser.add_argument(
#         "--run-dir", type=Path, required=True,
#         help="Existing run directory with stage 1 results",
#     )
#     parser.add_argument(
#         "--manual-drops", type=Path, required=True,
#         help="CSV with columns: fov, t, reason",
#     )
#     parser.add_argument(
#         "--output-dir", type=Path, default=None,
#         help="Output directory (default: auto-generated)",
#     )
#     parser.add_argument(
#         "--include-auto-drops", action="store_true",
#         help="Also apply automatic drops from stage 1",
#     )
#     parser.add_argument(
#         "--slurm", action="store_true",
#         help="Run on SLURM instead of locally",
#     )
#     args = parser.parse_args()

    run_dir = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/dynacell/run_20260320_111830")
    manual_drops_csv = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/manual_drops.csv")
    output_dir = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/dynacell/run_20260320_111830_manual")
    include_auto_drops = False
    local = False
    manual_crop(
        run_dir=run_dir,
        manual_drops_csv=manual_drops_csv,
        output_dir=output_dir,
        include_auto_drops=include_auto_drops,
        local=local,
    )
