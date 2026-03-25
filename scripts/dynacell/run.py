from datetime import datetime
import yaml
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import submitit
from glob import glob
from pathlib import Path
from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from dynacell.qc_report import generate_dataset_report

from dynacell.geometry import (
    NA_DET,
    LAMBDA_ILL,
    make_circular_mask,
    find_overlap_mask,
    find_inscribed_bbox,
    find_overlap_bbox_across_time,
)
from dynacell.plotting import plot_bbox_over_time, plot_overlap, plot_z_focus
from dynacell.qc import (
    compute_beads_registration_qc,
    compute_laplacian_qc,
    compute_entropy_qc,
    compute_dust_qc,
    compute_bleach_qc,
)
from dynacell.stage1 import (
    find_blank_frames,
    build_drop_list,
    compute_per_timepoint_metadata,
    compute_fov_metadata,
    compute_fov_core,
    run_fov_qc,
    QC_METRICS,
)
from dynacell.stage2 import crop_fov


## ===== Pipeline helpers =====

def _get_git_info(repo_path: str | Path | None = None) -> dict:
    """Return current git branch, commit hash, and dirty status for a repo."""
    cwd = str(repo_path) if repo_path else None
    info = {}
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, text=True
        ).strip()
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True
        ).strip()
        info["dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd, text=True
            ).strip()
        )
    except Exception as e:
        info["error"] = str(e)
    return info


def _resolve_zarr_paths(root_path: Path, dataset: str, lf_zarr_override: Path | None = None):
    """Resolve all input zarr paths for a dataset."""
    if lf_zarr_override is not None:
        lf_zarr = lf_zarr_override
    else:
        lf_zarr = root_path / dataset / "1-preprocess" / "label-free" / "0-reconstruct" / f"{dataset}.zarr"
    ls_zarr = root_path / dataset / "1-preprocess" / "light-sheet" / "raw" / "1-register" / f"{dataset}.zarr"
    ls_deconvolved_zarr = root_path / dataset / "1-preprocess" / "light-sheet" / "deconvolved" / "2-register" / f"{dataset}.zarr"
    bf_zarr = root_path / dataset / "0-convert" / f"{dataset}_symlink" / f"{dataset}_labelfree_1.zarr"
    # vs_zarr = root_path / dataset / "1-preprocess" / "label-free" / "1-virtual-stain" / f"{dataset}.zarr"
    return lf_zarr, ls_zarr, ls_deconvolved_zarr, bf_zarr


def _discover_and_filter_fovs(lf_zarr, include_fovs, exclude_fovs, beads_fov):
    """Discover FOV positions in the zarr and apply include/exclude filters."""
    position_dirpaths = sorted([Path(p) for p in glob(str(lf_zarr / "*" / "*" / "*"))])
    position_keys = [p.parts[-3:] for p in position_dirpaths]
    print(f"Found {len(position_keys)} FOVs")

    if include_fovs:
        include_set = {tuple(f.split("/")) for f in include_fovs}
        position_keys = [k for k in position_keys if k in include_set]
        print(f"  Filtered to {len(position_keys)} FOVs by include_fovs")

    if exclude_fovs:
        exclude_set = {tuple(f.split("/")) for f in exclude_fovs}
        excluded = [k for k in position_keys if k in exclude_set]
        position_keys = [k for k in position_keys if k not in exclude_set]
        if excluded:
            print(f"  Excluded by user: {len(excluded)} FOVs: {['/'.join(k) for k in excluded]}")

    if beads_fov is not None:
        beads_key = tuple(beads_fov.split("/"))
        # Check that beads FOV exists in the original (unfiltered) position list
        all_keys_set = {p.parts[-3:] for p in sorted(Path(lf_zarr).glob("*/*/*"))}
        if beads_key not in all_keys_set:
            print(f"WARNING: beads_fov '{beads_fov}' not found in dataset, skipping beads QC")
            beads_fov = None
        else:
            # Remove beads FOV from processing list — it's only used for registration QC
            position_keys = [k for k in position_keys if k != beads_key]
            print(f"  Beads FOV: {beads_fov} (registration QC only, excluded from processing)")
    else:
        print("  No beads_fov specified, skipping registration QC")

    return position_keys, beads_fov


def _get_plate_metadata(all_zarrs, first_fov):
    """Read channel names, scale, and estimate SLURM resources from the first FOV."""
    all_channel_names = []
    total_elements_per_t = 0
    scale = None

    for zarr_path in all_zarrs:
        fov_path = zarr_path / first_fov
        with open_ome_zarr(fov_path) as ds:
            _, C_i, Z_i, Y_i, X_i = ds.data.shape
            all_channel_names.extend(list(ds.channel_names))
            total_elements_per_t += C_i * Z_i * Y_i * X_i
            if scale is None:
                scale = list(ds.scale)

    bytes_per_element = np.dtype(np.float32).itemsize
    gb_per_timepoint = total_elements_per_t * bytes_per_element / 1e9
    gb_ram_per_cpu = max(4, int(np.ceil(gb_per_timepoint * 4)))

    return all_channel_names, scale, gb_ram_per_cpu


def _submit_stage1_jobs(
    position_keys, lf_zarr, ls_zarr, plots_dir,
    slurm_out_path, cluster, slurm_args,
    lf_mask_radius, n_std, z_window, z_final, beads_fov,
    z_index=None,
    qc_metrics=None,
    manual_drop_frames=None,
    qc_thresholds=None,
):
    """Submit stage 1 SLURM jobs in two phases: core metadata then QC metrics.

    Phase 1: Per-FOV core jobs (bbox, z_focus, blank frames, drop list).
    Phase 2: Per-(FOV, metric) QC jobs that read core outputs.
    Beads QC runs in parallel with phase 1.

    Parameters
    ----------
    qc_metrics : list of str or None
        Which QC metrics to run. If None, runs all QC_METRICS.
        Valid values: "laplacian", "entropy", "hf_ratio", "frc", "bleach", "fov_registration".

    Returns (ok_results, beads_qc_job). beads_qc_job may be None.
    """
    if qc_metrics is None:
        qc_metrics = QC_METRICS
    else:
        invalid = set(qc_metrics) - set(QC_METRICS)
        if invalid:
            raise ValueError(f"Unknown QC metrics: {invalid}. Valid: {QC_METRICS}")
    print(f"\n=== STAGE 1: Computing metadata for {len(position_keys)} FOVs ===")
    print(f"  Resources: {slurm_args['slurm_cpus_per_task']} CPUs, {slurm_args['slurm_mem_per_cpu']} RAM/CPU")

    # --- Phase 1: Core metadata per FOV ---
    print(f"\n--- Phase 1: Core metadata ({len(position_keys)} jobs) ---")
    executor_core = submitit.AutoExecutor(folder=slurm_out_path / "stage1_core", cluster=cluster)
    executor_core.update_parameters(slurm_job_name="dynacell_s1_core", **slurm_args)

    jobs_core = []
    fov_info = []  # (fov, fov_name, output_plots_dir)
    with submitit.helpers.clean_env(), executor_core.batch():
        for position_key in position_keys:
            fov = "/".join(position_key)
            fov_name = "_".join(position_key)
            output_plots_dir = plots_dir / fov_name
            output_plots_dir.mkdir(parents=True, exist_ok=True)

            fov_manual_drops = None
            if manual_drop_frames and fov in manual_drop_frames:
                fov_manual_drops = manual_drop_frames[fov]

            job = executor_core.submit(
                compute_fov_core,
                im_lf_path=lf_zarr / fov,
                im_ls_path=ls_zarr / fov,
                output_plots_dir=output_plots_dir,
                fov=fov,
                lf_mask_radius=lf_mask_radius,
                n_std=n_std,
                z_window=z_window,
                z_index=z_index,
                manual_drop_frames=fov_manual_drops,
                DEBUG=True,
                qc_thresholds=qc_thresholds,
            )
            jobs_core.append(job)
            fov_info.append((fov, fov_name, output_plots_dir))

    job_ids_core = [job.job_id for job in jobs_core]
    log_path_core = slurm_out_path / "stage1_core_job_ids.log"
    with log_path_core.open("w") as f:
        f.write("\n".join(job_ids_core))
    print(f"Phase 1: Submitted {len(jobs_core)} core jobs. IDs: {log_path_core}")

    # Submit beads registration QC + core metadata (runs in parallel with phase 1)
    beads_qc_job = None
    if beads_fov is not None:
        beads_fov_name = "_".join(beads_fov.split("/"))
        beads_plots_dir = plots_dir / beads_fov_name
        beads_plots_dir.mkdir(parents=True, exist_ok=True)

        executor_beads = submitit.AutoExecutor(
            folder=slurm_out_path / "beads_qc", cluster=cluster
        )
        executor_beads.update_parameters(slurm_job_name="dynacell_beads_qc", **slurm_args)
        beads_n_std = n_std
        if qc_thresholds and "beads_registration" in qc_thresholds:
            beads_n_std = qc_thresholds["beads_registration"].get("n_std", n_std)
        beads_qc_job = executor_beads.submit(
            compute_beads_registration_qc,
            im_lf_path=lf_zarr / beads_fov,
            im_ls_path=ls_zarr / beads_fov,
            output_plots_dir=beads_plots_dir,
            n_std=beads_n_std,
        )
        print(f"Beads QC: Submitted job {beads_qc_job.job_id}")

        # Also run core metadata on beads FOV (for overlap plot / diagnostics)
        executor_beads_core = submitit.AutoExecutor(
            folder=slurm_out_path / "beads_core", cluster=cluster
        )
        executor_beads_core.update_parameters(slurm_job_name="dynacell_beads_core", **slurm_args)
        executor_beads_core.submit(
            compute_fov_core,
            im_lf_path=lf_zarr / beads_fov,
            im_ls_path=ls_zarr / beads_fov,
            output_plots_dir=beads_plots_dir,
            fov=beads_fov,
            lf_mask_radius=lf_mask_radius,
            n_std=n_std,
            z_window=z_window,
            z_index=z_index,
            DEBUG=True,
            qc_thresholds=qc_thresholds,
        )
        print(f"Beads core: Submitted diagnostics job")

    # Wait for phase 1 completion
    print("\nWaiting for phase 1 (core) jobs to complete...")
    results_core = [job.result() for job in jobs_core]

    failed = [r for r in results_core if r.get("status") != "ok"]
    if failed:
        print(f"WARNING: {len(failed)} FOVs failed in phase 1:")
        for r in failed:
            print(f"  {r}")

    ok_results = [r for r in results_core if r.get("status") == "ok"]
    ok_fov_set = {"_".join(r["fov"].split("/")) for r in ok_results}

    # --- Phase 2: QC metrics per (FOV, metric) ---
    ok_fov_info = [
        (fov, fov_name, out_dir)
        for fov, fov_name, out_dir in fov_info
        if fov_name in ok_fov_set
    ]
    n_qc_jobs = len(ok_fov_info) * len(qc_metrics)
    print(f"\n--- Phase 2: QC metrics ({len(ok_fov_info)} FOVs x {len(qc_metrics)} metrics = {n_qc_jobs} jobs) ---")

    executor_qc = submitit.AutoExecutor(folder=slurm_out_path / "stage1_qc", cluster=cluster)
    executor_qc.update_parameters(slurm_job_name="dynacell_s1_qc", **slurm_args)

    jobs_qc = []
    with submitit.helpers.clean_env(), executor_qc.batch():
        for fov, fov_name, output_plots_dir in ok_fov_info:
            for metric in qc_metrics:
                job = executor_qc.submit(
                    run_fov_qc,
                    metric=metric,
                    im_lf_path=lf_zarr / fov,
                    im_ls_path=ls_zarr / fov,
                    output_plots_dir=output_plots_dir,
                    lf_mask_radius=lf_mask_radius,
                    z_final=z_final,
                    n_std=n_std,
                    qc_thresholds=qc_thresholds,
                )
                jobs_qc.append(job)

    job_ids_qc = [job.job_id for job in jobs_qc]
    log_path_qc = slurm_out_path / "stage1_qc_job_ids.log"
    with log_path_qc.open("w") as f:
        f.write("\n".join(job_ids_qc))
    print(f"Phase 2: Submitted {len(jobs_qc)} QC jobs. IDs: {log_path_qc}")

    # Wait for phase 2 completion
    print("\nWaiting for phase 2 (QC) jobs to complete...")
    qc_results = [job.result() for job in jobs_qc]
    qc_errors = [r for r in qc_results if isinstance(r, str) and r.startswith("error")]
    if qc_errors:
        print(f"WARNING: {len(qc_errors)} QC jobs had errors:")
        for e in qc_errors:
            print(f"  {e}")

    print(f"Phase 2 complete: {len(qc_results) - len(qc_errors)}/{len(qc_results)} QC jobs succeeded")

    return ok_results, beads_qc_job


def _load_stage1_results(stage1_dir, plots_dir, position_keys):
    """Read stage 1 results from an existing run directory."""
    print(f"\n=== Skipping stage 1, reading results from {stage1_dir} ===")
    global_summary_path = stage1_dir / "global_summary.csv"
    if global_summary_path.exists():
        ok_results = pd.read_csv(global_summary_path).to_dict("records")
    else:
        ok_results = []
        for position_key in position_keys:
            fov_name = "_".join(position_key)
            summary_path = plots_dir / fov_name / "fov_summary.csv"
            if summary_path.exists():
                summary = pd.read_csv(summary_path).to_dict("records")[0]
                if summary.get("status") == "ok":
                    ok_results.append(summary)
    print(f"  Loaded {len(ok_results)} FOV results from stage 1")
    return ok_results


def _collect_beads_qc(beads_qc_job, beads_fov, plots_dir, n_std):
    """Collect beads registration QC results from a running job or existing CSV."""
    beads_drop_indices = np.array([], dtype=int)
    if beads_qc_job is not None:
        print("\nWaiting for beads QC job to complete...")
        beads_qc_result = beads_qc_job.result()
        beads_drop_indices = beads_qc_result["drop_indices"]
        print(f"Beads QC: {len(beads_drop_indices)} bad registration timepoints")
    elif beads_fov is not None:
        beads_fov_name = "_".join(beads_fov.split("/"))
        beads_qc_csv = plots_dir / beads_fov_name / "registration_qc.csv"
        if beads_qc_csv.exists():
            beads_qc_df = pd.read_csv(beads_qc_csv)
            shift_mag = beads_qc_df["shift_magnitude"].values
            pearson = beads_qc_df["pearson_corr"].values
            mu_s, sigma_s = np.nanmean(shift_mag), np.nanstd(shift_mag)
            mu_c, sigma_c = np.nanmean(pearson), np.nanstd(pearson)
            shift_bad = np.where(shift_mag > mu_s + n_std * sigma_s)[0]
            corr_bad = np.where(pearson < mu_c - n_std * sigma_c)[0]
            beads_drop_indices = np.array(
                sorted(set(shift_bad) | set(corr_bad)), dtype=int
            )
            print(f"Beads QC (from CSV): {len(beads_drop_indices)} bad registration timepoints")
        else:
            print(f"WARNING: No beads QC CSV found at {beads_qc_csv}")
    return beads_drop_indices


def _qualify_fovs(
    ok_results, plots_dir, max_drops, beads_fov, beads_drop_indices,
    position_keys, exclude_fovs, output_dir,
):
    """Disqualify FOVs with too many drops and generate annotations.csv.

    Returns the qualified results list, or an empty list on fatal error.
    """
    fov_drop_counts = {}
    ok_fov_set = {"_".join(r["fov"].split("/")) for r in ok_results}

    for fov_result in ok_results:
        fov = fov_result["fov"]
        fov_name = "_".join(fov.split("/"))
        drop_path = plots_dir / fov_name / "drop_list.csv"
        n_drops = 0
        if drop_path.exists() and drop_path.stat().st_size > 0:
            drop_df = pd.read_csv(drop_path)
            n_drops = len(drop_df)
        fov_drop_counts[fov_name] = n_drops

    if max_drops is not None:
        disqualified_fovs = {
            fov_name for fov_name, n in fov_drop_counts.items() if n > max_drops
        }
    else:
        disqualified_fovs = set()

    # Filter to qualified
    qualified_results = [
        r for r in ok_results
        if "_".join(r["fov"].split("/")) not in disqualified_fovs
    ]

    print(f"\n=== FOV qualification (max_drops={max_drops}) ===")
    print(f"  Total FOVs: {len(position_keys)}")
    print(f"  Failed stage 1: {len(position_keys) - len(ok_results)}")
    if max_drops is not None:
        print(f"  Disqualified (>{max_drops} drops): {len(disqualified_fovs)}")
        for fov_name in sorted(disqualified_fovs):
            print(f"    {fov_name}: {fov_drop_counts[fov_name]} drops")
    else:
        print(f"  Disqualified: 0 (no threshold set)")
    print(f"  Qualified for stage 2: {len(qualified_results)}")

    if not qualified_results:
        print("ERROR: No qualified FOVs remaining after disqualification")

    return qualified_results


def _build_unified_drop_list(ok_results, plots_dir, beads_drop_indices, output_dir,
                             drop_metrics=None):
    """Merge per-FOV drops, beads QC, and QC metric outliers into a global drop list.

    Parameters
    ----------
    drop_metrics : list of str
        Sources whose outliers should be used to drop frames. Includes both
        core sources ("blank", "z_focus", "manual") and QC metric names
        (e.g. "laplacian", "frc"). Default ["blank", "z_focus", "manual"].

    Returns (global_keep_indices, global_drop_csv_path).
    Writes drop_list_all_fovs.csv.
    """
    # Mapping from drop_list.csv reason strings to drop_metrics names
    REASON_TO_SOURCE = {
        "blank_frame": "blank",
        "z_focus_outlier": "z_focus",
        "manual": "manual",
    }

    # Mapping from QC metric name to CSV filename
    METRIC_CSV = {
        "laplacian": "laplacian_qc.csv",
        "entropy": "entropy_qc.csv",
        "hf_ratio": "hf_ratio_qc.csv",
        "frc": "frc_qc.csv",
        "bleach": "bleach_qc.csv",
        "max_intensity": "max_intensity_qc.csv",
        "fov_registration": "fov_registration_qc.csv",
    }

    if drop_metrics is None:
        drop_metrics = ["blank", "z_focus", "manual"]
    drop_metrics_set = set(drop_metrics)

    ok_fovs = [r["fov"] for r in ok_results]
    T_total = ok_results[0]["T_total"]

    global_drop_set = set()
    drop_rows = []

    # Core drops from per-FOV drop_list.csv (blank, z_focus, manual)
    core_counts = {}
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        drop_path = plots_dir / fov_name / "drop_list.csv"
        if drop_path.exists() and drop_path.stat().st_size > 0:
            drop_df = pd.read_csv(drop_path)
            for _, row in drop_df.iterrows():
                t = int(row["t"])
                # A row can have multiple reasons (comma-separated)
                reasons = [r.strip() for r in str(row["reason"]).split(",")]
                # Check if any reason matches an enabled drop_metrics source
                matched = False
                for reason in reasons:
                    source = REASON_TO_SOURCE.get(reason, reason)
                    if source in drop_metrics_set:
                        matched = True
                        break
                if matched:
                    global_drop_set.add(t)
                    drop_rows.append({
                        "fov": fov_name,
                        "t": t,
                        "reason": row["reason"],
                    })
                    for reason in reasons:
                        source = REASON_TO_SOURCE.get(reason, reason)
                        core_counts[source] = core_counts.get(source, 0) + 1

    # QC metric outliers from drop_metrics
    qc_metric_names = drop_metrics_set - {"blank", "z_focus", "manual"}
    n_metric_drops = 0
    metric_counts = {}
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        for metric in qc_metric_names:
            csv_name = METRIC_CSV.get(metric)
            if csv_name is None:
                continue
            csv_path = plots_dir / fov_name / csv_name
            outlier_t = _read_qc_outliers(csv_path)
            new_t = outlier_t - global_drop_set
            if new_t:
                n_metric_drops += len(new_t)
                metric_counts[metric] = metric_counts.get(metric, 0) + len(new_t)
                global_drop_set.update(new_t)
            for t in sorted(outlier_t):
                drop_rows.append({
                    "fov": fov_name,
                    "t": int(t),
                    "reason": f"{metric}_outlier",
                })

    # Beads registration QC (always applied if beads_fov was specified)
    beads_drop_set = set(int(t) for t in beads_drop_indices)
    n_beads_new = len(beads_drop_set - global_drop_set)
    global_drop_set.update(beads_drop_set)
    for t in sorted(beads_drop_set):
        drop_rows.append({
            "fov": "beads_qc",
            "t": int(t),
            "reason": "bad_registration",
        })

    global_keep_indices = np.array(
        sorted(set(range(T_total)) - global_drop_set), dtype=int
    )

    print(f"\n=== Unified drop list ===")
    print(f"  drop_metrics: {drop_metrics}")
    for source, count in sorted(core_counts.items()):
        print(f"  {source}: {count} entries")
    if metric_counts:
        for metric, count in sorted(metric_counts.items()):
            print(f"  {metric}_outlier: {count} new timepoints")
    if len(beads_drop_indices) > 0:
        print(f"  beads_qc: {len(beads_drop_indices)} ({n_beads_new} new)")
    print(f"  Total global drops: {len(global_drop_set)}")
    print(f"  Remaining after unified drop: {len(global_keep_indices)} / {T_total}")

    # Write CSV
    drop_all_df = pd.DataFrame(drop_rows, columns=["fov", "t", "reason"])
    global_drop_csv = output_dir / "drop_list_all_fovs.csv"
    drop_all_df.to_csv(global_drop_csv, index=False)
    print(f"Saved combined drop list to {global_drop_csv}")
    print(f"  Total entries: {len(drop_all_df)}, unique timepoints dropped: {drop_all_df['t'].nunique()}")

    if len(drop_all_df) > 0:
        drop_counts = drop_all_df.groupby("fov").size().reset_index(name="n_dropped")
        print("  Drops per FOV:")
        for _, row in drop_counts.iterrows():
            print(f"    {row['fov']}: {row['n_dropped']}")

    return global_keep_indices, global_drop_csv


def _read_qc_outliers(csv_path, outlier_col="is_outlier"):
    """Read a per-timepoint QC CSV and return a set of outlier timepoints."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(csv_path)
    if outlier_col not in df.columns:
        return set()
    return set(int(row["t"]) for _, row in df.iterrows() if row[outlier_col] == 1)


def _read_qc_values(csv_path, value_col):
    """Read a per-timepoint QC CSV and return a dict {t: value}."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}
    df = pd.read_csv(csv_path)
    if value_col not in df.columns:
        return {}
    return {int(row["t"]): row[value_col] for _, row in df.iterrows()}


def _generate_annotations(
    ok_results, plots_dir, beads_drop_indices, global_keep_indices, output_dir, dataset,
):
    """Generate per-FOV annotation CSVs and a combined dataset-level CSV.

    Each per-FOV CSV is written to per_fov_analysis/<fov_name>/annotation.csv with columns:
        t, t_original, status, blank, bad_reg, out_of_focus,
        entropy_outlier, hf_outlier, frc_outlier, frc_mean_corr,
        reg_outlier, fov_reg_pcc, comments

    The combined CSV at output_dir/annotations.csv adds a 'fov' column.
    Only post-drop timepoints (global_keep_indices) are included.
    """
    beads_drop_set = set(int(t) for t in beads_drop_indices)

    # Build mapping: output index -> original timepoint
    keep_list = sorted(int(t) for t in global_keep_indices)
    t_out_to_orig = {i: orig_t for i, orig_t in enumerate(keep_list)}
    T_out = len(keep_list)

    columns = [
        "t", "t_original", "status", "blank", "bad_reg", "out_of_focus",
        "entropy_outlier", "hf_outlier", "frc_outlier", "frc_mean_corr",
        "reg_outlier", "fov_reg_pcc", "comments",
    ]

    all_rows = []
    for fov_result in ok_results:
        fov = fov_result["fov"]
        fov_name = "_".join(fov.split("/"))
        fov_dir = plots_dir / fov_name

        # Collect per-FOV drop reasons from drop_list.csv
        fov_blank_set = set()
        fov_oof_set = set()
        drop_path = fov_dir / "drop_list.csv"
        if drop_path.exists() and drop_path.stat().st_size > 0:
            drop_df = pd.read_csv(drop_path)
            for _, row in drop_df.iterrows():
                t_orig = int(row["t"])
                reason = str(row["reason"]).lower()
                if "blank" in reason:
                    fov_blank_set.add(t_orig)
                elif "z_focus" in reason or "entropy" in reason:
                    fov_oof_set.add(t_orig)

        # Read per-FOV QC outlier sets and values
        entropy_outliers = _read_qc_outliers(fov_dir / "entropy_qc.csv")
        hf_outliers = _read_qc_outliers(fov_dir / "hf_ratio_qc.csv")
        frc_outliers = _read_qc_outliers(fov_dir / "frc_qc.csv")
        frc_values = _read_qc_values(fov_dir / "frc_qc.csv", "frc_mean_corr")
        fov_reg_outliers = _read_qc_outliers(fov_dir / "fov_registration_qc.csv")
        fov_reg_values = _read_qc_values(fov_dir / "fov_registration_qc.csv", "pearson_corr")

        # Build rows for this FOV (only kept timepoints)
        fov_rows = []
        for t_out in range(T_out):
            t_orig = t_out_to_orig[t_out]
            fov_rows.append({
                "t": t_out,
                "t_original": t_orig,
                "status": 0,
                "blank": 1 if t_orig in fov_blank_set else 0,
                "bad_reg": 1 if t_orig in beads_drop_set else 0,
                "out_of_focus": 1 if t_orig in fov_oof_set else 0,
                "entropy_outlier": 1 if t_orig in entropy_outliers else 0,
                "hf_outlier": 1 if t_orig in hf_outliers else 0,
                "frc_outlier": 1 if t_orig in frc_outliers else 0,
                "frc_mean_corr": round(frc_values.get(t_orig, float("nan")), 4),
                "reg_outlier": 1 if t_orig in fov_reg_outliers else 0,
                "fov_reg_pcc": round(fov_reg_values.get(t_orig, float("nan")), 4),
                "comments": "",
            })

        # Write per-FOV annotation CSV with dataset header comment
        fov_df = pd.DataFrame(fov_rows, columns=columns)
        fov_annot_path = fov_dir / "annotation.csv"
        with open(fov_annot_path, "w") as f:
            f.write(f"# dataset: {dataset}\n")
            fov_df.to_csv(f, index=False)

        # Accumulate FOV-level summary for dataset CSV
        all_rows.append({
            "dataset": dataset,
            "fov": fov_name,
            "T_out": T_out,
            "n_dropped": fov_result["T_total"] - T_out,
            "status": 0,
            "comments": "",
        })

    # Write dataset-level annotations CSV (one row per FOV)
    ds_columns = ["dataset", "fov", "T_out", "n_dropped", "status", "comments"]
    combined_df = pd.DataFrame(all_rows, columns=ds_columns)
    annot_path = output_dir / "annotations.csv"
    combined_df.to_csv(annot_path, index=False)
    print(f"\nAnnotations: {len(combined_df)} FOVs")
    print(f"  Per-FOV CSVs (per-timepoint): per_fov_analysis/<fov>/annotation.csv")
    print(f"  Dataset CSV (per-FOV): {annot_path}")

    return annot_path


def _save_stage1_summary(ok_results, plots_dir, output_dir, all_channel_names,
                         z_final, global_keep_indices, drop_frames=True,
                         qc_thresholds=None):
    """Compute unified plate dims, save global summary and combined z_focus plots.

    Returns (T_min, Y_min, X_min, C_out, Z_out).
    """
    ok_fovs = [r["fov"] for r in ok_results]
    T_total = ok_results[0]["T_total"]
    if drop_frames:
        T_min = len(global_keep_indices)
    else:
        T_min = T_total
    Y_min = min(r["Y_crop"] for r in ok_results)
    X_min = min(r["X_crop"] for r in ok_results)
    C_out = len(all_channel_names)
    Z_out = z_final

    print(f"\n=== Stage 1 summary ===")
    print(f"  FOVs OK: {len(ok_results)}")
    print(f"  T_min (after drops): {T_min}")
    print(f"  Y_min: {Y_min}, X_min: {X_min}")
    print(f"  Output plate shape: ({T_min}, {C_out}, {Z_out}, {Y_min}, {X_min})")

    # Save global summary -- use actual cropped values, not per-FOV stage 1 values
    global_summary = pd.DataFrame(ok_results)
    global_summary["T_out"] = T_min
    global_summary["Y_crop"] = Y_min
    global_summary["X_crop"] = X_min
    global_summary.to_csv(output_dir / "global_summary.csv", index=False)
    print(f"Saved global summary to {output_dir / 'global_summary.csv'}")

    # Combined all-FOV plots
    from dynacell.plotting import (
        plot_z_focus_all_fovs,
        plot_laplacian_all_fovs,
        plot_hf_ratio_all_fovs,
        plot_entropy_all_fovs,
        plot_frc_all_fovs,
        plot_max_intensity_all_fovs,
        plot_drop_correlation_all_fovs,
        plot_outlier_correlation_all_fovs,
        plot_registration_pcc_all_fovs,
        plot_outlier_heatmap_fov_metric,
        plot_outlier_cooccurrence_matrix,
        plot_outlier_temporal_density,
        plot_metric_correlation_matrix,
    )

    def _save_all_fovs_plot(fig, data, output_dir, name, data_col=None):
        """Helper: save figure + CSV for an all-FOVs plot."""
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        if data:
            if data_col:
                csv_data = {k: v[data_col] for k, v in data.items()}
            else:
                csv_data = data
            df = pd.DataFrame(csv_data)
            df.index.name = "t"
            df.to_csv(output_dir / f"{name}.csv")
        print(f"Saved {name}.png to {output_dir}")

    fig, z_data = plot_z_focus_all_fovs(ok_fovs, plots_dir)
    _save_all_fovs_plot(fig, z_data, output_dir, "z_focus_all_fovs")

    fig, lap_data = plot_laplacian_all_fovs(ok_fovs, plots_dir)
    _save_all_fovs_plot(fig, lap_data, output_dir, "laplacian_all_fovs")

    _qt = qc_thresholds or {}
    hf_n = _qt.get("hf_ratio", {}).get("n_std", 2.5)
    ent_n = _qt.get("entropy", {}).get("n_std", 2.5)
    frc_n = _qt.get("frc", {}).get("n_std", 2.5)

    fig, hf_data = plot_hf_ratio_all_fovs(ok_fovs, plots_dir, n_std=hf_n)
    _save_all_fovs_plot(fig, hf_data, output_dir, "hf_ratio_all_fovs", data_col="hf_ratio")

    fig, ent_data = plot_entropy_all_fovs(ok_fovs, plots_dir, n_std=ent_n)
    _save_all_fovs_plot(fig, ent_data, output_dir, "entropy_all_fovs", data_col="entropy")

    fig, frc_data = plot_frc_all_fovs(ok_fovs, plots_dir, n_std=frc_n)
    _save_all_fovs_plot(fig, frc_data, output_dir, "frc_all_fovs", data_col="frc_mean_corr")

    fig, maxint_data = plot_max_intensity_all_fovs(ok_fovs, plots_dir)
    _save_all_fovs_plot(fig, maxint_data, output_dir, "max_intensity_all_fovs")

    T_total_plot = ok_results[0]["T_total"] if ok_results else 66
    drop_fig, drop_df, corr_text = plot_drop_correlation_all_fovs(
        ok_fovs, plots_dir, T_total_plot)
    if drop_fig is not None:
        drop_fig.savefig(output_dir / "drop_correlation_all_fovs.png",
                         dpi=150, bbox_inches="tight")
        plt.close(drop_fig)
        drop_df.to_csv(output_dir / "drop_correlation_all_fovs.csv", index=False)
        print(f"Saved drop_correlation_all_fovs.png to {output_dir}")
        if corr_text:
            print("Correlated drops (>=2 FOVs at same timepoint):")
            for line in corr_text:
                print(line)
    else:
        print("No drops found across FOVs — skipping drop correlation plot")

    fig, pcc_data = plot_registration_pcc_all_fovs(ok_fovs, plots_dir)
    _save_all_fovs_plot(fig, pcc_data, output_dir, "registration_pcc_all_fovs")

    # Outlier correlation (reporting-only QC outliers across FOVs)
    out_fig, out_df, out_text = plot_outlier_correlation_all_fovs(
        ok_fovs, plots_dir, T_total_plot)
    if out_fig is not None:
        out_fig.savefig(output_dir / "outlier_correlation_all_fovs.png",
                        dpi=150, bbox_inches="tight")
        plt.close(out_fig)
        out_df.to_csv(output_dir / "outlier_correlation_all_fovs.csv", index=False)
        print(f"Saved outlier_correlation_all_fovs.png to {output_dir}")
        if out_text:
            print("Correlated outliers (>=2 FOVs at same timepoint):")
            for line in out_text:
                print(line)
    else:
        print("No QC outliers found across FOVs — skipping outlier correlation plot")

    # Cross-FOV diagnostic views
    fig, _ = plot_outlier_heatmap_fov_metric(ok_fovs, plots_dir, T_total_plot)
    fig.savefig(output_dir / "outlier_heatmap_fov_metric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved outlier_heatmap_fov_metric.png to {output_dir}")

    fig = plot_outlier_cooccurrence_matrix(ok_fovs, plots_dir, T_total_plot)
    fig.savefig(output_dir / "outlier_cooccurrence_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved outlier_cooccurrence_matrix.png to {output_dir}")

    fig = plot_outlier_temporal_density(ok_fovs, plots_dir, T_total_plot)
    fig.savefig(output_dir / "outlier_temporal_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved outlier_temporal_density.png to {output_dir}")

    fig = plot_metric_correlation_matrix(ok_fovs, plots_dir, T_total_plot)
    fig.savefig(output_dir / "metric_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric_correlation_matrix.png to {output_dir}")

    return T_min, Y_min, X_min, C_out, Z_out


def _run_stage2(
    ok_results, all_zarrs, output_zarr, plots_dir,
    slurm_out_path, cluster, slurm_args, global_drop_csv,
    z_final, T_min, Y_min, X_min, all_channel_names, scale,
    overlay_channels, output_dir, drop_frames=True, z_crop=None,
):
    """Create output plate, submit stage 2 crop jobs, wait, and generate QC report."""
    ok_fovs = [r["fov"] for r in ok_results]
    C_out = len(all_channel_names)
    Z_out = z_final

    print(f"\n=== STAGE 2: Cropping {len(ok_results)} FOVs ===")

    ok_position_keys = [tuple(fov.split("/")) for fov in ok_fovs]
    create_empty_plate(
        store_path=output_zarr,
        position_keys=ok_position_keys,
        shape=(T_min, C_out, Z_out, Y_min, X_min),
        chunks=(1, 1, Z_out, Y_min, X_min),
        scale=scale,
        channel_names=all_channel_names,
        dtype=np.float32,
        version='0.5',
    )
    print(f"Created output plate at {output_zarr}")

    lf_zarr, ls_zarr, ls_deconvolved_zarr, bf_zarr = all_zarrs

    executor_s2 = submitit.AutoExecutor(folder=slurm_out_path / "stage2", cluster=cluster)
    executor_s2.update_parameters(slurm_job_name="dynacell_s2_crop", **slurm_args)

    jobs_s2 = []
    with submitit.helpers.clean_env(), executor_s2.batch():
        for fov in ok_fovs:
            fov_name = "_".join(fov.split("/"))
            input_zarr_paths = [lf_zarr / fov, ls_zarr / fov, ls_deconvolved_zarr / fov, bf_zarr / fov]  # vs_zarr excluded

            job = executor_s2.submit(
                crop_fov,
                input_zarr_paths=input_zarr_paths,
                output_zarr=output_zarr,
                output_plots_dir=plots_dir / fov_name,
                fov=fov,
                global_drop_csv=global_drop_csv,
                z_final=z_final,
                T_out=T_min,
                Y_out=Y_min,
                X_out=X_min,
                drop_frames=drop_frames,
                z_crop=z_crop,
            )
            jobs_s2.append(job)

    job_ids_s2 = [job.job_id for job in jobs_s2]
    log_path_s2 = slurm_out_path / "stage2_job_ids.log"
    with log_path_s2.open("w") as f:
        f.write("\n".join(job_ids_s2))
    print(f"Stage 2: Submitted {len(jobs_s2)} jobs. IDs: {log_path_s2}")

    print("\nWaiting for stage 2 jobs to complete...")
    for job in jobs_s2:
        job.result()
    print(f"\nDone! Output plate at {output_zarr}")

    # Generate QC report
    print(f"\n=== Generating QC report ===")
    generate_dataset_report(output_dir, overlay_channels=overlay_channels)


def _finalize_run_log(run_log, run_log_path, start_time):
    """Update run log with execution time."""
    elapsed_sec = time.time() - start_time
    run_log["execution_time_sec"] = round(elapsed_sec, 1)
    run_log["execution_time_human"] = (
        f"{int(elapsed_sec // 3600)}h {int((elapsed_sec % 3600) // 60)}m {int(elapsed_sec % 60)}s"
    )
    with open(run_log_path, "w") as f:
        yaml.dump(run_log, f, default_flow_style=False, sort_keys=False)
    print(f"Updated run log with execution time: {run_log['execution_time_human']}")


## ===== Orchestrator =====

def run_all_fovs(
    root_path: Path,
    dataset: str,
    lf_mask_radius: float = 0.75,
    z_final: int = 64,
    n_std: float = 2.5,
    local: bool = False,
    stage1_run_dir: Path | None = None,
    beads_fov: str | None = None,
    max_drops: int | None = None,
    overlay_channels: list[str] | None = None,
    exclude_fovs: list[str] | None = None,
    include_fovs: list[str] | None = None,
    z_window: int | None = None,
    lf_zarr_override: Path | None = None,
    stages: list[int] | None = None,
    z_index: int | str | None = None,
    drop_frames: bool = True,
    qc_metrics: list[str] | None = None,
    drop_metrics: list[str] | None = None,
    annotations_dir: Path | None = None,
    qc_thresholds: dict | None = None,
    slurm_config: dict | None = None,
    z_crop: int | None = None,
):
    """Two-stage pipeline for dynacell preprocessing.

    Stage 1: Compute bbox, z_focus, drop list per FOV (parallel submitit jobs).
    Stage 2: After stage 1 completes, gather min T/Y/X, create plate, crop all FOVs.

    Parameters
    ----------
    stage1_run_dir : Path or None
        If provided, skip stage 1 and read its results from this existing run
        directory (e.g. run_20260312_112144). Stage 2 output goes into a new run dir.
    beads_fov : str or None
        FOV key for the beads position (e.g. "C/1/000000"). Used for
        registration QC: per-timepoint phase cross-correlation between
        LF and LS beads channels is computed and timepoints with poor
        registration are added to the global drop list.
    include_fovs : list of str or None
        If provided, only process these FOVs (e.g. ["A/1/000001", "A/1/001001"]).
        Applied before exclude_fovs. If None, process all discovered FOVs.
    stages : list of int or None
        Which stages to run: [1], [2], or [1, 2]. Default None means [1, 2].
        Stage 2 alone requires stage1_run_dir to read existing results.
    z_index : int, "mid", or None
        If an int, use this fixed z index instead of auto-detecting z_focus.
        If "mid", use the mid-Z slice (Z // 2).
        If None (default), compute z_focus per timepoint.
    z_crop : int or None
        If set, use this fixed z as the crop center in stage 2 instead of
        the per-timepoint z_focus. Independent of z_index (z_focus detection
        still runs for QC when z_index is None).
    drop_frames : bool
        If True (default), drop flagged timepoints in stage 2 (blank, z_focus
        outliers, entropy outliers, bad registration). If False, keep all
        timepoints — the drop list is still computed for QC but not applied.
    qc_metrics : list of str or None
        Which QC metrics to compute in stage 1 phase 2. If None (default),
        runs all: laplacian, entropy, hf_ratio, frc, bleach, max_intensity,
        fov_registration. All metrics report `is_outlier` in their CSVs.
    drop_metrics : list of str or None
        Which sources' outliers should be used to drop frames. Combines core
        sources and QC metric names into a single list. Valid values:
          - "blank" — blank (all-zero) frames
          - "z_focus" — z_focus outlier frames
          - "manual" — manual drops from annotations
          - Any QC metric name (e.g. "laplacian", "frc", "entropy", etc.)
        If None (default), uses ["blank", "z_focus", "manual"].
        QC metric names must be a subset of qc_metrics.
    annotations_dir : Path or None
        Path to pre-annotations directory (created by dynacell_init_annotations.py).
        Contains annotations.csv (FOV-level) and per_fov/<fov>/annotation.csv
        (timepoint-level). FOVs with exclude=1 are excluded. Timepoints with
        exclude=1 are added to the drop list with reason "manual".
    """
    if stages is None:
        stages = [1, 2]
    if 2 in stages and 1 not in stages and stage1_run_dir is None:
        raise ValueError("Stage 2 alone requires stage1_run_dir to read existing results.")

    # Validate drop_metrics
    CORE_DROP_SOURCES = {"blank", "z_focus", "manual"}
    if drop_metrics is None:
        drop_metrics = ["blank", "z_focus", "manual"]
    else:
        effective_qc = qc_metrics if qc_metrics is not None else QC_METRICS
        valid_sources = CORE_DROP_SOURCES | set(effective_qc)
        invalid = set(drop_metrics) - valid_sources
        if invalid:
            raise ValueError(
                f"Unknown drop_metrics: {invalid}. "
                f"Valid: {sorted(valid_sources)}"
            )
    print(f"Drop metrics: {drop_metrics}")

    # Resolve input zarr paths
    lf_zarr, ls_zarr, ls_deconvolved_zarr, bf_zarr = _resolve_zarr_paths(
        root_path, dataset, lf_zarr_override
    )
    all_zarrs = [lf_zarr, ls_zarr, ls_deconvolved_zarr, bf_zarr]

    # Setup output directories
    if stage1_run_dir is not None:
        stage1_dir = root_path / dataset / "dynacell" / stage1_run_dir
        plots_dir = stage1_dir / "per_fov_analysis"
        print(f"Reusing stage 1 results from: {stage1_dir}")
    else:
        stage1_dir = None
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root_path / dataset / "dynacell" / f"run_{run_id}"
    output_zarr = output_dir / f"{dataset}.zarr"
    if stage1_dir is None:
        plots_dir = output_dir / "per_fov_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Symlink per_fov_analysis so the QC report can find FOV plots
    if stage1_dir is not None and stage1_dir != output_dir:
        link = output_dir / "per_fov_analysis"
        if not link.exists():
            link.symlink_to(plots_dir)
    print(f"Run ID: {run_id}")
    print(f"Output dir: {output_dir}")

    # Write run log
    _run_start_time = time.time()
    biahub_repo = Path(__file__).resolve().parent.parent.parent
    run_log = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "parameters": {
            "root_path": str(root_path),
            "dataset": dataset,
            "lf_mask_radius": lf_mask_radius,
            "z_final": z_final,
            "n_std": n_std,
            "local": local,
            "stage1_run_dir": str(stage1_run_dir) if stage1_run_dir else None,
            "beads_fov": beads_fov,
            "max_drops": max_drops,
            "overlay_channels": overlay_channels,
            "exclude_fovs": exclude_fovs,
            "include_fovs": include_fovs,
            "stages": stages,
            "z_window": z_window,
            "z_index": z_index,
            "drop_frames": drop_frames,
            "qc_metrics": qc_metrics,
            "drop_metrics": drop_metrics,
            "annotations_dir": str(annotations_dir) if annotations_dir else None,
            "qc_thresholds": qc_thresholds,
            "slurm_config": slurm_config,
        },
        "git": _get_git_info(biahub_repo),
    }
    run_log_path = output_dir / "run_log.yaml"
    with open(run_log_path, "w") as f:
        yaml.dump(run_log, f, default_flow_style=False, sort_keys=False)
    print(f"Saved run log to {run_log_path}")

    # Read pre-annotations (FOV-level exclusions)
    annot_exclude_fovs = []
    manual_drop_frames = {}  # fov_key -> list of timepoints to drop
    if annotations_dir is not None:
        annotations_dir = Path(annotations_dir)
        ds_annot = annotations_dir / "annotations.csv"
        if ds_annot.exists():
            annot_df = pd.read_csv(ds_annot)
            excluded = annot_df[annot_df["exclude"] == 1]["fov"].tolist()
            annot_exclude_fovs = excluded
            print(f"Pre-annotations: {len(excluded)} FOVs excluded by annotations")
        # Read per-FOV timepoint exclusions
        per_fov_dir = annotations_dir / "per_fov"
        if per_fov_dir.exists():
            for fov_dir in per_fov_dir.iterdir():
                if not fov_dir.is_dir():
                    continue
                fov_annot = fov_dir / "annotation.csv"
                if fov_annot.exists():
                    fov_df = pd.read_csv(fov_annot, comment="#")
                    excluded_t = fov_df[fov_df["exclude"] == 1]["t"].tolist()
                    if excluded_t:
                        fov_key = "/".join(fov_dir.name.split("_"))
                        manual_drop_frames[fov_key] = [int(t) for t in excluded_t]
            if manual_drop_frames:
                n_total = sum(len(v) for v in manual_drop_frames.values())
                print(f"Pre-annotations: {n_total} timepoints manually excluded across {len(manual_drop_frames)} FOVs")

    # Merge annotation exclusions with user-specified exclude_fovs
    if annot_exclude_fovs:
        if exclude_fovs is None:
            exclude_fovs = annot_exclude_fovs
        else:
            exclude_fovs = list(set(exclude_fovs + annot_exclude_fovs))

    # Discover and filter FOVs
    position_keys, beads_fov = _discover_and_filter_fovs(
        lf_zarr, include_fovs, exclude_fovs, beads_fov
    )

    # Read plate metadata and estimate resources
    all_channel_names, scale, gb_ram_per_cpu = _get_plate_metadata(
        all_zarrs, "/".join(position_keys[0])
    )
    # SLURM configuration: defaults overridden by slurm_config from YAML
    sc = slurm_config or {}
    num_cpus = sc.get("cpus_per_task", 4)
    cluster = "local" if local else "slurm"
    slurm_out_path = output_dir / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)
    scripts_dir = str(Path(__file__).resolve().parent.parent)
    slurm_args = {
        "slurm_mem_per_cpu": sc.get("mem_per_cpu", f"{gb_ram_per_cpu}G"),
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": sc.get("array_parallelism", 100),
        "slurm_time": sc.get("time", 120),
        "slurm_partition": sc.get("partition", "gpu"),
        "slurm_setup": [
            "conda activate biahub",
            f"export PYTHONPATH={scripts_dir}:$PYTHONPATH",
        ],
    }

    # Stage 1: compute per-FOV metadata or load existing results
    if 1 in stages and stage1_dir is None:
        ok_results, beads_qc_job = _submit_stage1_jobs(
            position_keys=position_keys,
            lf_zarr=lf_zarr, ls_zarr=ls_zarr,
            plots_dir=plots_dir,
            slurm_out_path=slurm_out_path, cluster=cluster, slurm_args=slurm_args,
            lf_mask_radius=lf_mask_radius, n_std=n_std,
            z_window=z_window, z_final=z_final,
            beads_fov=beads_fov,
            z_index=z_index,
            qc_metrics=qc_metrics,
            manual_drop_frames=manual_drop_frames,
            qc_thresholds=qc_thresholds,
        )
        if not ok_results:
            print("ERROR: No FOVs succeeded in stage 1")
            return
    elif stage1_dir is not None:
        ok_results = _load_stage1_results(stage1_dir, plots_dir, position_keys)
        beads_qc_job = None
        if not ok_results:
            print("ERROR: No valid stage 1 results found")
            return

    # Collect beads registration QC results
    beads_n_std = n_std
    if qc_thresholds and "beads_registration" in qc_thresholds:
        beads_n_std = qc_thresholds["beads_registration"].get("n_std", n_std)
    beads_drop_indices = _collect_beads_qc(beads_qc_job, beads_fov, plots_dir, beads_n_std)

    # Dataset-level dust QC on LF Phase channel
    if 1 in stages:
        ok_fov_keys = [r["fov"] for r in ok_results]
        blank_frames_per_fov = {}
        for r in ok_results:
            fov_name = "_".join(r["fov"].split("/"))
            drop_path = plots_dir / fov_name / "drop_list.csv"
            if drop_path.exists() and drop_path.stat().st_size > 0:
                drop_df = pd.read_csv(drop_path)
                blank_t = drop_df[
                    drop_df["reason"].str.contains("blank", case=False, na=False)
                ]["t"].values
                blank_frames_per_fov[r["fov"]] = set(int(t) for t in blank_t)
        print(f"\n=== Dust QC ===")
        compute_dust_qc(
            lf_zarr=lf_zarr,
            fov_keys=ok_fov_keys,
            output_dir=output_dir,
            lf_mask_radius=lf_mask_radius,
            blank_frames_per_fov=blank_frames_per_fov,
        )

        print(f"\n=== Bleach QC (dataset summary from per-FOV CSVs) ===")
        compute_bleach_qc(
            fov_keys=ok_fov_keys,
            plots_dir=plots_dir,
            output_dir=output_dir,
        )

    # Qualify FOVs (disqualify those with too many drops)
    ok_results = _qualify_fovs(
        ok_results, plots_dir, max_drops, beads_fov, beads_drop_indices,
        position_keys, exclude_fovs, output_dir,
    )
    if not ok_results:
        _finalize_run_log(run_log, run_log_path, _run_start_time)
        return

    # Build unified drop list across all qualified FOVs + beads QC
    global_keep_indices, global_drop_csv = _build_unified_drop_list(
        ok_results, plots_dir, beads_drop_indices, output_dir,
        drop_metrics=drop_metrics,
    )

    # Generate per-FOV and combined annotations CSVs
    _generate_annotations(
        ok_results, plots_dir, beads_drop_indices, global_keep_indices,
        output_dir, dataset,
    )

    # Save stage 1 summary and combined plots
    T_min, Y_min, X_min, C_out, Z_out = _save_stage1_summary(
        ok_results, plots_dir, output_dir, all_channel_names,
        z_final, global_keep_indices, drop_frames=drop_frames,
        qc_thresholds=qc_thresholds,
    )

    # Early exit if only stage 1 was requested
    if 2 not in stages:
        _finalize_run_log(run_log, run_log_path, _run_start_time)
        print(f"\nStage 1 only — skipping stage 2. Output: {output_dir}")
        return

    # Stage 2: crop into unified plate
    _run_stage2(
        ok_results=ok_results, all_zarrs=all_zarrs,
        output_zarr=output_zarr, plots_dir=plots_dir,
        slurm_out_path=slurm_out_path, cluster=cluster, slurm_args=slurm_args,
        global_drop_csv=global_drop_csv, z_final=z_final,
        T_min=T_min, Y_min=Y_min, X_min=X_min,
        all_channel_names=all_channel_names, scale=scale,
        overlay_channels=overlay_channels, output_dir=output_dir,
        drop_frames=drop_frames, z_crop=z_crop,
    )
    _finalize_run_log(run_log, run_log_path, _run_start_time)


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return as dict."""
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def run_from_config(config_path: str | Path, local: bool = False, stage1_run_dir: Path | None = None):
    """Run the pipeline from a YAML config file."""
    cfg = load_config(config_path)

    ds = cfg["dataset"]
    proc = cfg.get("processing", {})
    fovs = cfg.get("fovs", {})

    run_all_fovs(
        root_path=Path(ds["root_path"]),
        dataset=ds["name"],
        lf_mask_radius=proc.get("lf_mask_radius", 0.75),
        z_final=proc.get("z_final", 64),
        n_std=cfg.get("qc_thresholds", {}).get("z_focus", {}).get("n_std", 2.5),
        local=local,
        stage1_run_dir=stage1_run_dir,
        beads_fov=fovs.get("beads"),
        overlay_channels=cfg.get("overlay_channels"),
        exclude_fovs=fovs.get("exclude"),
        include_fovs=fovs.get("include"),
        z_window=proc.get("z_window"),
        z_index=proc.get("z_index"),
        stages=cfg.get("stages"),
        qc_metrics=cfg.get("qc_metrics"),
        drop_metrics=cfg.get("drop_metrics"),
        annotations_dir=Path(cfg["annotations_dir"]) if cfg.get("annotations_dir") else None,
        qc_thresholds=cfg.get("qc_thresholds"),
        slurm_config=cfg.get("slurm"),
        z_crop=proc.get("z_crop"),
        max_drops=cfg.get("max_drops"),
    )
