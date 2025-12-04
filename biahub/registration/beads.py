import glob
import os
import shutil

from datetime import datetime
from pathlib import Path
from typing import Optional

import ants
import click
import dask.array as da
import numpy as np
import pandas as pd
import submitit
import yaml

from iohub import open_ome_zarr
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from skimage.feature import match_descriptors
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform

from biahub.characterize_psf import detect_peaks
from biahub.cli.parsing import (
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import (
    _check_nan_n_zeros,
    estimate_resources,
    model_to_yaml,
)
from biahub.registration.utils import convert_transform_to_ants
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    EstimateRegistrationSettings,
)
from biahub.registration.graph_matching import Graph, GraphMatcher


def qc_bead_overlap(
    source_channel: da.Array,
    target_channel: da.Array,
    beads_match_settings: BeadsMatchSettings,
    tform: ArrayLike = np.eye(4),
    radius: int = 6,
    verbose: bool = False,
    t_idx: int = None,
    output_folder_path: Path = None,
):
    """
    Compute overlap-based QC metrics (IoU, Dice) and geometric accuracy (MSE)
    between detected bead peaks from LF (ref) and LS (mov) channels.

    Args:
        ref_peaks: (N_ref, 3) array of LF bead coordinates (z, y, x)
        mov_peaks: (N_mov, 3) array of LS bead coordinates (z, y, x)
        matches:   (M, 2) matched indices (optional)
        radius:    spherical neighborhood radius (voxels)
        ref_shape: optional 3D shape for visualization masks

    Returns:
        dict with:
          - overlap_count, total_ref, total_mov
          - iou, dice, mse_vox
          - ref_mask, mov_mask, sum_mask (if ref_shape provided)
    """
    if t_idx is None:
        source_channel_zyx = source_channel
        target_channel_zyx = target_channel
    else:
        source_channel_zyx = np.asarray(source_channel[t_idx])
        target_channel_zyx = np.asarray(target_channel[t_idx])

    source_data_ants = ants.from_numpy(source_channel_zyx)
    target_data_ants = ants.from_numpy(target_channel_zyx)
    source_channel_zyx = (
        convert_transform_to_ants(np.asarray(tform))
        .apply_to_image(source_data_ants, reference=target_data_ants)
        .numpy()
    )

    mov_peaks, ref_peaks = detect_bead_peaks(
        source_channel_zyx=source_channel_zyx,
        target_channel_zyx=target_channel_zyx,
        source_peaks_settings=beads_match_settings.source_peaks_settings,
        target_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=False,
    )
    if len(mov_peaks) == 0 or len(ref_peaks) == 0:
        print("No peaks found, returning nan metrics")
        return {
            "total_peaks_ref": 0,
            "total_peaks_mov": 0,
            "overlap_count": 0,
            "overlap_fraction": 0,
            "iou": 0,
            "score": 0,
        }

    # ---- Overlap counting using KDTree ----
    mov_tree = cKDTree(mov_peaks)

    ref_overlap_mask = np.zeros(len(ref_peaks), dtype=bool)
    mov_overlap_mask = np.zeros(len(mov_peaks), dtype=bool)

    for i, p in enumerate(ref_peaks):
        idx = mov_tree.query_ball_point(p, r=radius)
        if idx:
            ref_overlap_mask[i] = True
            mov_overlap_mask[idx] = True

    overlap_count = int(ref_overlap_mask.sum())
    total_ref = len(ref_peaks)
    total_mov = len(mov_peaks)
    union = total_ref + total_mov - overlap_count

    # ---- IoU and Dice ----
    iou = overlap_count / union if union > 0 else np.nan

    qc_metrics = {
        "total_peaks_ref": total_ref,
        "total_peaks_mov": total_mov,
        "overlap_count": overlap_count,
        "overlap_fraction": overlap_count / max(min(total_mov, total_ref), 1),
        "iou": iou,
    }
    qc_metrics["score"] = qc_score(qc_metrics)

    if verbose:
        click.echo(f"QC Metrics: {qc_metrics}")

    if output_folder_path is not None:
        qc_metrics_df = pd.DataFrame([qc_metrics])
        if t_idx is not None:
            out_path = output_folder_path / f"{t_idx}.csv"

        else:
            out_path = output_folder_path / "qc_metrics.csv"
        qc_metrics_df.to_csv(out_path, index=False)

    return qc_metrics


def _config_with_hungarian_overrides(
    config: EstimateRegistrationSettings,
    *,
    cost_threshold,
    max_ratio,
    k,
    dist,
    edge_angle,
    edge_length,
    pca_dir,
    pca_aniso,
    edge_descriptor,
) -> EstimateRegistrationSettings:
    """
    Return a NEW, validated config with overrides applied to Hungarian settings.
    Works even if nested fields are dicts, since we modify the dict dump and
    revalidate via the Pydantic model constructor.
    """
    data = config.model_dump()  # deep dict
    hs = data["beads_match_settings"]["hungarian_match_settings"]

    hs["cost_threshold"] = float(cost_threshold)
    hs["max_ratio"] = float(max_ratio)

    # edge graph
    hs.setdefault("edge_graph_settings", {})
    hs["edge_graph_settings"]["k"] = int(k)
    hs["edge_graph_settings"]["method"] = hs["edge_graph_settings"].get("method", "knn")

    # cost matrix + weights
    hs.setdefault("cost_matrix_settings", {})
    cms = hs["cost_matrix_settings"]
    cms.setdefault("weights", {})
    w = cms["weights"]
    w["dist"] = float(dist)
    w["edge_angle"] = float(edge_angle)
    w["edge_length"] = float(edge_length)
    w["pca_dir"] = float(pca_dir)
    w["pca_aniso"] = float(pca_aniso)
    w["edge_descriptor"] = float(edge_descriptor)

    # Rebuild validated model
    return EstimateRegistrationSettings(**data)


def qc_score(m, weight_overlap_fraction=0.60, weight_iou=0.40):
    """
    Compute the QC score based on the overlap fraction, IoU.
    The weights are defined in the `weight_overlap_fraction`, `weight_iou` parameters.
    """
    overlap_fraction = float(m.get("overlap_fraction", 0) or 0)
    iou = float(m.get("iou", 0) or 0)

    return weight_overlap_fraction * overlap_fraction + weight_iou * iou


def grid_search_registration(
    source_channel_zyx: da.Array,
    target_channel_zyx: da.Array,
    config: EstimateRegistrationSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
    cluster: str = 'local',
):

    slurm_out_path = output_folder_path / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)
    Z, Y, X = source_channel_zyx.shape
    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(1, 2, Z, Y, X), ram_multiplier=5, max_num_cpus=16
    )

    # Prepare SLURM arguments

    slurm_args = {
        "slurm_job_name": "grid_search_registration",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 30,
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }

    # Submitit executor
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")

    # prepare T0 data and peaks once
    source_channel_zyx = np.asarray(source_channel_zyx).astype(np.float32)
    target_channel_zyx = np.asarray(target_channel_zyx).astype(np.float32)

    # define grids
    cost_threshold_list = [0.05]
    k_list = [10]
    max_ratio_list = [1]
    weight_dist_list = [0.5, 1.0]
    weight_edge_angle_list = [0, 0.5, 1.0]
    weight_edge_length_list = [0, 0.5, 1.0]
    weight_pca_dir_list = [0]
    weight_pca_aniso_list = [0]
    weight_edge_descriptor_list = [0]

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for ct in cost_threshold_list:
            for k in k_list:
                for mr in max_ratio_list:
                    for wd in weight_dist_list:
                        for wa in weight_edge_angle_list:
                            for wl in weight_edge_length_list:
                                for wpdir in weight_pca_dir_list:
                                    for wpan in weight_pca_aniso_list:
                                        for wed in weight_edge_descriptor_list:
                                            key = f"ct{ct}_k{k}_mr{mr}_wd{wd}_wa{wa}_wl{wl}_wpdir{wpdir}_wpan{wpan}_wed{wed}"
                                            trial_outdir = output_folder_path / "grid" / key
                                            trial_outdir.mkdir(parents=True, exist_ok=True)
                                            job = executor.submit(
                                                test_cfg_registration,
                                                source_channel_zyx=source_channel_zyx,
                                                target_channel_zyx=target_channel_zyx,
                                                config=config,
                                                cost_threshold=ct,
                                                max_ratio=mr,
                                                k=k,
                                                weight_dist=wd,
                                                weight_edge_angle=wa,
                                                weight_edge_length=wl,
                                                weight_pca_dir=wpdir,
                                                weight_pca_aniso=wpan,
                                                weight_edge_descriptor=wed,
                                                verbose=verbose,
                                                output_folder_path=trial_outdir,
                                            )
                                            jobs.append(job)

    wait_for_jobs_to_finish(jobs)

    # --- aggregate results ---
    records = []
    for csv_path in glob.glob(str(output_folder_path / "grid" / "*" / "qc_metrics.csv")):
        trial_dir = Path(csv_path).parent

        df = pd.read_csv(csv_path)
        df["trial_dir"] = str(trial_dir)
        records.append(df)

    if len(records) == 0:
        click.echo("No grid results found; falling back to original settings.")
        return config
    else:
        all_qc = pd.concat(records, ignore_index=True)
        # pick best
        best_row = all_qc.sort_values("score", ascending=False).iloc[0]
        best_dir = Path(best_row["trial_dir"])
        best_cfg_path = best_dir / "config.yml"

        with open(best_cfg_path, "r") as f:
            best_cfg_dict = yaml.safe_load(f)
        best_cfg = EstimateRegistrationSettings(**best_cfg_dict)

        summary_csv = output_folder_path / "grid_summary.csv"
        all_qc.sort_values("score", ascending=False).to_csv(summary_csv, index=False)
        print(f"Wrote grid summary to: {summary_csv}")

        return best_cfg


def test_cfg_registration(
    source_channel_zyx: da.Array,
    target_channel_zyx: da.Array,
    config: EstimateRegistrationSettings,
    cost_threshold: float,
    max_ratio: float,
    k: int,
    weight_dist: float,
    weight_edge_angle: float,
    weight_edge_length: float,
    weight_pca_dir: float,
    weight_pca_aniso: float,
    weight_edge_descriptor: float,
    verbose: bool = False,
    output_folder_path: Path = None,
) -> list[ArrayLike]:
    key = (
        f"ct:{cost_threshold},k:{k},mr:{max_ratio},"
        f"wd:{weight_dist},wa:{weight_edge_angle},wl:{weight_edge_length},"
        f"wpdir:{weight_pca_dir},wpan:{weight_pca_aniso},wed:{weight_edge_descriptor}"
    )
    print("[grid] trying", key)

    # 1) build a NEW, validated config for this candidate
    cfg_try = _config_with_hungarian_overrides(
        config,
        cost_threshold=cost_threshold,
        max_ratio=max_ratio,
        k=k,
        dist=weight_dist,
        edge_angle=weight_edge_angle,
        edge_length=weight_edge_length,
        pca_dir=weight_pca_dir,
        pca_aniso=weight_pca_aniso,
        edge_descriptor=weight_edge_descriptor,
    )

    tform = estimate_transform_from_beads(
        t_idx=None,
        source_channel=source_channel_zyx,
        target_channel=target_channel_zyx,
        beads_match_settings=cfg_try.beads_match_settings,
        affine_transform_settings=cfg_try.affine_transform_settings,
        verbose=False,
        output_folder_path=None,
    )

    qc_bead_overlap(
        source_channel=source_channel_zyx,
        target_channel=target_channel_zyx,
        beads_match_settings=cfg_try.beads_match_settings,
        tform=tform,
        radius=6,
        verbose=verbose,
        t_idx=None,
        output_folder_path=output_folder_path,
    )

    model_to_yaml(cfg_try, output_folder_path / "config.yml")

    return cfg_try


def beads_based_registration(
    config: EstimateRegistrationSettings,
    source_channel_tzyx: da.Array,
    target_channel_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings = None,
    affine_transform_settings: AffineTransformSettings = None,
    verbose: bool = False,
    cluster: bool = False,
    sbatch_filepath: Path = None,
    output_folder_path: Path = None,
    quality_control: bool = True,
    threshold_score: float = 0.5,
    grid_search: bool = True,
) -> list[ArrayLike]:
    """
    Perform beads-based temporal registration of 4D data using affine transformations.

    This function calculates timepoint-specific affine transformations to align a source channel
    to a target channel in 4D (T, Z, Y, X) data. It validates, smooths, and interpolates transformations
    across timepoints for consistent registration.

    Parameters
    ----------
    source_channel_tzyx : da.Array
       4D array (T, Z, Y, X) of the source channel (Dask array).
    target_channel_tzyx : da.Array
       4D array (T, Z, Y, X) of the target channel (Dask array).
    beads_match_settings : BeadsMatchSettings
        Settings for the beads match.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, prints detailed logs of the registration process.
    cluster : bool
        If True, uses the cluster.
    sbatch_filepath : Path
        Path to the sbatch file.
    output_folder_path : Path
        Path to the output folder.
    quality_control : bool
        If True, performs quality control.
    threshold_score : float
        Threshold score for the quality control.
    grid_search : bool
        If True, performs grid search.
    Returns
    -------
    list[ArrayLike]
        List of affine transformation matrices (4x4), one for each timepoint.
        Invalid or missing transformations are interpolated.

    Notes
    -----
    Each timepoint is processed in parallel using submitit executor.
    Use verbose=True for detailed logging during registration. The verbose output will be saved at the same level as the output zarr.
    """

    (T, Z, Y, X) = source_channel_tzyx.shape
    output_transforms_path = output_folder_path / "xyz_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)
    initial_transform = affine_transform_settings.approx_transform

    # firt t with data
    for t_initial in range(T):
        source_channel_tzyx_t = source_channel_tzyx[t_initial]
        if np.sum(source_channel_tzyx_t) == 0 or np.sum(target_channel_tzyx[t_initial]) == 0:
            continue
        else:
            break

    if t_initial == T:
        click.echo("No timepoint with data found")
        return None

    if grid_search:
        output_path_test = output_folder_path / f"test_t{t_initial}_user_config"
        output_path_test.mkdir(parents=True, exist_ok=True)

        approx_transform = estimate_transform_from_beads(
            source_channel=source_channel_tzyx,
            target_channel=target_channel_tzyx,
            beads_match_settings=beads_match_settings,
            affine_transform_settings=affine_transform_settings,
            verbose=verbose,
            output_folder_path=None,
            t_idx=t_initial,
        )

        qc_metrics = qc_bead_overlap(
            source_channel=source_channel_tzyx,
            target_channel=target_channel_tzyx,
            beads_match_settings=beads_match_settings,
            tform=approx_transform,
            verbose=verbose,
            output_folder_path=output_path_test,
            t_idx=t_initial,
        )

        # if qc_metrics["score"] < threshold_score:
        #     click.echo("User config is not good enough, performing grid search")
        #     output_folder_path_grid_search = output_folder_path / f"grid_search/t_{t_initial}"
        #     output_folder_path_grid_search.mkdir(parents=True, exist_ok=True)
        #     cfg_grid_search = grid_search_registration(
        #         source_channel_zyx=source_channel_tzyx[t_initial],
        #         target_channel_zyx=target_channel_tzyx[t_initial],
        #         config=config,
        #         verbose=verbose,
        #         output_folder_path=output_folder_path_grid_search,
        #         cluster=cluster,
        #     )
        #     beads_match_settings = cfg_grid_search.beads_match_settings

    if affine_transform_settings.use_prev_t_transform:
        for t in range(T):
            if np.sum(source_channel_tzyx[t]) == 0 or np.sum(target_channel_tzyx[t]) == 0:
                click.echo(f"Timepoint {t} has no data, skipping")
            else:
                click.echo(f"Timepoint {t} has data, estimating transform")
                approx_transform = estimate_transform_from_beads(
                    source_channel=source_channel_tzyx,
                    target_channel=target_channel_tzyx,
                    beads_match_settings=beads_match_settings,
                    affine_transform_settings=affine_transform_settings,
                    verbose=verbose,
                    output_folder_path=output_transforms_path,
                    t_idx=t,
                )

                if approx_transform is not None:
                    print(f"Using approx transform for timepoint {t+1}: {approx_transform}")
                    affine_transform_settings.approx_transform = approx_transform
                else:
                    print(f"Using initial transform for timepoint {t+1}: {initial_transform}")
                    affine_transform_settings.approx_transform = initial_transform

    else:
        num_cpus, gb_ram_per_cpu = estimate_resources(
            shape=(T, 2, Z, Y, X), ram_multiplier=5, max_num_cpus=16
        )

        # Prepare SLURM arguments
        slurm_args = {
            "slurm_job_name": "estimate_registration",
            "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
            "slurm_cpus_per_task": num_cpus,
            "slurm_array_parallelism": 100,
            "slurm_time": 30,
            "slurm_partition": "preempted",
            "slurm_use_srun": False,
        }

        if sbatch_filepath:
            slurm_args.update(sbatch_to_submitit(sbatch_filepath))

        slurm_out_path = output_folder_path / "slurm_output"
        slurm_out_path.mkdir(parents=True, exist_ok=True)

        # Submitit executor
        executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
        executor.update_parameters(**slurm_args)
        click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")

        # Submit jobs
        jobs = []
        with submitit.helpers.clean_env(), executor.batch():
            for t in range(T):
                job = executor.submit(
                    estimate_transform_from_beads,
                    source_channel=source_channel_tzyx,
                    target_channel=target_channel_tzyx,
                    beads_match_settings=beads_match_settings,
                    affine_transform_settings=affine_transform_settings,
                    verbose=verbose,
                    output_folder_path=output_transforms_path,
                    t_idx=t,
                )
                jobs.append(job)

        # Save job IDs
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = slurm_out_path / f"job_ids_{timestamp}.log"
        with open(log_path, "w") as log_file:
            for job in jobs:
                log_file.write(f"{job.job_id}\n")

        wait_for_jobs_to_finish(jobs)

    transforms = []
    for t in range(T):
        file_path = output_transforms_path / f"{t}.npy"
        if not os.path.exists(file_path):
            transforms.append(None)
        else:
            T_zyx_shift = np.load(file_path).tolist()
            transforms.append(T_zyx_shift)
    # Remove the output temporary folder
    # shutil.rmtree(output_transforms_path)

    if quality_control:

        num_cpus, gb_ram_per_cpu = estimate_resources(
            shape=(T, 2, Z, Y, X), ram_multiplier=5, max_num_cpus=16
        )

        # Prepare SLURM arguments
        slurm_args = {
            "slurm_job_name": "qc_bead_overlap",
            "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
            "slurm_cpus_per_task": num_cpus,
            "slurm_array_parallelism": 100,
            "slurm_time": 30,
            "slurm_partition": "preempted",
            "slurm_use_srun": False,
        }

        if sbatch_filepath:
            slurm_args.update(sbatch_to_submitit(sbatch_filepath))

        slurm_out_path = output_folder_path / "slurm_output"
        slurm_out_path.mkdir(parents=True, exist_ok=True)
        output_folder_qc = output_folder_path / "qc_metrics_per_timepoint"
        output_folder_qc.mkdir(parents=True, exist_ok=True)

        # Submitit executor
        executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
        executor.update_parameters(**slurm_args)
        click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")

        # Submit jobs
        jobs = []
        with submitit.helpers.clean_env(), executor.batch():
            for t in range(T):
                job = executor.submit(
                    qc_bead_overlap,
                    source_channel=source_channel_tzyx,
                    target_channel=target_channel_tzyx,
                    beads_match_settings=beads_match_settings,
                    tform=transforms[t],
                    verbose=verbose,
                    output_folder_path=output_folder_qc,
                    t_idx=t,
                )
                jobs.append(job)

        # Save job IDs
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = slurm_out_path / f"job_ids_{timestamp}.log"
        with open(log_path, "w") as log_file:
            for job in jobs:
                log_file.write(f"{job.job_id}\n")

        wait_for_jobs_to_finish(jobs)

        output_folder_qc_summary = output_folder_path / "qc_summary"
        output_folder_qc_summary.mkdir(parents=True, exist_ok=True)

        qc_summary_df = pd.DataFrame()
        timepoints = []
        for t in range(T):
            file_path = output_folder_qc / f"{t}.csv"
            if file_path.exists():
                timepoints.append(t)
                qc_metrics_df = pd.read_csv(file_path)
                qc_metrics_df["timepoint"] = t
                qc_summary_df = pd.concat([qc_summary_df, qc_metrics_df])

        qc_summary_df.to_csv(output_folder_qc_summary / "qc_summary.csv", index=False)

        # AFTER (safe names + nicer CSV)
        num_cols = [
            c
            for c in qc_summary_df.columns
            if c != "timepoint" and pd.api.types.is_numeric_dtype(qc_summary_df[c])
        ]

        std_dev = qc_summary_df[num_cols].std()
        mean_ = qc_summary_df[num_cols].mean()
        min_ = qc_summary_df[num_cols].min()
        max_ = qc_summary_df[num_cols].max()
        median_ = qc_summary_df[num_cols].median()
        q1 = qc_summary_df[num_cols].quantile(0.25)
        q3 = qc_summary_df[num_cols].quantile(0.75)
        iqr = q3 - q1
        range_ = max_ - min_

        summary_stats_df = pd.DataFrame(
            {
                "metric": num_cols,
                "mean": mean_[num_cols].values,
                "std_dev": std_dev[num_cols].values,
                "min": min_[num_cols].values,
                "q1": q1[num_cols].values,
                "median": median_[num_cols].values,
                "q3": q3[num_cols].values,
                "max": max_[num_cols].values,
                "range": range_[num_cols].values,
                "iqr": iqr[num_cols].values,
            }
        )

        summary_stats_path = output_folder_qc_summary / "summary_stats.csv"
        summary_stats_df.to_csv(summary_stats_path, index=False)

        low_score_timepoints = []
        for t in timepoints:
            score_t = qc_summary_df[qc_summary_df["timepoint"] == t]["score"].values[0]
            if score_t < threshold_score:
                click.echo(f"Timepoint {t} has low score: {score_t}")
                low_score_timepoints.append(t)
        # save the low score timepoints to a file
        with open(output_folder_path / "low_score_timepoints.txt", "w") as f:
            for t in low_score_timepoints:
                f.write(f"{t}\n")

        # plot per metric over time
        # add x on the low score timepoints
        for metric in qc_summary_df.columns:
            if metric != "timepoint":
                plt.figure()
                plt.plot(qc_summary_df["timepoint"].values, qc_summary_df[metric].values)
                plt.scatter(
                    low_score_timepoints,
                    qc_summary_df[metric].values[low_score_timepoints],
                    color="red",
                    marker="x",
                )
                plt.xlabel("Timepoint")
                plt.ylabel(metric)
                if metric in ["overlap_fraction", "iou", "score"]:
                    plt.ylim(0, 1)
                elif metric in ["overlap_count", "total_peaks_ref", "total_peaks_mov"]:
                    plt.ylim(0, max(qc_summary_df[metric].values) + 10)
                plt.savefig(output_folder_qc_summary / f"{metric}.png")
                plt.close()

    return transforms


def detect_bead_peaks(
    source_channel_zyx: da.Array,
    target_channel_zyx: da.Array,
    source_peaks_settings: DetectPeaksSettings,
    target_peaks_settings: DetectPeaksSettings,
    verbose: bool = False,
    filter_dirty_peaks: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Detect peaks in source and target channels using the detect_peaks function.

    Parameters
    ----------
    source_channel_zyx : da.Array
        (T, Z, Y, X) array of the source channel (Dask array).
    target_channel_zyx : da.Array
        (T, Z, Y, X) array of the target channel (Dask array).
    source_peaks_settings : DetectPeaksSettings
        Settings for the source peaks.
    target_peaks_settings : DetectPeaksSettings
        Settings for the target peaks.
    verbose : bool
        If True, prints detailed logs during the process.
    filter_dirty_peaks : bool
        If True, filters the dirty peaks.
    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        Tuple of (source_peaks, target_peaks).
    """
    if verbose:
        click.echo('Detecting beads in source dataset')

    source_peaks = detect_peaks(
        source_channel_zyx,
        block_size=source_peaks_settings.block_size,
        threshold_abs=source_peaks_settings.threshold_abs,
        nms_distance=source_peaks_settings.nms_distance,
        min_distance=source_peaks_settings.min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo('Detecting beads in target dataset')

    target_peaks = detect_peaks(
        target_channel_zyx,
        block_size=target_peaks_settings.block_size,
        threshold_abs=target_peaks_settings.threshold_abs,
        nms_distance=target_peaks_settings.nms_distance,
        min_distance=target_peaks_settings.min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo(f'Total of peaks in source dataset: {len(source_peaks)}')
        click.echo(f'Total of peaks in target dataset: {len(target_peaks)}')

    if len(source_peaks) < 2 or len(target_peaks) < 2:
        click.echo('Not enough beads detected')
        return
    if filter_dirty_peaks:
        print("Filtering dirty peaks")
        with open_ome_zarr(
            Path(
                "/hpc/projects/intracellular_dashboard/viral-sensor/dirty_on_mantis/lf_mask_2025_05_01_A549_DENV_sensor_DENV_T_9_0.zarr/C/1/000000"
            )
        ) as dirty_mask_ds:
            dirty_mask_load = np.asarray(dirty_mask_ds.data[0, 0])

        # filter the dirty peaks
        # Keep only peaks whose (y, x) column is clean across all Z slices
        target_peaks_filtered = []
        for peak in target_peaks:
            z, y, x = peak.astype(int)
            if (
                0 <= y < dirty_mask_load.shape[1]
                and 0 <= x < dirty_mask_load.shape[2]
                and not dirty_mask_load[:, y, x].any()  # True if all Z are clean at (y, x)
            ):
                target_peaks_filtered.append(peak)
        target_peaks = np.array(target_peaks_filtered)
    return source_peaks, target_peaks


def get_matches_from_beads(
    source_peaks: ArrayLike,
    target_peaks: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
    verbose: bool = False,
) -> ArrayLike:
    """
    Get matches from beads using the hungarian algorithm.

    Parameters
    ----------
    source_peaks : ArrayLike
        (n, 2) array of source peaks.
    target_peaks : ArrayLike
        (m, 2) array of target peaks.
    beads_match_settings : BeadsMatchSettings
        Settings for the beads match.
    verbose : bool
        If True, prints detailed logs during the process.

    Returns
    -------
    ArrayLike
        (n, 2) array of matches.
    """
    if verbose:
        click.echo(f'Getting matches from beads with settings: {beads_match_settings}')

    if beads_match_settings.algorithm == 'match_descriptor':
        match_descriptor_settings = beads_match_settings.match_descriptor_settings

        matcher = GraphMatcher(
            algorithm='descriptor',
            cross_check=match_descriptor_settings.cross_check,
            max_ratio=match_descriptor_settings.max_ratio,
            metric=match_descriptor_settings.distance_metric,
            verbose=verbose
        )

        matches = matcher.match(moving, reference)
        print(f"Descriptor: {len(matches)} matches")

    elif beads_match_settings.algorithm == 'hungarian':

        hungarian_match_settings = beads_match_settings.hungarian_match_settings
        moving = Graph.from_nodes(source_peaks, mode='knn', k=hungarian_match_settings.edge_graph_settings.k)
        reference = Graph.from_nodes(target_peaks, mode='knn', k=hungarian_match_settings.edge_graph_settings.k)

        matcher = GraphMatcher(
            algorithm='hungarian',
            weights=hungarian_match_settings.cost_matrix_settings.weights,
            cost_threshold=hungarian_match_settings.cost_threshold,
            cross_check=hungarian_match_settings.cross_check,
            max_ratio=hungarian_match_settings.max_ratio,
            verbose=verbose
        )

        matches = matcher.match(moving, reference)
        print(f"Hungarian: {len(matches)} matches")


    # Filter as part of the pipeline
    matches = matcher.filter_matches(
        matches,
        moving,
        reference,
        angle_threshold=beads_match_settings.filter_angle_threshold,
        min_distance_quantile=beads_match_settings.filter_min_distance_threshold,
        max_distance_quantile=beads_match_settings.filter_max_distance_threshold
    )
        

    if verbose:
        click.echo(f'Total of matches: {len(matches)}')

    return matches


def estimate_transform(
    matches: ArrayLike,
    source_peaks: ArrayLike,
    target_peaks: ArrayLike,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
) -> ArrayLike:
    """
    Estimate the affine transformation matrix between source and target channels
    based on detected bead matches at a specific timepoint.

    Parameters
    ----------
    matches : ArrayLike
        (n, 2) array of matches.
    source_peaks : ArrayLike
        (n, 2) array of source peaks.
    target_peaks : ArrayLike
        (n, 2) array of target peaks.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, prints detailed logs during the process.

    Returns
    -------
    ArrayLike
        (4, 4) array of the affine transformation matrix.
    """
    if verbose:
        click.echo(f"Estimating transform with settings: {affine_transform_settings}")

    if affine_transform_settings.transform_type == 'affine':
        tform = AffineTransform(dimensionality=3)

    elif affine_transform_settings.transform_type == 'euclidean':
        tform = EuclideanTransform(dimensionality=3)

    elif affine_transform_settings.transform_type == 'similarity':
        tform = SimilarityTransform(dimensionality=3)

    else:
        raise ValueError(f'Unknown transform type: {affine_transform_settings.transform_type}')

    tform.estimate(source_peaks[matches[:, 0]], target_peaks[matches[:, 1]])

    return tform


def estimate_transform_from_beads(
    t_idx: int,
    source_channel: da.Array,
    target_channel: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    output_folder_path: Path = None,
) -> list | None:
    """
    Calculate the affine transformation matrix between source and target channels
    based on detected bead matches at a specific timepoint.

    This function detects beads in both source and target datasets, matches them,
    and computes an affine transformation to align the two channels. It applies
    various filtering steps, including angle-based filtering, to improve match quality.

    Parameters
    ----------
    t_idx : int
        Timepoint index to process.
    source_channel : da.Array
       4D array (T, Z, Y, X) of the source channel (Dask array).
    target_channel : da.Array
       4D array (T, Z, Y, X) of the target channel (Dask array).
    beads_match_settings : BeadsMatchSettings
        Settings for the beads match.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, prints detailed logs during the process.
    slurm : bool
        If True, uses SLURM for parallel processing.
    output_folder_path : Path
        Path to save the output.

    Returns
    -------
    list | None
        A 4x4 affine transformation matrix as a nested list if successful,
        or None if no valid transformation could be calculated.

    Notes
    -----
    Uses ANTsPy for initial transformation application and bead detection.
    Peaks (beads) are detected using a block-based algorithm with thresholds for source and target datasets.
    Bead matches are filtered based on distance and angular deviation from the dominant direction.
    If fewer than three matches are found after filtering, the function returns None.
    """
    if t_idx is not None:
        click.echo(f'Processing timepoint: {t_idx}')
        source_channel_zyx = np.asarray(source_channel[t_idx]).astype(np.float32)
        target_channel_zyx = np.asarray(target_channel[t_idx]).astype(np.float32)
    else:
        source_channel_zyx = np.asarray(source_channel).astype(np.float32)
        target_channel_zyx = np.asarray(target_channel).astype(np.float32)

    if _check_nan_n_zeros(source_channel_zyx) or _check_nan_n_zeros(target_channel_zyx):
        click.echo(f'Beads data is missing at timepoint {t_idx}')
        return

    approx_tform = np.asarray(affine_transform_settings.approx_transform)
    source_data_ants = ants.from_numpy(source_channel_zyx)
    target_data_ants = ants.from_numpy(target_channel_zyx)
    source_data_reg_approx = (
        convert_transform_to_ants(approx_tform)
        .apply_to_image(source_data_ants, reference=target_data_ants)
        .numpy()
    )

    source_peaks, target_peaks = detect_bead_peaks(
        source_channel_zyx=source_data_reg_approx,
        target_channel_zyx=target_channel_zyx,
        source_peaks_settings=beads_match_settings.source_peaks_settings,
        target_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=verbose,
    )
    

    matches = get_matches_from_beads(
        source_peaks=source_peaks,
        target_peaks=target_peaks,
        beads_match_settings=beads_match_settings,
        verbose=verbose,
    )


    if len(matches) < 3:
        click.echo(
            f'Source and target beads were not matches successfully for timepoint {t_idx}'
        )
        return

    tform = estimate_transform(
        matches=matches,
        source_peaks=source_peaks,
        target_peaks=target_peaks,
        affine_transform_settings=affine_transform_settings,
        verbose=verbose,
    )
    compount_tform = np.asarray(approx_tform) @ tform.inverse.params

    if verbose:
        click.echo(f'Matches: {matches}')
        click.echo(f"tform.params: {tform.params}")
        click.echo(f"tform.inverse.params: {tform.inverse.params}")
        click.echo(f"compount_tform: {compount_tform}")
    if output_folder_path:
        print(f"Saving transform to {output_folder_path}")
        output_folder_path.mkdir(parents=True, exist_ok=True)
        np.save(output_folder_path / f"{t_idx}.npy", compount_tform)

    return compount_tform.tolist()


def estimate_xyz_stabilization_with_beads(
    channel_tzyx: da.Array,
    beads_match_settings: BeadsMatchSettings,
    affine_transform_settings: AffineTransformSettings,
    verbose: bool = False,
    cluster: str = "local",
    sbatch_filepath: Optional[Path] = None,
    output_folder_path: Path = None,
) -> list[ArrayLike]:
    """
    Estimate the xyz stabilization for a single position.

    Parameters
    ----------
    channel_tzyx : da.Array
        Source channel data.
    beads_match_settings : BeadsMatchSettings
        Settings for the beads match.
    affine_transform_settings : AffineTransformSettings
        Settings for the affine transform.
    verbose : bool
        If True, print verbose output.
    cluster : str
        Cluster to use.
    sbatch_filepath : Path
        Path to the sbatch file.
    output_folder_path : Path
        Path to the output folder.

    Returns
    -------
    list[ArrayLike]
        List of the xyz stabilization for each timepoint.
    """

    (T, Z, Y, X) = channel_tzyx.shape

    if beads_match_settings.t_reference == "first":
        target_channel_tzyx = np.broadcast_to(channel_tzyx[0], (T, Z, Y, X)).copy()
    elif beads_match_settings.t_reference == "previous":
        target_channel_tzyx = np.roll(channel_tzyx, shift=-1, axis=0)
        target_channel_tzyx[0] = channel_tzyx[0]

    else:
        raise ValueError("Invalid reference. Please use 'first' or 'previous as reference")

    output_transforms_path = output_folder_path / "xyz_transforms"
    output_transforms_path.mkdir(parents=True, exist_ok=True)
    initial_transform = affine_transform_settings.approx_transform
    if affine_transform_settings.use_prev_t_transform:
        for t in range(1, T, 1):
            approx_transform = estimate_transform_from_beads(
                source_channel_tzyx=channel_tzyx,
                target_channel_tzyx=target_channel_tzyx,
                verbose=verbose,
                beads_match_settings=beads_match_settings,
                affine_transform_settings=affine_transform_settings,
                slurm=True,
                output_folder_path=output_transforms_path,
                t_idx=t,
            )
        if approx_transform is not None:
            print(f"Using approx transform for timepoint {t+1}: {approx_transform}")
            affine_transform_settings.approx_transform = approx_transform
        else:
            print(f"Using initial transform for timepoint {t+1}: {initial_transform}")
            affine_transform_settings.approx_transform = initial_transform
    else:

        # Compute transformations in parallel

        num_cpus, gb_ram_per_cpu = estimate_resources(
            shape=(T, 1, Z, Y, X), ram_multiplier=5, max_num_cpus=16
        )

        # Prepare SLURM arguments
        slurm_args = {
            "slurm_job_name": "estimate_focus_z",
            "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
            "slurm_cpus_per_task": num_cpus,
            "slurm_array_parallelism": 100,
            "slurm_time": 30,
            "slurm_partition": "preempted",
        }

        if sbatch_filepath:
            slurm_args.update(sbatch_to_submitit(sbatch_filepath))

        slurm_out_path = output_folder_path / "slurm_output"
        slurm_out_path.mkdir(parents=True, exist_ok=True)

        # Submitit executor
        executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
        executor.update_parameters(**slurm_args)

        click.echo(f"Submitting SLURM focus estimation jobs with resources: {slurm_args}")

        # Submit jobs
        jobs = []
        with submitit.helpers.clean_env(), executor.batch():
            for t in range(1, T, 1):
                job = executor.submit(
                    estimate_transform_from_beads,
                    source_channel_tzyx=channel_tzyx,
                    target_channel_tzyx=target_channel_tzyx,
                    verbose=verbose,
                    beads_match_settings=beads_match_settings,
                    affine_transform_settings=affine_transform_settings,
                    slurm=True,
                    output_folder_path=output_transforms_path,
                    t_idx=t,
                )
                jobs.append(job)

        # Save job IDs
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = slurm_out_path / f"job_ids_{timestamp}.log"
        with open(log_path, "w") as log_file:
            for job in jobs:
                log_file.write(f"{job.job_id}\n")

        wait_for_jobs_to_finish(jobs)

    # Load the transforms
    transforms = [np.eye(4).tolist()]
    for t in range(1, T):
        file_path = output_transforms_path / f"{t}.npy"
        if not os.path.exists(file_path):
            transforms.append(None)
            click.echo(f"Transform for timepoint {t} not found.")
        else:
            T_zyx_shift = np.load(file_path).tolist()
            transforms.append(T_zyx_shift)

    # Check if the number of transforms matches the number of timepoints
    if len(transforms) != T:
        raise ValueError(
            f"Number of transforms {len(transforms)} does not match number of timepoints {T}"
        )

    # Remove the output folder
    shutil.rmtree(output_transforms_path)

    return transforms
