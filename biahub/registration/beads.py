import glob
import os
import shutil

from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple, Union

import ants
import click
import dask.array as da
import napari
import numpy as np
import pandas as pd
import submitit
import yaml

from iohub import open_ome_zarr
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from skimage.feature import match_descriptors
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from waveorder.focus import focus_from_transverse_band

from biahub.characterize_psf import detect_peaks
from biahub.cli.parsing import (
    config_filepath,
    local,
    output_filepath,
    sbatch_filepath,
    sbatch_to_submitit,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.cli.utils import (
    _check_nan_n_zeros,
    estimate_resources,
    model_to_yaml,
    yaml_to_model,
)
from biahub.optimize_registration import _optimize_registration
from biahub.register import (
    convert_transform_to_ants,
    convert_transform_to_numpy,
    get_3D_rescaling_matrix,
    get_3D_rotation_matrix,
)
from biahub.settings import (
    AffineTransformSettings,
    AntsRegistrationSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)
from biahub.registration.utils import plot_translations, validate_transforms, interpolate_transforms, check_transforms_difference, evaluate_transforms, save_transforms

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
    weight_dist_list = [0.5,1.0]
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
        click.echo(f"No timepoint with data found")
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

        if qc_metrics["score"] < threshold_score:
            click.echo(f"User config is not good enough, performing grid search")
            output_folder_path_grid_search = output_folder_path / f"grid_search/t_{t_initial}"
            output_folder_path_grid_search.mkdir(parents=True, exist_ok=True)
            cfg_grid_search = grid_search_registration(
                source_channel_zyx=source_channel_tzyx[t_initial],
                target_channel_zyx=target_channel_tzyx[t_initial],
                config=config,
                verbose=verbose,
                output_folder_path=output_folder_path_grid_search,
                cluster=cluster,
            )
            beads_match_settings = cfg_grid_search.beads_match_settings

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


def get_local_pca_features(
    points: ArrayLike,
    edges: list[tuple[int, int]],
) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute dominant direction and anisotropy for each point using PCA,
    using neighborhoods defined by existing graph edges.

    Parameters
    ----------
    points : ArrayLike
        (n, 2) array of points.
    edges : list[tuple[int, int]]
        List of edges (i, j) in the graph.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        - directions : (n, 3) array of dominant directions.
        - anisotropy : (n,) array of anisotropy.

    Notes
    -----
    The PCA features are computed as the dominant direction and anisotropy of the local neighborhood of each point.
    The direction is the first principal component of the local neighborhood.
    The anisotropy is the ratio of the first to third principal component of the local neighborhood.
    """
    n = len(points)
    directions = np.zeros((n, 3))
    anisotropy = np.zeros(n)

    # Build neighbor list from edges
    from collections import defaultdict

    neighbor_map = defaultdict(list)
    for i, j in edges:
        neighbor_map[i].append(j)

    for i in range(n):
        neighbors = neighbor_map[i]
        if not neighbors:
            directions[i] = np.nan
            anisotropy[i] = np.nan
            continue

        local_points = points[neighbors].astype(np.float32)
        local_points -= local_points.mean(axis=0)
        _, S, Vt = np.linalg.svd(local_points, full_matrices=False)

        directions[i] = Vt[0] if Vt.shape[0] > 0 else np.zeros(3)
        anisotropy[i] = S[0] / (S[2] + 1e-5) if len(S) >= 3 else 0.0

    return directions, anisotropy


def get_edge_descriptors(
    points: ArrayLike,
    edges: list[tuple[int, int]],
) -> ArrayLike:
    """
    Compute edge descriptors for a set of points.

    Parameters
    ----------
    points : ArrayLike
        (n, 2) array of points.
    edges : list[tuple[int, int]]
        List of edges (i, j) in the graph.

    Returns
    -------
    ArrayLike
        (n, 4) array of edge descriptors.
        Each row contains:
        - mean length
        - std length
        - mean angle
        - std angle

    Notes
    -----
    The edge descriptors are computed as the mean and standard deviation of the lengths and angles of the edges.
    """
    n = len(points)
    desc = np.zeros((n, 4))
    for i in range(n):
        neighbors = [j for a, j in edges if a == i]
        if not neighbors:
            continue
        vectors = points[neighbors] - points[i]
        lengths = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        desc[i, 0] = np.mean(lengths)
        desc[i, 1] = np.std(lengths)
        desc[i, 2] = np.mean(angles)
        desc[i, 3] = np.std(angles)
    return desc


def get_edge_attrs(
    points: ArrayLike,
    edges: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """
    Compute edge distances and angles for a set of points.

    Parameters
    ----------
    points : ArrayLike
        (n, 2) array of points.
    edges : list[tuple[int, int]]
        List of edges (i, j) in the graph.

    Returns
    -------
    tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]
        - distances : dict[tuple[int, int], float]
        - angles : dict[tuple[int, int], float]

    """
    distances, angles = {}, {}
    for i, j in edges:
        vec = points[j] - points[i]
        d = np.linalg.norm(vec)
        angle = np.arctan2(vec[1], vec[0])
        distances[(i, j)] = distances[(j, i)] = d
        angles[(i, j)] = angles[(j, i)] = angle
    return distances, angles


def match_hungarian_local_cost(
    i: int,
    j: int,
    s_neighbors: list[int],
    t_neighbors: list[int],
    source_attrs: dict[tuple[int, int], float],
    target_attrs: dict[tuple[int, int], float],
    default_cost: float,
) -> float:
    """
    Match neighbor edges between two graphs using the Hungarian algorithm for local cost estimation.
    The cost is the mean of the absolute differences between the source and target edge attributes.

    Parameters
    ----------
    i : int
        Index of the source edge.
    j : int
        Index of the target edge.
    s_neighbors : list[int]
        List of source neighbors.
    t_neighbors : list[int]
        List of target neighbors.
    source_attrs : dict[tuple[int, int], float]
        Dictionary of source edge attributes.
    target_attrs : dict[tuple[int, int], float]
        Dictionary of target edge attributes.
    """
    C = np.full((len(s_neighbors), len(t_neighbors)), default_cost)

    # compute cost matrix
    for ii, sn in enumerate(s_neighbors):
        # get target neighbors
        for jj, tn in enumerate(t_neighbors):
            s_edge = (i, sn)
            t_edge = (j, tn)
            if s_edge in source_attrs and t_edge in target_attrs:
                C[ii, jj] = abs(source_attrs[s_edge] - target_attrs[t_edge])

    # use hungarian algorithm to find the best match
    row_ind, col_ind = linear_sum_assignment(C)
    # get the mean of the matched costs
    matched_costs = C[row_ind, col_ind]
    # return the mean of the matched costs

    return matched_costs.mean() if len(matched_costs) > 0 else default_cost


def compute_edge_consistency_cost(
    n: int,
    m: int,
    source_attrs: dict[tuple[int, int], float],
    target_attrs: dict[tuple[int, int], float],
    source_edges: list[tuple[int, int]],
    target_edges: list[tuple[int, int]],
    default: float = 1e6,
    hungarian: bool = True,
) -> ArrayLike:
    """
    Compute the cost matrix for matching edges between two graphs.

    Parameters
    ----------
    n : int
        Number of source edges.
    m : int
        Number of target edges.
    source_attrs : dict[tuple[int, int], float]
        Dictionary of source edge attributes.
    target_attrs : dict[tuple[int, int], float]
        Dictionary of target edge attributes.
    source_edges : list[tuple[int, int]]
        List of edges (i, j) in source graph.
    target_edges : list[tuple[int, int]]
        List of edges (i, j) in target graph.
    default : float
        Default value for the cost matrix.
    hungarian : bool
        Whether to use the Hungarian algorithm for local cost estimation.
        If False, the cost matrix is computed as the mean of the absolute differences between the source and target edge attributes.
        If True, the cost matrix is computed as the mean of the absolute differences between the source and target edge attributes using the Hungarian algorithm.

    Returns
    -------
    ArrayLike
        Cost matrix of shape (n, m).

    Notes
    -----
    The cost matrix is computed as the mean of the absolute differences between the source and target edge attributes.
    """
    cost_matrix = np.full((n, m), default)
    for i in range(n):
        # get source neighbors
        s_neighbors = [j for a, j in source_edges if a == i]
        for j in range(m):
            # get target neighbors
            t_neighbors = [k for a, k in target_edges if a == j]
            if hungarian:
                # hungarian algorithm based cost estimation
                cost_matrix[i, j] = match_hungarian_local_cost(
                    i, j, s_neighbors, t_neighbors, source_attrs, target_attrs, default
                )
            else:
                # position based cost estimation (mean of the absolute differences between the source and target edge attributes)
                common_len = min(len(s_neighbors), len(t_neighbors))
                diffs = []
                for k in range(common_len):
                    s_edge = (i, s_neighbors[k])
                    t_edge = (j, t_neighbors[k])
                    if s_edge in source_attrs and t_edge in target_attrs:
                        v1 = source_attrs[s_edge]
                        v2 = target_attrs[t_edge]
                        diff = np.abs(v1 - v2)
                        diffs.append(diff)
                cost_matrix[i, j] = np.mean(diffs) if diffs else default

    return cost_matrix


def compute_cost_matrix(
    source_peaks: ArrayLike,
    target_peaks: ArrayLike,
    source_edges: list[tuple[int, int]],
    target_edges: list[tuple[int, int]],
    weights: dict[str, float] = None,
    distance_metric: str = 'euclidean',
    normalize: bool = False,
) -> ArrayLike:
    """
    Compute a cost matrix for matching peaks between two graphs based on:
    - Euclidean or other distance between peaks
    - Consistency in edge distances
    - Consistency in edge angles
    - PCA features
    - Edge descriptors

    Parameters
    ----------
    source_peaks : ArrayLike
        (n, 2) array of source node coordinates.
    target_peaks : ArrayLike
        (m, 2) array of target node coordinates.
    source_edges : list[tuple[int, int]]
        List of edges (i, j) in source graph.
    target_edges : list[tuple[int, int]]
        List of edges (i, j) in target graph.
    weights : dict[str, float]
        Weights for different cost components.
    distance_metric : str
        Metric for direct point-to-point distances.
    normalize : bool
        Whether to normalize the cost matrix.

    Notes
    -----
    The cost matrix is computed as the sum of the weighted costs for each component.
    The weights are defined in the `weights` parameter.
    The default weights are:
    - dist: 0.5
    - edge_angle: 1.0
    - edge_length: 1.0
    - pca_dir: 0.0
    - pca_aniso: 0.0
    - edge_descriptor: 0.0

    Returns
    -------
    ArrayLike
        Cost matrix of shape (n, m).
    """
    n, m = len(source_peaks), len(target_peaks)
    C_total = np.zeros((n, m))

    # --- Default weights ---
    default_weights = {
        "dist": 0.5,
        "edge_angle": 1.0,
        "edge_length": 1.0,
        "pca_dir": 0.0,
        "pca_aniso": 0.0,
        "edge_descriptor": 0.0,
    }
    if weights is None:
        weights = default_weights
    else:
        weights = {**default_weights, **weights}  # override defaults

    # --- Base distance cost ---
    if weights["dist"] > 0:
        C_dist = cdist(source_peaks, target_peaks, metric=distance_metric)
        if normalize:
            C_dist /= C_dist.max()
        C_total += weights["dist"] * C_dist

    # --- Edge angle and length costs ---
    source_dists, source_angles = get_edge_attrs(source_peaks, source_edges)
    target_dists, target_angles = get_edge_attrs(target_peaks, target_edges)

    if weights["edge_length"] > 0:
        C_edge_len = compute_edge_consistency_cost(
            n=n,
            m=m,
            source_attrs=source_dists,
            target_attrs=target_dists,
            source_edges=source_edges,
            target_edges=target_edges,
            default=1e6,
        )
        if normalize:
            C_edge_len /= C_edge_len.max()
        C_total += weights["edge_length"] * C_edge_len

    if weights["edge_angle"] > 0:
        C_edge_ang = compute_edge_consistency_cost(
            n=n,
            m=m,
            source_attrs=source_angles,
            target_attrs=target_angles,
            source_edges=source_edges,
            target_edges=target_edges,
            default=np.pi,
        )
        if normalize:
            C_edge_ang /= np.pi
        C_total += weights["edge_angle"] * C_edge_ang

    # --- PCA features ---
    if weights["pca_dir"] > 0 or weights["pca_aniso"] > 0:
        dirs_s, aniso_s = get_local_pca_features(source_peaks, source_edges)
        dirs_t, aniso_t = get_local_pca_features(target_peaks, target_edges)

        if weights["pca_dir"] > 0:
            dot = np.clip(np.dot(dirs_s, dirs_t.T), -1.0, 1.0)
            C_dir = 1 - np.abs(dot)
            if normalize:
                C_dir /= C_dir.max()
            C_total += weights["pca_dir"] * C_dir

        if weights["pca_aniso"] > 0:
            C_aniso = np.abs(aniso_s[:, None] - aniso_t[None, :])
            if normalize:
                C_aniso /= C_aniso.max()
            C_total += weights["pca_aniso"] * C_aniso
    # --- Edge descriptors ---
    if weights["edge_descriptor"] > 0:
        desc_s = get_edge_descriptors(source_peaks, source_edges)
        desc_t = get_edge_descriptors(target_peaks, target_edges)
        C_desc = cdist(desc_s, desc_t)
        if normalize:
            C_desc /= C_desc.max()
        C_total += weights["edge_descriptor"] * C_desc

    return C_total


def build_edge_graph(
    points: ArrayLike,
    mode: Literal["knn", "radius", "full"] = "knn",
    k: int = 5,
    radius: float = 30.0,
) -> list[tuple[int, int]]:
    """
    Build a set of edges for a graph based on a given strategy.

    Parameters
    ----------
    points : ArrayLike
        (N, 3) array of 3D point coordinates.
    mode : Literal["knn", "radius", "full"]
        Mode for building the edge graph.
    k : int
        Number of neighbors if mode == "knn".
    radius : float
        Distance threshold if mode == "radius".

    Returns
    -------
    list[tuple[int, int]]
        List of (i, j) index pairs representing edges.
    """
    n = len(points)
    if n <= 1:
        return []

    if mode == "knn":
        k_eff = min(k + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k_eff).fit(points)
        _, indices = nbrs.kneighbors(points)
        edges = [(i, j) for i in range(n) for j in indices[i] if i != j]

    elif mode == "radius":
        graph = radius_neighbors_graph(
            points, radius=radius, mode='connectivity', include_self=False
        )
        if graph.nnz == 0:
            return []
        edges = [(i, j) for i in range(n) for j in graph[i].nonzero()[1]]

    elif mode == "full":
        edges = [(i, j) for i in range(n) for j in range(n) if i != j]

    return edges


def match_hungarian_global_cost(
    C: ArrayLike,
    cost_threshold: float = 1e5,
    dummy_cost: float = 1e6,
    max_ratio: float = None,
) -> ArrayLike:
    """
    Runs Hungarian matching with padding for unequal-sized graphs,
    optionally applying max_ratio filtering similar to match_descriptors.

    Parameters
    ----------
    C : ArrayLike
        Cost matrix of shape (n_A, n_B).
    cost_threshold : float
        Maximum cost to consider a valid match.
    dummy_cost : float
        Cost assigned to dummy nodes (must be > cost_threshold).
    max_ratio : float, optional
        Maximum allowed ratio between best and second-best cost.

    Returns
    -------
    ArrayLike
        Array of shape (N_matches, 2) with valid (A_idx, B_idx) pairs.
    """
    n_A, n_B = C.shape
    n = max(n_A, n_B)

    # Pad cost matrix to square shape
    C_padded = np.full((n, n), fill_value=dummy_cost)
    C_padded[:n_A, :n_B] = C

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(C_padded)

    matches = []
    for i, j in zip(row_ind, col_ind):
        if i >= n_A or j >= n_B:
            continue  # matched with dummy
        if C[i, j] >= cost_threshold:
            continue  # too costly

        if max_ratio is not None:
            # Find second-best match for i
            costs_i = C[i, :]
            sorted_costs = np.sort(costs_i)
            if len(sorted_costs) > 1:
                second_best = sorted_costs[1]
                ratio = C[i, j] / (second_best + 1e-10)  # avoid division by zero
                if ratio > max_ratio:
                    continue  # reject if not sufficiently better
            # else (only one candidate) => accept by default

        matches.append((i, j))

    return np.array(matches)


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


def get_matches_from_hungarian(
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
    hungarian_settings = beads_match_settings.hungarian_match_settings
    cost_settings = hungarian_settings.cost_matrix_settings
    edge_settings = hungarian_settings.edge_graph_settings
    source_edges = build_edge_graph(
        source_peaks, mode=edge_settings.method, k=edge_settings.k, radius=edge_settings.radius
    )
    target_edges = build_edge_graph(
        target_peaks, mode=edge_settings.method, k=edge_settings.k, radius=edge_settings.radius
    )

    if hungarian_settings.cross_check:
        # Step 1: A → B
        C_ab = compute_cost_matrix(
            source_peaks,
            target_peaks,
            source_edges,
            target_edges,
            weights=cost_settings.weights,
            distance_metric=hungarian_settings.distance_metric,
            normalize=cost_settings.normalize,
        )

        matches_ab = match_hungarian_global_cost(
            C_ab,
            cost_threshold=np.quantile(C_ab, hungarian_settings.cost_threshold),
            max_ratio=hungarian_settings.max_ratio,
        )

        # Step 2: B → A (swap arguments)
        C_ba = compute_cost_matrix(
            target_peaks,
            source_peaks,
            target_edges,
            source_edges,
            weights=cost_settings.weights,
            distance_metric=hungarian_settings.distance_metric,
            normalize=cost_settings.normalize,
        )

        matches_ba = match_hungarian_global_cost(
            C_ba,
            cost_threshold=np.quantile(C_ba, hungarian_settings.cost_threshold),
            max_ratio=hungarian_settings.max_ratio,
        )

        # Step 3: Invert matches_ba to compare
        reverse_map = {(j, i) for i, j in matches_ba}

        # Step 4: Keep only symmetric matches
        matches = np.array([[i, j] for i, j in matches_ab if (i, j) in reverse_map])
    else:
        # without cross-check

        C = compute_cost_matrix(
            source_peaks,
            target_peaks,
            source_edges,
            target_edges,
            weights=cost_settings.weights,
            distance_metric=hungarian_settings.distance_metric,
            normalize=cost_settings.normalize,
        )

        matches = match_hungarian_global_cost(
            C,
            cost_threshold=np.quantile(C, hungarian_settings.cost_threshold),
            max_ratio=hungarian_settings.max_ratio,
        )
    return matches


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
        matches = match_descriptors(
            source_peaks,
            target_peaks,
            metric=match_descriptor_settings.distance_metric,
            max_ratio=match_descriptor_settings.max_ratio,
            cross_check=match_descriptor_settings.cross_check,
        )

    elif beads_match_settings.algorithm == 'hungarian':
        matches = get_matches_from_hungarian(
            source_peaks=source_peaks,
            target_peaks=target_peaks,
            beads_match_settings=beads_match_settings,
            verbose=verbose,
        )

    if verbose:
        click.echo(f'Total of matches: {len(matches)}')

    return matches


def filter_matches(
    matches: ArrayLike,
    source_peaks: ArrayLike,
    target_peaks: ArrayLike,
    angle_threshold: float = 30,
    min_distance_threshold: float = 0.01,
    max_distance_threshold: float = 0.95,
    verbose: bool = False,
) -> ArrayLike:
    """
    Filter matches based on angle and distance thresholds.

    Parameters
    ----------
    matches : ArrayLike
        (n, 2) array of matches.
    source_peaks : ArrayLike
        (n, 2) array of source peaks.
    target_peaks : ArrayLike
        (n, 2) array of target peaks.
    angle_threshold : float
        Maximum allowed deviation from dominant angle (degrees).
    min_distance_threshold : float
        Lower quantile cutoff for distance filtering (e.g. 0.05 keeps matches above 5th percentile).
    max_distance_threshold : float
        Upper quantile cutoff for distance filtering (e.g. 0.95 keeps matches below 95th percentile).
    verbose : bool
        If True, prints detailed logs.

    Returns
    -------
    ArrayLike
        (n, 2) array of filtered matches.
    """
    # --- Distance filtering ---
    if min_distance_threshold is not None or max_distance_threshold is not None:
        dist = np.linalg.norm(
            source_peaks[matches[:, 0]] - target_peaks[matches[:, 1]], axis=1
        )

        low = np.quantile(dist, min_distance_threshold)
        high = np.quantile(dist, max_distance_threshold)

        if verbose:
            click.echo(
                f"Filtering matches with distance quantiles: [{min_distance_threshold}, {max_distance_threshold}]"
            )
            click.echo(f"Distance range: [{low:.3f}, {high:.3f}]")

        keep = (dist >= low) & (dist <= high)
        matches = matches[keep]

        if verbose:
            click.echo(f"Total matches after distance filtering: {len(matches)}")

    # --- Angle filtering ---
    if angle_threshold:
        vectors = target_peaks[matches[:, 1]] - source_peaks[matches[:, 0]]
        angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles_deg = np.degrees(angles_rad)

        bins = np.linspace(-180, 180, 36)
        hist, bin_edges = np.histogram(angles_deg, bins=bins)
        dominant_bin_index = np.argmax(hist)
        dominant_angle = (
            bin_edges[dominant_bin_index] + bin_edges[dominant_bin_index + 1]
        ) / 2

        filtered_indices = np.where(np.abs(angles_deg - dominant_angle) <= angle_threshold)[0]
        matches = matches[filtered_indices]

        if verbose:
            click.echo(f"Total matches after angle filtering: {len(matches)}")

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

    matches = filter_matches(
        matches=matches,
        source_peaks=source_peaks,
        target_peaks=target_peaks,
        angle_threshold=beads_match_settings.filter_angle_threshold,
        min_distance_threshold=beads_match_settings.filter_min_distance_threshold,
        max_distance_threshold=beads_match_settings.filter_max_distance_threshold,
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


