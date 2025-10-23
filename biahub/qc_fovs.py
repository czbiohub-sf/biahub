import csv
import os
import warnings
from pathlib import Path
from typing import List

import click
import numpy as np
import submitit

from iohub import open_ome_zarr
from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import sbatch_filepath, local, monitor, sbatch_to_submitit


def detect_blank_fov(data: np.ndarray, min_val: float, max_val: float) -> bool:
    """
    Detect if a FOV is blank based on min/max values and NaN checks.

    Parameters
    ----------
    data : np.ndarray
        3D array (Z, Y, X) containing the image data for a single channel
    min_val : float
        Minimum value in the data
    max_val : float
        Maximum value in the data

    Returns
    -------
    bool
        True if FOV is blank, False otherwise
    """
    # Check if either min or max is NaN
    if np.isnan(min_val) or np.isnan(max_val):
        return True
    
    # Check if max value is 0
    if max_val == 0:
        return True
    
    # Check if min and max are the same (no variation in data)
    if min_val == max_val:
        return True
    
    return False


def analyze_fov_channel(data: np.ndarray, fov_name: str, channel: str) -> dict:
    """
    Analyze a single FOV channel and return quality metrics.
    
    Parameters
    ----------
    data : np.ndarray
        3D array (Z, Y, X) containing the image data for a single channel
    fov_name : str
        Name of the field of view
    channel : str
        Channel name
        
    Returns
    -------
    dict
        Dictionary containing QC metrics
    """
    # Compute statistics efficiently
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    median_val = float(np.median(data))
    
    # Detect if FOV is blank
    is_blank = detect_blank_fov(data, min_val, max_val)
    
    return {
        'fov_name': fov_name,
        'channel': channel,
        'blank': is_blank,
        'min': min_val,
        'max': max_val,
        'median': median_val
    }


def process_single_fov_for_qc(
    zarr_path: Path, well_path: str, field_path: str, output_dir: Path
) -> Path:
    """
    Process a single FOV for QC analysis and write results to individual CSV.
    """
    fov_name = f"{well_path}/{field_path}"
    safe_fov_name = fov_name.replace('/', '_')
    output_csv = output_dir / f"qc_fov_{safe_fov_name}.csv"
    
    print(f"Processing FOV {fov_name} -> {output_csv}")
    
    try:
        with open_ome_zarr(zarr_path, mode='r') as store:
            field_store = store[well_path][field_path]
            dataset = field_store['0']
            
            # Get channel names
            if hasattr(field_store, 'zattrs') and 'omero' in field_store.zattrs:
                omero_metadata = field_store.zattrs['omero']
                channel_names = [ch['label'] for ch in omero_metadata.get('channels', [])]
            else:
                num_channels = dataset.shape[1]
                channel_names = [f'Channel_{i}' for i in range(num_channels)]
            
            # Process each channel for this FOV
            fov_results = []
            for c_idx, channel_name in enumerate(channel_names):
                channel_data = dataset[0, c_idx]
                result = analyze_fov_channel(channel_data, fov_name, channel_name)
                fov_results.append(result)
            
            # Write results to individual CSV file
            append_to_csv(fov_results, output_csv, write_header=True)
            print(f"Completed FOV {fov_name} -> {output_csv}")
            
    except Exception as e:
        # Write error info to CSV with more details
        error_result = {
            'fov_name': fov_name,
            'channel': f'ERROR: {str(e)}',
            'blank': True,
            'min': float('nan'),
            'max': float('nan'),
            'median': float('nan')
        }
        append_to_csv([error_result], output_csv, write_header=True)
        print(f"Error processing FOV {fov_name}: {e}")
        # Don't re-raise the exception to prevent job failure
    
    return output_csv


def process_hcs_zarr_store(zarr_path: Path, output_csv: Path) -> int:
    """
    Process an OME HCS zarr store and analyze all FOVs sequentially.
    """
    total_processed = 0
    first_fov = True
    
    try:
        with open_ome_zarr(zarr_path, mode='r') as store:
            if hasattr(store, 'zgroup') and 'plate' in store.zattrs:
                plate_metadata = store.zattrs['plate']
                
                for well_info in plate_metadata.get('wells', []):
                    well_path = well_info['path']
                    well_store = store[well_path]
                    
                    if 'well' in well_store.zattrs:
                        well_metadata = well_store.zattrs['well']
                        
                        for field_info in well_metadata.get('images', []):
                            field_path = field_info['path']
                            fov_name = f"{well_path}/{field_path}"
                            
                            try:
                                field_store = well_store[field_path]
                                dataset = field_store['0']
                                
                                # Get channel names
                                if hasattr(field_store, 'zattrs') and 'omero' in field_store.zattrs:
                                    omero_metadata = field_store.zattrs['omero']
                                    channel_names = [ch['label'] for ch in omero_metadata.get('channels', [])]
                                else:
                                    num_channels = dataset.shape[1]
                                    channel_names = [f'Channel_{i}' for i in range(num_channels)]
                                
                                # Process each channel for this FOV
                                fov_results = []
                                for c_idx, channel_name in enumerate(channel_names):
                                    channel_data = dataset[0, c_idx]
                                    result = analyze_fov_channel(channel_data, fov_name, channel_name)
                                    fov_results.append(result)
                                    total_processed += 1
                                
                                # Append results to CSV after processing all channels in this FOV
                                append_to_csv(fov_results, output_csv, write_header=first_fov)
                                first_fov = False
                                click.echo(f"Processed FOV {fov_name} ({len(fov_results)} channels)")
                                
                            except Exception as e:
                                click.echo(f"Warning: Could not process field {fov_name}: {e}")
                                continue
                                    
            else:
                # Single position zarr store
                dataset = store['0']
                fov_name = zarr_path.name
                
                # Get channel names
                if hasattr(store, 'channel_names'):
                    channel_names = store.channel_names
                else:
                    num_channels = dataset.shape[1]
                    channel_names = [f'Channel_{i}' for i in range(num_channels)]
                
                fov_results = []
                for c_idx, channel_name in enumerate(channel_names):
                    channel_data = dataset[0, c_idx]
                    result = analyze_fov_channel(channel_data, fov_name, channel_name)
                    fov_results.append(result)
                    total_processed += 1
                
                # Write results to CSV
                append_to_csv(fov_results, output_csv, write_header=True)
                click.echo(f"Processed single FOV {fov_name} ({len(fov_results)} channels)")
                    
    except Exception as e:
        click.echo(f"Error opening zarr store {zarr_path}: {e}")
        raise
        
    return total_processed


def append_to_csv(results: List[dict], output_path: Path, write_header: bool = False) -> None:
    """
    Append QC results to CSV file.
    """
    if not results:
        return
        
    fieldnames = ['fov_name', 'channel', 'blank', 'min', 'max', 'median']
    
    mode = 'w' if write_header else 'a'
    with open(output_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)


def combine_csv_files(csv_dir: Path, output_csv: Path) -> int:
    """
    Combine individual FOV CSV files into a single output CSV.
    """
    fieldnames = ['fov_name', 'channel', 'blank', 'min', 'max', 'median']
    total_rows = 0
    
    # Find all CSV files starting with 'qc_fov_'
    csv_files = list(csv_dir.glob('qc_fov_*.csv'))
    
    click.echo(f"Found {len(csv_files)} individual CSV files to combine.")
    
    if not csv_files:
        click.echo("No individual CSV files found to combine.")
        return 0
    
    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for csv_file in sorted(csv_files):
            try:
                with open(csv_file, 'r') as infile:
                    reader = csv.DictReader(infile)
                    for row in reader:
                        writer.writerow(row)
                        total_rows += 1
                        
                # Clean up individual file after combining
                csv_file.unlink()
                
            except Exception as e:
                click.echo(f"Warning: Error processing {csv_file}: {e}")
                continue
    
    return total_rows


def process_hcs_zarr_parallel(
    zarr_path: Path, 
    output_csv: Path,
    sbatch_filepath: str = None,
    local: bool = False,
    monitor: bool = True
) -> int:
    """
    Process an OME HCS zarr store in parallel using Slurm.
    
    Parameters
    ----------
    zarr_path : Path
        Path to the OME HCS zarr store
    output_csv : Path
        Path to final combined CSV file
    sbatch_filepath : str, optional
        Path to sbatch configuration file
    local : bool, optional
        Run locally instead of on Slurm
    monitor : bool, optional
        Monitor job progress
        
    Returns
    -------
    int
        Total number of FOV-channel combinations processed
    """
    # Create temporary directory for individual CSV files
    temp_csv_dir = output_csv.parent / "temp_qc_csvs"
    temp_csv_dir.mkdir(exist_ok=True)
    
    # Create slurm output directory
    slurm_out_path = output_csv.parent / "slurm_output"
    slurm_out_path.mkdir(exist_ok=True)
    
    # Collect all FOV information
    fov_jobs = []
    
    try:
        with open_ome_zarr(zarr_path, mode='r') as store:
            if hasattr(store, 'zgroup') and 'plate' in store.zattrs:
                # HCS store
                plate_metadata = store.zattrs['plate']
                
                for well_info in plate_metadata.get('wells', []):
                    well_path = well_info['path']
                    well_store = store[well_path]
                    
                    if 'well' in well_store.zattrs:
                        well_metadata = well_store.zattrs['well']
                        
                        for field_info in well_metadata.get('images', []):
                            field_path = field_info['path']
                            fov_jobs.append((well_path, field_path))
            else:
                # Single position zarr - not suitable for parallel processing
                click.echo("Single position zarr detected. Using sequential processing.")
                return process_hcs_zarr_store(zarr_path, output_csv)
                
    except Exception as e:
        click.echo(f"Error analyzing zarr structure: {e}")
        raise
    
    if not fov_jobs:
        click.echo("No FOVs found to process.")
        return 0
    
    click.echo(f"Found {len(fov_jobs)} FOVs to process in parallel.")
    
    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "qc-fovs",
        "slurm_mem_per_cpu": "8G",  # Increased memory
        "slurm_cpus_per_task": 16 if local else 2,  # Use 16 CPUs for local, 2 for SLURM
        "slurm_array_parallelism": 100,  # Reduced parallelism to avoid overwhelming system
        "slurm_time": 30,  # Increased timeout to 30 minutes per job
        "slurm_partition": "preempted",
    }
    
    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))
    
    # Choose cluster
    cluster = "slurm"
    if local:
        cluster = "local"
    if os.environ.get("CI") == "true":
        cluster = "debug"
    
    # Prepare and submit jobs
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)
    
    click.echo("Submitting SLURM jobs...")
    jobs = []
    
    with submitit.helpers.clean_env(), executor.batch():
        for well_path, field_path in fov_jobs:
            job = executor.submit(
                process_single_fov_for_qc,
                zarr_path,
                well_path,
                field_path,
                temp_csv_dir
            )
            jobs.append(job)
    
    # Log job IDs
    job_ids = [job.job_id for job in jobs]
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))
    
    # Monitor jobs if requested
    if monitor:
        click.echo("Monitoring job progress...")
        monitor_jobs(jobs, [f"{well_path}/{field_path}" for well_path, field_path in fov_jobs])
    
    # Check job completion status
    failed_jobs = 0
    completed_jobs = 0
    for job in jobs:
        try:
            result = job.result()  # This will raise an exception if the job failed
            completed_jobs += 1
        except Exception as e:
            failed_jobs += 1
            click.echo(f"Job failed: {e}")
    
    click.echo(f"Job completion status: {completed_jobs} completed, {failed_jobs} failed out of {len(jobs)} total")
    
    # Combine all individual CSV files
    click.echo("Combining individual CSV files...")
    total_processed = combine_csv_files(temp_csv_dir, output_csv)
    
    # Clean up temporary directory
    try:
        temp_csv_dir.rmdir()
    except OSError:
        click.echo(f"Warning: Could not remove temporary directory {temp_csv_dir}")
    
    return total_processed


@click.command("qc-fovs")
@click.option(
    "-i", "--input-zarr",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to OME HCS zarr store or OME zarr store"
)
@sbatch_filepath()
@local()
@monitor()
def qc_fovs_cli(input_zarr: Path, sbatch_filepath: str = None, local: bool = False, monitor: bool = True):
    """
    Analyze FOVs in an OME HCS zarr store to identify blank FOVs and compute quality metrics.
    
    This command iterates through all fields of view (FOVs) in an OME HCS zarr store,
    analyzes each channel, and identifies blank FOVs based on min/max values and NaN checks.
    
    A FOV is considered blank if:
    - Min and max values are the same (no variation)
    - Either min or max value is NaN
    - Max value is 0
    
    Output CSV contains: fov_name, channel, blank (T/F), min, max, median
    
    Example:
        biahub qc-fovs -i /path/to/data.zarr
    """
    
    click.echo(f"Analyzing FOVs in {input_zarr}")
    
    # Output CSV at the top level of the zarr store
    output_csv = input_zarr / "qc_fov.csv"
    
    # Suppress warnings during processing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Use parallel processing for HCS stores, sequential for single position
        try:
            with open_ome_zarr(input_zarr, mode='r') as store:
                is_hcs = hasattr(store, 'zgroup') and 'plate' in store.zattrs
        except Exception as e:
            click.echo(f"Error checking zarr structure: {e}")
            return
        
        if is_hcs and not local:
            # Parallel processing for HCS stores
            total_processed = process_hcs_zarr_parallel(
                input_zarr, output_csv, sbatch_filepath, local, monitor
            )
        else:
            # Sequential processing for single position or local mode
            if is_hcs:
                click.echo("Using sequential processing (local mode or single position).")
            total_processed = process_hcs_zarr_store(input_zarr, output_csv)
    
    if total_processed == 0:
        click.echo("No FOVs found to analyze.")
        return
    
    # Read the CSV to compute summary statistics
    results = []
    try:
        with open(output_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            results = list(reader)
    except Exception as e:
        click.echo(f"Error reading results CSV: {e}")
        return
    
    # Print summary statistics
    total_fovs = len(set(result['fov_name'] for result in results))
    total_channels = len(set(result['channel'] for result in results))
    blank_fov_channels = sum(1 for result in results if result['blank'] == 'True')
    
    click.echo(f"\nSummary:")
    click.echo(f"  CSV file: {output_csv}")
    click.echo(f"  Total FOVs analyzed: {total_fovs}")
    click.echo(f"  Total channels: {total_channels}")
    click.echo(f"  Total FOV-channel combinations: {len(results)}")
    click.echo(f"  Blank FOV-channel combinations: {blank_fov_channels}")
    if len(results) > 0:
        click.echo(f"  Blank percentage: {blank_fov_channels/len(results)*100:.1f}%")


if __name__ == "__main__":
    qc_fovs_cli()