import submitit
import click
from biahub.cli.parsing import output_dirpath
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from typing import Optional, Union

def wait_for_jobs_to_finish(jobs: list[submitit.Job]) -> None:
    """
    Wait for SLURM jobs to finish using a progress bar with tqdm.

    Parameters
    ----------
    jobs : list
        A list of submitit Job objects that represent the SLURM jobs to wait for.
    Returns
    -------
    None
    """
    for job in tqdm(
        submitit.helpers.as_completed(jobs), total=len(jobs), desc="Waiting for jobs to finish"
    ):
        try:
            pass  # as_completed polls every 10 seconds by default, so we don't need to do anything here
        except Exception as e:
            print(f"Job {job.job_id} failed with exception: {e}")


def check_job_logs(log_dir: Union[str, Path], job_ids: Optional[list[Union[str, int]]] = None, output_dir: Union[str, Path], = None) -> pd.DataFrame:
    """
    Check SLURM logs for specific job IDs or all jobs in a folder,
    reporting statuses and saving a filtered CSV with non-successful jobs.

    Parameters
    ----------
    log_dir : str or Path
        Path to the SLURM log directory.
    job_ids : list[str or int], optional
        Specific job IDs to check. If None, will scan all *_log.out files in the directory.

    Returns
    -------
    df_filtered : pd.DataFrame
        DataFrame with only non-success job statuses.
    """
    log_dir = Path(log_dir)
    records = []

    if job_ids is None:
        job_ids = sorted([
            fname.name.replace("_log.out", "").replace("_log.err", "")
            for fname in log_dir.glob("*_log.out")
        ])

    for job_id in tqdm(job_ids, desc="Analyzing SLURM logs"):
        job_id = str(job_id)
        log_out = log_dir / f"{job_id}_log.out"
        log_err = log_dir / f"{job_id}_log.err"

        if not log_out.exists() and not log_err.exists():
            status = "LOG NOT FOUND"
        else:
            try:
                logs_combined = ""
                if log_out.exists():
                    logs_combined += log_out.read_text().lower()
                if log_err.exists():
                    logs_combined += log_err.read_text().lower()

                if "job completed successfully" in logs_combined or "exiting after successful completion" in logs_combined:
                    status = "SUCCESS"
                elif "error" in logs_combined or "fail" in logs_combined:
                    status = "FAILED"
                else:
                    status = "UNKNOWN"
            except Exception as e:
                status = f"ERROR READING LOG: {e}"

        records.append((job_id, status))

    df = pd.DataFrame(records, columns=["job_id", "status"])

    # Count summary
    total = len(df)
    counts = df["status"].value_counts()
    summary = {
        "Total": total,
        "Success": counts.get("SUCCESS", 0),
        "Failed": sum(counts[s] for s in counts.index if "FAIL" in s or "ERROR" in s),
        "Not found": counts.get("LOG NOT FOUND", 0),
        "Unknown": total - counts.get("SUCCESS", 0) - counts.get("LOG NOT FOUND", 0) - sum(counts[s] for s in counts.index if "FAIL" in s or "ERROR" in s)
    }

    # Filter out successes
    df_filtered = df[df["status"] != "SUCCESS"]

    # Save report
    output_path = output_dir / "job_status_report.csv"
    df_filtered.to_csv(output_path, index=False)

    # Append summary
    with open(output_path, "a") as f:
        f.write("\nSummary\n")
        for key, value in summary.items():
            f.write(f"{key},{value}\n")

    # Print
    print(f"\nSLURM Job Report (excluding SUCCESS):")
    print(df_filtered.to_string(index=False))
    print(f"\nSummary: {summary}")
    print(f"Report saved to: {output_path}")

    return df_filtered



@click.command("check-job-logs")
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--job-ids", "-j", multiple=True, help="Specific job IDs to check. If not provided, all jobs in the log directory will be checked.")
@click.opition("--output-dir", "-o", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None, help="Output directory for the report CSV file.")
def check_job_logs_cli(
    log_dir: Path,
    job_ids: Optional[list[Union[str, int]]] = None,
    output_dir:Path = None) -> None:
    
    """Check SLURM job logs for specific job IDs or all jobs in a folder,
    reporting statuses and saving a filtered CSV with non-successful jobs.
    """
    
    check_job_logs(log_dir, job_ids=job_ids if job_ids else None, output_dir=log_dir if not output_dir else output_dir)
    

if __name__ == "__main__":
    check_job_logs_cli()
