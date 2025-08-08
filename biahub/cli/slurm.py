import os

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import click
import pandas as pd
import submitit

from tqdm import tqdm


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


def check_job_logs(
    log_dir: Union[str, Path],
    job_ids: Optional[list[Union[str, int]]] = None,
    output_dir: Union[str, Path] = None,
) -> pd.DataFrame:
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
        job_ids = sorted(
            [
                fname.name.replace("_log.out", "").replace("_log.err", "")
                for fname in log_dir.glob("*_log.out")
            ]
        )

    for job_id in tqdm(job_ids, desc="Analyzing SLURM logs"):
        job_id = str(job_id)

        all_files = os.listdir(log_dir)
        matching_files = [f for f in all_files if f"{job_id}" in Path(f).stem]

        files_with_out = [f for f in matching_files if f.endswith("_log.out")]

        for file_out in files_with_out:
            log_out = log_dir / file_out
            log_err = log_dir / file_out.replace("_log.out", "_log.err")
            id = Path(log_err).stem.replace("_0_log", "")  # safer

            if not log_out.exists() and not log_err.exists():
                status = "LOG NOT FOUND"
            else:
                try:
                    logs_combined = ""
                    if log_out.exists():
                        logs_combined += log_out.read_text().lower()
                    if log_err.exists():
                        logs_combined += log_err.read_text().lower()

                    if (
                        "job completed successfully" in logs_combined
                        or "exiting after successful completion" in logs_combined
                    ):
                        status = "SUCCESS"
                    else:
                        status = "FAILED"
                except Exception as e:
                    status = f"ERROR READING LOG: {e}"
                records.append((id, status))

    df = pd.DataFrame(records, columns=["Job_ID", "Status"])
    df.sort_values(by="Status", ascending=True, inplace=True)

    # Count summary
    total = len(df)
    counts = df["Status"].value_counts()
    summary = {
        "Total": total,
        "Success": counts.get("SUCCESS", 0),
        "Failed": sum(counts[s] for s in counts.index if "FAIL" in s or "ERROR" in s),
        "Not found": counts.get("LOG NOT FOUND", 0),
    }
    # add summary row
    summary_row = pd.DataFrame([summary])
    df = pd.concat([summary_row, df], ignore_index=True)

    # Timestamped output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"job_status_report_{timestamp}.csv"
    output_path = Path(output_dir) / output_filename

    # Save report
    df.to_csv(output_path, index=False)
    click.echo("...............................................")
    click.echo(f"Total: {summary['Total']}")
    click.echo(f"Success: {summary['Success']}")
    click.echo(f"Failed: {summary['Failed']}")
    click.echo("...............................................")
    click.echo(f"Summary saved to: {output_path}")

    return df


@click.command("check-job-logs")
@click.option(
    "--log-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to SLURM log directory.",
)
@click.option(
    "--job-ids",
    type=str,
    multiple=True,
    help="Specific job IDs to check. If not provided, all jobs in the log directory will be checked.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Output directory for the report CSV file.",
)
def check_job_logs_cli(log_dir: Path, job_ids: tuple, output_dir: Optional[Path]):
    """
    CLI for checking SLURM job logs and saving a filtered CSV report with non-success statuses.
    """
    job_id_list = list(job_ids) if job_ids else None
    check_job_logs(
        log_dir, job_ids=job_id_list, output_dir=output_dir if output_dir else log_dir
    )


if __name__ == "__main__":
    check_job_logs_cli()
