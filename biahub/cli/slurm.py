from tqdm import tqdm
import submitit

def wait_for_jobs_to_finish(
    jobs: list[submitit.Job]
) -> None:
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
    for job in tqdm(submitit.helpers.as_completed(jobs), total=len(jobs), desc="Waiting for jobs to finish"):
        try:
            pass
        except Exception as e:
            print(f"Job {job.job_id} failed with exception: {e}")
