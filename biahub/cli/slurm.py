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
