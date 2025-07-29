import subprocess
import time

from tqdm import tqdm


def wait_for_jobs_to_finish(
    job_ids: list[str],
    sleep_time: int = 60,
    use_sacct: bool = False,
) -> None:
    """
    Wait for SLURM jobs to finish using a progress bar with tqdm.

    Parameters
    ----------
    job_ids : list[str]
        List of SLURM job IDs.
    sleep_time : int
        Seconds to wait between checks.
    use_sacct : bool
        If True, uses `sacct`; otherwise uses `squeue`.

    Returns
    -------
    None
    """

    unfinished = set(job_ids)
    pbar = tqdm(
        total=len(unfinished),
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} Jobs Remaining',
        leave=True,
    )

    while unfinished:
        if use_sacct:
            time.sleep(sleep_time)
            still_running = set()

            for job_id in unfinished:
                result = subprocess.run(
                    [
                        "sacct",
                        "-j",
                        job_id,
                        "--format=JobID,State",
                        "--parsable2",
                        "--noheader",
                    ],
                    stdout=subprocess.PIPE,
                    text=True,
                )
                states = set()
                for line in result.stdout.strip().splitlines():
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        _, state = parts
                        states.add(state)

                if any(
                    s
                    in {
                        "RUNNING",
                        "PENDING",
                        "COMPLETING",
                        "CONFIGURING",
                        "REQUEUED",
                        "RESIZING",
                        "SUSPENDED",
                    }
                    for s in states
                ):
                    still_running.add(job_id)

            newly_finished = unfinished - still_running
            pbar.update(len(newly_finished))
            unfinished = still_running

        else:
            time.sleep(sleep_time)

            result = subprocess.run(
                ["squeue", "--noheader", "--format=%i"], stdout=subprocess.PIPE, text=True
            )
            running_ids = set(result.stdout.strip().splitlines())
            still_running = unfinished & running_ids
            newly_finished = unfinished - still_running

            pbar.update(len(newly_finished))
            unfinished = still_running

        pbar.set_postfix_str(f"{len(unfinished)} remaining")

    pbar.close()
