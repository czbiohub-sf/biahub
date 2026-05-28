"""Toy processing functions demonstrating submitit DebugExecutor failure modes.

Two modes:
  --mode success  : completes normally, but logging is split by DebugExecutor
  --mode fail     : raises an exception, triggering pdb.post_mortem() hang

Each mode can run with or without submitit wrapping (--bypass-submitit).
"""

import argparse
import logging
import os
import sys
logger = logging.getLogger(__name__)


def toy_transform(value: float, should_fail: bool = False) -> float:
    """Simulates a per-position processing step."""
    logger.info("toy_transform starting, pid=%d", os.getpid())
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    logger.info("NXF_TASK_WORKDIR=%s", os.environ.get("NXF_TASK_WORKDIR", "<unset>"))

    if should_fail:
        raise RuntimeError(
            "Simulated processing error (e.g. OOM, corrupted zarr, bad config)"
        )

    result = value * 2.0
    logger.info("toy_transform done, result=%f", result)
    return result


def run_with_submitit_debug(should_fail: bool) -> None:
    """Wraps toy_transform in submitit DebugExecutor — the problematic path."""
    import submitit

    executor = submitit.AutoExecutor(folder="./submitit_logs", cluster="debug")
    executor.update_parameters(timeout_min=5)

    job = executor.submit(toy_transform, value=21.0, should_fail=should_fail)
    logger.info("Submitted debug job: %s", job.job_id)

    # This triggers synchronous execution inside DebugJob.results()
    # On failure: drops into pdb.post_mortem() → hangs forever
    result = job.results()
    logger.info("Job result: %s", result)


def run_direct(should_fail: bool) -> None:
    """Calls toy_transform directly — the recommended path for Nextflow."""
    result = toy_transform(value=21.0, should_fail=should_fail)
    logger.info("Direct result: %f", result)


def main():
    parser = argparse.ArgumentParser(description="Toy submitit debug demo")
    parser.add_argument(
        "--mode",
        choices=["success", "fail"],
        required=True,
        help="'success' completes normally; 'fail' raises an exception",
    )
    parser.add_argument(
        "--bypass-submitit",
        action="store_true",
        help="Call the function directly instead of wrapping in DebugExecutor",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    should_fail = args.mode == "fail"

    if args.bypass_submitit:
        logger.info("Running DIRECT (no submitit)")
        try:
            run_direct(should_fail)
        except Exception:
            logger.exception("Direct call failed cleanly — Nextflow sees exit code 1")
            sys.exit(1)
    else:
        logger.info("Running via submitit DebugExecutor")
        run_with_submitit_debug(should_fail)


if __name__ == "__main__":
    main()
