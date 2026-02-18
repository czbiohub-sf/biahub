import os
import argparse

# Predefined list of directories to check when --all is passed without arguments
PREDEFINED_LOG_DIRS = [
    "1-preprocess/label-free/0-reconstruct/slurm_output",
    "1-preprocess/label-free/1-virtual-stain/slurm_output",
    "1-preprocess/label-free/2-stabilize/phase/slurm_output",
    "1-preprocess/label-free/2-stabilize/virtual-stain/slurm_output",
    "1-preprocess/label-free/3-track/slurm_output",
    "1-preprocess/light-sheet/deconvolved/0-decon/slurm_output",
    "1-preprocess/light-sheet/deconvolved/1-deskew/slurm_output",
    "1-preprocess/light-sheet/deconvolved/2-stabilize-register/slurm_output",
    "1-preprocess/light-sheet/raw/0-deskew/slurm_output",
    "1-preprocess/light-sheet/raw/1-register/slurm_output",
    "1-preprocess/light-sheet/raw/2-stabilize/slurm_output",
  
]

def check_slurm_logs(log_dir):
    """Check SLURM log files in a directory for success or failure keywords."""
    summary = []

    for fname in os.listdir(log_dir):
        if fname.endswith("_log.out"):
            job_base = fname.replace("_log.out", "")
            out_path = os.path.join(log_dir, fname)

            try:
                with open(out_path) as f:
                    out_content = f.read().lower()

                if "error" in out_content or "fail" in out_content:
                    status = "FAILED"
                else:
                    status = "SUCCESS"

            except Exception as e:
                status = f"ERROR READING FILE: {e}"

            summary.append((job_base, status))

    print(f"\nJob Summary for {log_dir}:")
    count_failed = 0
    for job, status in sorted(summary):
        if status == "FAILED":
            count_failed += 1
            print(f"{job}: {status}")
    print(f"Total failed jobs: {count_failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check SLURM *_log.out files in one or more directories for 'error' or 'fail' status."
    )
    
    # Option 1: Check a single directory
    parser.add_argument(
        "--log_dir", type=str,
        help="Path to a single directory containing SLURM *_log.out files."
    )
    
    # Option 2: Check multiple directories
    parser.add_argument(
        "--all", nargs="*", metavar="DIR",
        help="List of directories to check. If omitted, a predefined list will be used."
    )

    args = parser.parse_args()

    # If --all is used (with or without paths)
    if args.all is not None:
        # Use passed paths or fallback to predefined list
        log_dirs = args.all if args.all else PREDEFINED_LOG_DIRS
        for dir_path in log_dirs:
            if os.path.isdir(dir_path):
                check_slurm_logs(dir_path)
            else:
                print(f"Warning: {dir_path} is not a valid directory.")

    # If --log_dir is used
    elif args.log_dir:
        check_slurm_logs(args.log_dir)

    # If nothing is provided
    else:
        parser.error("You must specify either --log_dir or --all")


"""
USAGE:

# Check a single log directory
$ python check_logs.py --log_dir /path/to/logs

# Check multiple directories provided in the command
$ python check_logs.py --all /path/to/logs1 /path/to/logs2

# Check a predefined list of directories
$ python check_logs.py --all
"""
