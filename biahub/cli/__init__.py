import os

# Delete SLURM_* environment variables
#
# Submitting biahub jobs from a non-login node will cause the newly submitted
# SLURM jobs to inherit from the local job's SLURM_* environment variables,
# which can lead to conflicting instructions.
for k in list(os.environ):
    if k.startswith("SLURM_"):
        os.environ.pop(k)
