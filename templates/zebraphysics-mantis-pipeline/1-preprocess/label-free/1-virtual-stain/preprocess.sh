#!/bin/bash

#SBATCH --job-name=VSpreprocess
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-48:00:00

module load anaconda/latest
conda activate viscy


# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi


viscy preprocess \
  --data_path ../0-reconstruct/${DATASET}.zarr \
  --channel_names -1 \
  --num_workers 32 \
  --block_size 32

