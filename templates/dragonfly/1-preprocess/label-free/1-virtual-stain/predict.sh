#!/bin/bash

module load anaconda/2022.05
conda activate viscy

# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

CURRENT_DIR=$(pwd)

INPUT_DATASET=$(pwd)/../0-reconstruct/${DATASET}.zarr
OUTPUT_FOLDER=$(pwd)/${DATASET}
CONFIG_FILE=$(pwd)/predict.yml

positions=($INPUT_DATASET/*/*/*)

mkdir -p "$OUTPUT_FOLDER"
ARRAY_JOB_ID=$(sbatch --parsable --array=0-$((${#positions[@]}-1))%36 predict_slurm.sh $INPUT_DATASET $OUTPUT_FOLDER $CONFIG_FILE)

# Merge after the previous job is done
sbatch --parsable --dependency=afterok:$ARRAY_JOB_ID ./combine.sh "$OUTPUT_FOLDER" "$INPUT_DATASET" "${OUTPUT_FOLDER}.zarr"
