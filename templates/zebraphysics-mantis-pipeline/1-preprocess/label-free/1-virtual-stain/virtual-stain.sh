#!/bin/bash

module load anaconda
conda activate biahub

# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub virtual-stain \
    --input-position-dirpaths ../0-reconstruct/${DATASET}.zarr/*/*/* \
    --output-dirpath ${DATASET}.zarr \
    --predict-config-filepath predict.yml \
    --preprocess-config-filepath preprocess.yml \
    --path-viscy-env /hpc/mydata/taylla.theodoro/anaconda/2022.05/x86_64/envs/viscy \
    --run-mode all \
