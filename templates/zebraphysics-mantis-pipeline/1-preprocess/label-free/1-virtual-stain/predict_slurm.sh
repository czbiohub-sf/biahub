#!/bin/bash

#SBATCH --job-name=predict_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-72:00:00
#SBATCH --output=./slurm_output/%j.out
#SBATCH --constraint=a100|a6000|a40

# Load modules
module load anaconda/2022.05
conda activate viscy


positions=()
for p in $1/*/*/*; do
    positions+=($p)
done

log_dir="$(pwd)/logs/$SLURM_ARRAY_TASK_ID"
mkdir -p $log_dir
cd $log_dir

cat $3

viscy predict \
    -c $3 \
    --data.init_args.data_path ${positions[$SLURM_ARRAY_TASK_ID]} \
    --trainer.callbacks+=viscy.translation.predict_writer.HCSPredictionWriter \
    --trainer.callbacks.output_store=$2/$SLURM_ARRAY_TASK_ID.zarr \
    --trainer.default_root_dir=$log_dir