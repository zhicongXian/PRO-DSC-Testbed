#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_elastic_net_regularizer_gamma_sweep
#SBATCH --array=0-9
#SBATCH --output=logs_coil100_gamma_%A_%a.out
#SBATCH --time=10-00:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

# debug info
hostname
which python3
nvidia-smi

env

# venv
source /home/wiss/xian/venvs/subspace_clustering_3_12/bin/activate
export BLAS=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3
export LAPACK=/usr/lib/x86_64-linux-gnu/lapack/liblapack.a
# pip install -U pip setuptools wheel
# train

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    start_gamma=10
else
    start_gamma=$((SLURM_ARRAY_TASK_ID * 100))
fi

end_gamma=$(((SLURM_ARRAY_TASK_ID + 1) * 100))

echo "Running start_gamma=${start_gamma}, end_gamma=${end_gamma}"


python3 ./main_subspace_gamma_sweep_with_elastic_net.py --data=coil100 --start_gamma=$start_gamma --end_gamma=$end_gamma --experiment_name="wandb_sweep_elastic_net_coil100_from{$start_gamma}_to{$end_gamma}" --seed=0

