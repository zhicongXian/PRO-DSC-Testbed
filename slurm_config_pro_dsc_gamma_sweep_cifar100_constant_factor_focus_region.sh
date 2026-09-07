#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_constant_factor_sweep_cifar100_03_to_10
#SBATCH --array=0-10
#SBATCH --output=logs_cifar100_constant_factor_%A_%a.out
#SBATCH --time=10-00:00:00
#SBATCH --mem=64G
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



starts=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 )
ends=(0.4 0.5 0.6 0.7 0.8 5 0.9 1.0 )

start_constant_factor="${starts[$SLURM_ARRAY_TASK_ID]}"
end_constant_factor="${ends[$SLURM_ARRAY_TASK_ID]}"

echo "Range: ${start_constant_factor} to ${end_constant_factor}"


python3 ./main_auto_gamma_with_hpo_final_constant_factor_sweep.py --data=cifar100 --start_constant_factor=$start_constant_factor \
 --end_constant_factor=$end_constant_factor --experiment_name="wandb_constant_sweep_cifar100_from{$start_constant_factor}_to{$end_constant_factor}_beta_500" --seed=0

