#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_constant_factor_sweep_cifar100_0.28_to_0.26
#SBATCH --output=logs_cifar100_constant_factor2.out
#SBATCH --time=10-00:00:00
#SBATCH --mem=128G
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






python3 ./main_auto_gamma_with_hpo_final_constant_factor_sweep.py --data=cifar100 --start_constant_factor=0.18 \
 --end_constant_factor=0.26 --experiment_name="wandb_constant_sweep_cifar100_from_018_to_026" --seed=0

