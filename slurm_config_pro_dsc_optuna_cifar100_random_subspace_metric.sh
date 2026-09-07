#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_random_optuna_cifar100_subspace_metric
#SBATCH --output=cifar100_random_automl.txt
#SBATCH --ntasks=1
#SBATCH --time=10-00:00:00
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
python3 ./main_gamma_optuna_random_with_projected_subspace.py --data=cifar100  --seeds=[11,12,13,14,15,16,17,18,19,20] --experiment_name=cifar100_random_automl_subspace_metric   >> cifar100_random_automl_out.txt

