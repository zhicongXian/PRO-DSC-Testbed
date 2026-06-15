#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_optuna_cifar100
#SBATCH --output=optuna_cifar100_auto_gamma.txt
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
python3 ./main_gamma_optuna_automl.py --data=cifar100  --experiment_name=cifar100_optuna_automl_new   >> optuna_cifar100_auto_gamma_out.txt

