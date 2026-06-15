#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_optuna_cifar10_mcr
#SBATCH --output=cifar10_mcr_optuna2.txt
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
python3 ./main_gamma_optuna_automl.py --data=cifar10-mcr --experiment_name=optuna_automl_new >> cifar10_mcr_optuna_out2.txt

