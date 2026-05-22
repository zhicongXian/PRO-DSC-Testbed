#!/usr/bin/env bash
#
#SBATCH --job-name=PRO_DSC_optuna_si_mnist
#SBATCH --output=mnist_optuna.txt
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
python3 ./main_gamma_optuna_automl.py --data=mnist --experiment_name=mnist_optuna_automl >> mnist_optuna_out.txt

