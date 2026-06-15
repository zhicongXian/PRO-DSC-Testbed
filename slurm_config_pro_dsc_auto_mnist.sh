#!/usr/bin/env bash
#
#SBATCH --job-name=AutoMNIST
#SBATCH --output=mnist_auto_gamma.txt
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
# old main_auto_gamma
python3 ./main_auto_gamma_gradient_balancing_with_hpo.py --data=mnist  --experiment_name=mnist_auto_gamma_new   >> mnist_auto_gamma_out.txt

