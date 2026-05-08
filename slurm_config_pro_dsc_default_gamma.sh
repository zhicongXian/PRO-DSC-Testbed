#!/usr/bin/env bash
#
#SBATCH --job-name=DefaultGamma_cifar10_mcr
#SBATCH --output=default_cifa10_mcr_data_dependent_regularization.txt
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
python3 ./main_default_gamma.py --data=cifar10-mcr --experiment_name=cifa10_mcr_data_default_gamma_208 --epo=5000 >> default_cifa10_mcr_data_dependent_regularization_out.txt

