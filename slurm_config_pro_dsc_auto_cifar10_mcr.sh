#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_cifar10_mcr_auto_gamma
#SBATCH --output=cifar10_mcr_auto_gamma.txt
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
# old one: main_auto_gamma_effective_rank
python3 ./main_auto_gamma_with_hpo_final.py --data=cifar10-mcr --seeds=[42,1,2,3,4,5,6,7,8,9] --experiment_name=cifar10_mcr_auto_gamma_with_optuna   >> cifar10_mcr_auto_gamma_out.txt

