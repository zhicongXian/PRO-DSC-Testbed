#!/usr/bin/env bash
#
#SBATCH --job-name=C10GamSwe
#SBATCH --output=cifa10_gamma_sweep.txt
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
python3 ./main_gamma_sweep.py --data=cifar10 --experiment_name=wandb_sweep_cifar10_seed2 --seed=2 --epo=5000 >> cifa10_gamma_sweep_out.txt

