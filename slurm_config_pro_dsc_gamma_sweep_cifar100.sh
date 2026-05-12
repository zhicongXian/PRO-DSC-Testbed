#!/usr/bin/env bash
#
#SBATCH --job-name=C100GamSwe
#SBATCH --output=cifa100_gamma_sweep.txt
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
python3 ./main_gamma_sweep.py --data=cifar100 --experiment_name=wandb_sweep_cifar100 --seed=1 --epo=5000 >> cifa100_gamma_sweep_out.txt

