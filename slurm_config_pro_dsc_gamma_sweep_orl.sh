#!/usr/bin/env bash
#
#SBATCH --job-name=GamSwORL
#SBATCH --output=gamma_sweep_orl.txt
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
python3 ./main_subspace_gamma_sweep.py --data=orl --experiment_name=wandb_sweep_orl_seed1 --seed=1 >> gamma_sweep_orl_out.txt

