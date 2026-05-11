#!/usr/bin/env bash
#
#SBATCH --job-name=42AutoCoil100
#SBATCH --output=coil100_pretrain_knn.txt
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
python3 ./main_subspace_auto_gamma.py --data=coil100 --load_pretrain --experiment_name=coil100_auto_gamma_seed42_round2 --seeds=[42]   >> coil100_pretrain_knn_out.txt

