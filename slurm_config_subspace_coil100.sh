#!/usr/bin/env bash
#
#SBATCH --job-name=CoCoilPro
#SBATCH --output=subspace_coil.txt
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
# export PYTHONPATH="${PYTHONPATH}:/home/wiss/xian/venvs/subspace_clustering_env/bin/python3.9/site-packages:/usr/lib/python3.9/site-packages"
# pip install -U pip setuptools wheel
# train
python3 ./main_subspace_coil100.py --data=coil100 --bs=120 --load_pretrain  >> subspace_coil_out.txt
