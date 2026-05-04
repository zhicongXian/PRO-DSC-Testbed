#!/usr/bin/env bash
#
#SBATCH --job-name=DefaultCoil100
#SBATCH --output=default_subspace_coil.txt
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
python3 ./main_subspace.py --data=coil100  >> default_subspace_coil_out.txt
