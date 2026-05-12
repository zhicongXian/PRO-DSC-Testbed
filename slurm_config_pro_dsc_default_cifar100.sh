#!/usr/bin/env bash
#
#SBATCH --job-name=100DeCIFAR
#SBATCH --output=cifar100_default.txt
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
python3 ./main_default_gamma.py --data=cifar100 --experiment_name=cifar100_default >> cifar100_default_out.txt

