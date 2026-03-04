#!/usr/bin/env bash
#
#SBATCH --job-name=pro_cifar100
#SBATCH --output=pro_cifar_100.txt
#SBATCH --ntasks=1
#SBATCH --time=5-23:00:00
#SBATCH --gres=gpu:1

# debug info
hostname
which python3
nvidia-smi

env

# venv
source /home/wiss/xian/venvs/subspace_clustering_env/bin/activate
export BLAS=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3
export LAPACK=/usr/lib/x86_64-linux-gnu/lapack/liblapack.a
# pip install -U pip setuptools wheel
# train
python3 ./main_knn_enhanced_cont.py --data=cifar10-mcr --epo=5000 >> pro_cifar_100_out.txt
