#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_elastic_net_regularizer_auto_gamma_coil100
#SBATCH --output=coil100_pretrain_knn.txt
#SBATCH --time=10-00:00:00
#SBATCH --mem=128G
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


python3 ./main_subspace_auto_gamma_elastic_net_hpo.py --data=coil100 --seeds=[42,0,1,2,3,4,5,6,7,8,9] --experiment_name=auto_gamma_coil100_elastic_net2

