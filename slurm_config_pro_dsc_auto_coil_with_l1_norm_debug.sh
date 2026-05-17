#!/usr/bin/env bash
#
#SBATCH --job-name=DebugL1NormAutoCoil100
#SBATCH --output=l1_norm_coil100_pretrain_knn_debug.txt
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
python3 ./main_subspace_auto_gamma_with_l1_norm.py --data=coil100  --experiment_name=coil100_auto_gamma_l1_norm_rerun_debug --load_pretrain --seeds=[42,0,1,2,3,4,5,6,7,8,9]   >> l1_norm_coil100_pretrain_knn_debug_out.txt

