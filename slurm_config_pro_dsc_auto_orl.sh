#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_orl_auto_gamma
#SBATCH --output=orl_auto.txt
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
# previously l1 norm main_subspace_auto_gamma_l1_rerun.py
python3 ./main_subspace_auto_gamma_with_l1_norm.py --data=orl --experiment_name=orl_auto_gamma_with_si --seeds=[42,1,2,3,4,5,6,7,8,9] >> orl_auto_out.txt

