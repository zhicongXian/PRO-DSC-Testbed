#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_optuna_hopkins155
#SBATCH --output=hopkins155_optuna.txt
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
python3 ./main_gamma_optuna_automl2_trajectory_embedding.py --data=trajectory_embedding --experiment_name=hopkins155_optuna_automl_new >> hopkins155_optuna_out.txt

