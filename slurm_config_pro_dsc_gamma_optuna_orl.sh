#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_optuna_orl
#SBATCH --output=optuna_orl.txt
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
python3 ./main_subspace_gamma_optuna_automl.py --data=orl --experiment_name=orl_optuna_automl_new --seeds=[42,0,1,2,3,4,5,6,7,8,9] >> rerun_gamma_sweep_orl_out.txt

