#!/usr/bin/env bash
#
#SBATCH --job-name=pro_dsc_cifar100_auto_gamma_elastic_net
#SBATCH --output=cifar100_auto_gamma.txt
#SBATCH --ntasks=1
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

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
python3 ./main_auto_gamma_with_hpo_final_elastic_net.py --data=cifar100  --experiment_name=cifar100_auto_gamma_with_optuna_elastic_net   >> cifar100_auto_gamma_out.txt

