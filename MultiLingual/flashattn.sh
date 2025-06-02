#!/bin/bash
#SBATCH --job-name=install-flashattn
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH --output=install-%j.out

# Load modules
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3


export CUDA_HOME=/cm/shared/modulefiles/cuda12.3/toolkit/12.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


# Activate conda
source /var/scratch/tkl206/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Install flash-attn with correct CUDA env
pip install --no-cache-dir --force-reinstall flash-attn
