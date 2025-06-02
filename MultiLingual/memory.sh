#!/bin/bash
#SBATCH --job-name=memoryCheck
#SBATCH --time=14:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out

# Load GPU modules
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

which nvidia-smi
