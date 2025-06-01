#!/bin/bash
#SBATCH --job-name=large_MM
#SBATCH --time=56:15:00
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



# Load Conda from scratch installation
source /var/scratch/tkl206/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Redirect Hugging Face & datasets cache to scratch
export TRANSFORMERS_CACHE=/var/scratch/tkl206/hf_cache
export HF_DATASETS_CACHE=/var/scratch/tkl206/hf_cache
mkdir -p /var/scratch/tkl206/hf_cache

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True

# Run your script
cd $HOME/rag-structure-comparison/MultiModal
python multiModalRun.py
