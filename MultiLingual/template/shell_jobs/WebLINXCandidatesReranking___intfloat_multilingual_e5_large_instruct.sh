#!/bin/bash
#SBATCH --job-name=WebLINXCandidatesReranking_intfloat_multilingual_e5_large_instruct
#SBATCH --time=09:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH --output=logs/WebLINXCandidatesReranking_intfloat_multilingual_e5_large_instruct-%j.out

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

# Run script
cd $HOME/rag-structure-comparison/MultiLingual
python python_tasks/WebLINXCandidatesReranking___intfloat_multilingual_e5_large_instruct.py
