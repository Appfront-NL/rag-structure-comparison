#!/bin/bash
#SBATCH --job-name=WikipediaRerankingMultilingual_BAAI_bge_m3
#SBATCH --time=09:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH --output=logs/WikipediaRerankingMultilingual_BAAI_bge_m3-%j.out

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
python python_tasks/WikipediaRerankingMultilingual___BAAI_bge_m3.py
