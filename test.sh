#!/bin/bash
#SBATCH --job-name=mteb_test
#SBATCH --time=00:30:00
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
# source /var/scratch/tkl206/anaconda3/etc/profile.d/conda.sh
# conda activate myenv

# Run the test script
cd $HOME/rag-structure-comparison  # or wherever testTwo.py lives
python testTwo.py
