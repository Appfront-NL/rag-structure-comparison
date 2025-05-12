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

# Activate conda
source /var/scratch/<your-username>/anaconda3/bin/activate
conda activate myenv  # Replace with your env name if different

# Run the test
mkdir -p $HOME/mteb_test/results
cd $HOME/mteb_test
python testTwo.py
