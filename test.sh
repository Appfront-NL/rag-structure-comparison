#!/bin/bash
#SBATCH --job-name=mteb_cpu_test
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p longq         # or defq if you're running during night/weekend
#SBATCH --output=slurm-%j.out

# Activate conda (no GPU needed)
source /var/scratch/tkl206/anaconda3/bin/activate
conda activate myenv  # use your actual conda env name

# Run the test
mkdir -p $HOME/mteb_test/results
cd $HOME/mteb_test
python testTwo.py
