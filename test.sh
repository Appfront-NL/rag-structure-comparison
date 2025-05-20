#!/bin/bash
#SBATCH --job-name=mteb_test
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out

cd $HOME/rag-structure-comparison
python testTwo.py
