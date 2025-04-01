#!/bin/bash -l

#SBATCH -n 64                # 64 cœurs
#SBATCH --mem=0
#SBATCH --time=0-10:00:00

# Load correct python and cuda modules
module load python/3.12.8
module load cuda/cuda-12.1

# Activate python environment
source venv/bin/activate

python RIR_generation/generate_rirs.py