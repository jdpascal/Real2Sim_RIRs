#!/bin/bash -l

#SBATCH -p publicgpu
#SBATCH --nodes=1               # This needs to match --ddp-nodes
#SBATCH --ntasks-per-node=2     # This needs to match --ddp-devices-per-node
#SBATCH --gres=gpu:2            # Request N GPUs per machine

#SBATCH --mem=0
#SBATCH --time=0-20:00:00

# Load correct python and cuda modules
module load python/3.12.8
module load cuda/cuda-12.1

# Activate python environment
source venv/bin/activate

# Debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
export NCCL_SOCKET_IFNAME=^docker0,lo

# export CUDA_VISIBLE_DEVICES=1
# export TORCH_USE_CUDA_DSA

# Run main with training arguments
srun python main.py \
    --config configs/ve/MultiRIR_ncsnpp_SBVE.py \
    --mode train \
    --eval-folder eval \
    --workdir exp/ve/MultiRIR_ncsnpp_continuous \
    --ddp-nodes 1\
    --ddp-devices-per-node 2
