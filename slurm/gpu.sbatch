#!/bin/bash
# Usage: sbatch slurm/gpu.sbatch <command>
#SBATCH --job-name=plb-gpu
#SBATCH --partition=triton-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source slurm/_env.sh
nvidia-smi
"$@"
