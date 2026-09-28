#!/bin/bash
# Usage: sbatch slurm/cpu.sbatch make data baselines test
#SBATCH --job-name=plb-cpu
#SBATCH --partition=triton
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source slurm/_env.sh
"$@"
