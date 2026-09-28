#!/bin/bash
# Usage: sbatch slurm/cpu.sbatch <command>
#SBATCH --job-name=plb-cpu
#SBATCH --partition=triton
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source slurm/_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$@"
