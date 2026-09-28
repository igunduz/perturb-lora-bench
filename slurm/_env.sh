# Sourced by sbatch scripts: caches and env in scratch.
SCRATCH_ROOT=/icbb_triton/scratch/$USER
export CACHE_DIR_HELICAL_PREFIX=$SCRATCH_ROOT
export HF_HOME=$SCRATCH_ROOT/.cache/huggingface
export WANDB_DIR=$SCRATCH_ROOT/wandb
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

eval "$(conda shell.bash hook)"
conda activate $SCRATCH_ROOT/envs/plb
mkdir -p logs
