#!/bin/bash
#SBATCH --job-name=vr-train
#SBATCH --partition=interactive
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --account=aikyamlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=data/db/train.%j.log
# Train the region classifier on a GPU. Submit from the repo root: sbatch training/train.sbatch
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"
exec /home/rbc5xp/.conda/envs/nnedit/bin/python training/train.py "$@"
