#!/bin/bash
#SBATCH --job-name=vr-qwenvl
#SBATCH --partition=interactive
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --account=aikyamlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=26G
#SBATCH --time=04:00:00
#SBATCH --output=data/db/model_server.%j.log
# Qwen2.5-VL-7B-Instruct model server for the viewer. Submit from the repo root:
#   sbatch scripts/model_server.sbatch
# Then watch data/db/model_server.<jobid>.log; the backend picks up data/db/model_server.url by itself.
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"
mkdir -p data/db
exec /home/rbc5xp/.conda/envs/nnedit/bin/python scripts/model_server.py --port 8100
