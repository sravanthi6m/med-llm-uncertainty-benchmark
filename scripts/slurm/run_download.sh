#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 2  # Number of CPU cores
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00
#SBATCH -o /project/pi_hongyu_umass_edu/zonghai/abstention/sushrita/med-llm-uncertainty-benchmark/outputs/slurm/generate_id-%j.out  # Specify where to save terminal output, %j = job ID will be filled by slurm

python /project/pi_hongyu_umass_edu/zonghai/abstention/sushrita/med-llm-uncertainty-benchmark/quantify_uncertainty/models/download_model.py