#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 2  # Number of CPU cores
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00
#SBATCH -o /project/pi_hongyu_umass_edu/zonghai/abstention/sushrita/med-llm-uncertainty-benchmark/outputs/slurm/generate_id-%j.out  # Specify where to save terminal output, %j = job ID will be filled by slurm

MODEL="/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/models/Phi-4-mini"
DATASET="/project/pi_hongyu_umass_edu/zonghai/abstention/sushrita/med-llm-uncertainty-benchmark/data/sample_test.json"
PROMPT="shared"
FEW_SHOT=0
COT=0
VERSION="v1"
CAL_RATIO=0.3
ALPHA=0.1
MODEL_KEY=""
DATASET_TYPE="NoAbst"

python /project/pi_hongyu_umass_edu/zonghai/abstention/sushrita/med-llm-uncertainty-benchmark/scripts/python/launch_experiment.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --prompt "$PROMPT" \
    --few_shot "$FEW_SHOT" \
    --cot "$COT" \
    --cal_ratio "$CAL_RATIO" \
    --alpha "$ALPHA" \
    --version "$VERSION" \
    --model_key "$MODEL_KEY" \
    --dataset_type "$DATASET_TYPE"