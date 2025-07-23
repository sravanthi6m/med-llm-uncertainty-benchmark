#!/bin/bash

MODEL="gpt-4.1-nano"
MODEL_NAME="gpt-4.1-nano"
DATASET="/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/sample_dataset.json"
PROMPT="shared"
FEW_SHOT=1
COT=0
VERSION="v1"
CAL_RATIO=0.3
ALPHA=0.1
MODEL_KEY="OPENAI_API_KEY"
DATASET_TYPE="Test"

python /Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/scripts/python/launch_experiment.py \
    --model "$MODEL" \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --prompt "$PROMPT" \
    --few_shot "$FEW_SHOT" \
    --cot "$COT" \
    --cal_ratio "$CAL_RATIO" \
    --alpha "$ALPHA" \
    --version "$VERSION" \
    --model_key "$MODEL_KEY" \
    --dataset_type "$DATASET_TYPE"