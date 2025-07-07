#!/bin/bash

MODEL="gpt-4.1-mini"
# DATASET="/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/medqa_1_test_noabst.json"
# PERTURBED_DATASET="/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/perturbed_medqa_1_test_noabst.json"
MODEL_KEY="OPENAI_API_KEY"
DATASETS=("/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/medqa_1_test_noabst.json"
 "/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/medqa_1_test_randabst.json"
 "/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/amboss_alldiff_train_noabst.json"
 "/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/amboss_alldiff_train_randabst.json"
)

PERTURBED_DATASETS=("/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/perturbed_medqa_1_test_noabst.json"
 "/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/perturbed_medqa_1_test_randabst.json"
 "/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/perturbed_amboss_alldiff_train_noabst.json"
 "/Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/datasets/perturbed_amboss_alldiff_train_randabst.json"
 )

# for i in "${!DATASETS[@]}"; do
DATASET=${DATASETS[0]}
PERTURBED=${PERTURBED_DATASETS[0]}

echo "Running on dataset: $DATASET"
echo "Outputting perturbed dataset to: $PERTURBED"
python /Users/sushritayerra/work/abstention_01/med-llm-uncertainty-benchmark/quantify_uncertainty/dataset_scripts/create_perturbed_dataset.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --perturbed_dataset "$PERTURBED_DATASET" \
    --model_key "$MODEL_KEY"
# done