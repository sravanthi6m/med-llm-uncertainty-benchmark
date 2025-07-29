#!/usr/bin/env bash

# =================================================================================
# Script for hyperparameter tuning jobs to find the optimal k values.
# =================================================================================


DATA_DIR="/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/data"
LOG_DIR="/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/outputs/slurm_master_tuning"
mkdir -p "$LOG_DIR"

DATE_TAG=$(date +"%Y%m%d-%H%M")
TRACKER_FILENAME="run_tracker_tuning_${DATE_TAG}.csv"

###########################
MODELS_FOR_TUNING=(
    # "Qwen3-06B"
    # "Llama-32-1B-Instruct"
    # "medgemma-4b-it"
    # "Phi-4-mini"
    "Llama-31-8B-Instruct"
    # "Qwen25-14B-Instruct"
    #"medgemma-27b-text-it"
)

DATASETS=("medqa" "amboss")

OG_K_VALUES_TO_TUNE=(1 2 3 4 5)
PERTURBED_K_VALUES_TO_TUNE=()
ABST_TYPES=("noabst" "randabst")
###########################
EMBEDDING_MODEL="text-embedding-ada-002"

for model in "${MODELS_FOR_TUNING[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        
        GPUS=4; MEM="200GB"; TIME="1-00:00:00"
        #CONSTRAINT="vram40"
        CONSTRAINT="[a40|v100|a100]"
        PARTITION="gpu"
        if [[ "$model" == *70B* ]]; then
            GPUS=5; MEM="250GB"; TIME="2-00:00:00"
        elif [[ "$model" == *27b* || "$model" == *32B* ]]; then
            GPUS=4; MEM="200GB"; TIME="1-12:00:00"
        fi
        
        for k in "${OG_K_VALUES_TO_TUNE[@]}"; do
            for abst_type in "${ABST_TYPES[@]}"; do
                JOB_NAME="TUNING_${model}_${dataset}_k${k}_og_${abst_type}"
                
                if [ "$dataset" == "medqa" ]; then
                    TEST_FILE="${DATA_DIR}/val_100_medqa_1_${abst_type}.json"
                    #TEST_FILE="${DATA_DIR}/val_30_medqa_1_${abst_type}.json" ####
                    FEW_SHOT_POOL="${DATA_DIR}/few_shot_pool_medqa_1_train_${abst_type}.json"
                else
                    TEST_FILE="${DATA_DIR}/val_100_amboss_alldiff_${abst_type}.json"
                    FEW_SHOT_POOL="${DATA_DIR}/few_shot_pool_amboss_train_test_alldiff_${abst_type}.json"
                fi

                sbatch --nodes=1 --partition=${PARTITION} --gpus=${GPUS} --mem=${MEM} --time=${TIME} --constraint=${CONSTRAINT} \
                       --job-name="${JOB_NAME}" \
                       --output="${LOG_DIR}/${JOB_NAME}-%j-${DATE_TAG}.out" \
                       --error="${LOG_DIR}/${JOB_NAME}-%j-${DATE_TAG}.err" \
                       ./scripts/submit_job.sh \
                         --model "$model" \
                         --dataset "$dataset" \
                         --k "$k" \
                         --abst_type "$abst_type" \
                         --few_shot_pool "$FEW_SHOT_POOL" \
                         --test_file "$TEST_FILE" \
                         --embedding_model "$EMBEDDING_MODEL" \
                         --tracker_file "$TRACKER_FILENAME"
            done
        done
        
        for k in "${PERTURBED_K_VALUES_TO_TUNE[@]}"; do
            for abst_type in "${ABST_TYPES[@]}"; do
                JOB_NAME="TUNING_${model}_${dataset}_k${k}_perturbed_${abst_type}"

                if [ "$dataset" == "medqa" ]; then
                    
                    TEST_FILE="${DATA_DIR}/perturbed_val_100_medqa_1_${abst_type}.json" 
                    FEW_SHOT_POOL="${DATA_DIR}/perturbed_blended_few_shot_pool_medqa_1_train_${abst_type}.json"
                else
                    TEST_FILE="${DATA_DIR}/perturbed_val_100_amboss_alldiff_${abst_type}.json"
                    FEW_SHOT_POOL="${DATA_DIR}/perturbed_blended_few_shot_pool_amboss_train_test_alldiff_${abst_type}.json"
                fi

                sbatch --nodes=1 --partition=${PARTITION} --gpus=${GPUS} --mem=${MEM} --time=${TIME} --constraint=${CONSTRAINT} \
                       --job-name="${JOB_NAME}" \
                       --output="${LOG_DIR}/${JOB_NAME}-%j-${DATE_TAG}.out" \
                       --error="${LOG_DIR}/${JOB_NAME}-%j-${DATE_TAG}.err" \
                       ../submit_job.sh \
                         --model "$model" \
                         --dataset "$dataset" \
                         --k "$k" \
                         --perturbed \
                         --abst_type "$abst_type" \
                         --few_shot_pool "$FEW_SHOT_POOL" \
                         --test_file "$TEST_FILE" \
                         --embedding_model "$EMBEDDING_MODEL" \
                         --tracker_file "$TRACKER_FILENAME"
            done
        done
    done
done

echo "All k-tuning jobs for both original and perturbed sets have been submitted."
