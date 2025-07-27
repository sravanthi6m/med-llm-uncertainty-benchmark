#!/usr/bin/env bash

# =================================================================================
# Master script for all experimental configurations
# Requests resources and submits a separate Slurm job for each run
# =================================================================================

DATA_DIR="/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/data"
LOG_DIR="/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/outputs/slurm_master_all_fs"
mkdir -p "$LOG_DIR"

DATE_TAG=$(date +"%Y%m%d-%H%M")
TRACKER_FILENAME="run_tracker_tuning_${DATE_TAG}.csv"

##############################

MODELS=(
    # "Llama-32-1B-Instruct"
    # "Llama-32-3B-Instruct"
    # "Llama-31-8B-Instruct"
    # "Qwen25-05B-Instruct"
    # "Qwen25-15B-Instruct"
    # "Qwen25-3B-Instruct"
    # "Qwen25-7B-Instruct"
    # "Qwen25-14B-Instruct"
    # "Phi-4-mini"
    # "phi-4"
    "Qwen25-32B-Instruct"
    # 
    # "Qwen3-06B"
    # "Qwen3-1-7B"    
    # "Qwen3-4B"    
    # "Qwen3-8B"    
    #"Qwen3-14B"    
    "Qwen3-32B"    
    #"gemma-3-4b"
    #"medgemma-4b-it"    
    # "Phi-35-mini-instruct"
    # "MediPhi-Instruct"    
    #"gemma-3-27b-it"
    "medgemma-27b-text-it"
)

DATASETS=("medqa")
OG_K_VALUES=()
PERTURBED_K_VALUES=(0)
ABST_TYPES=("noabst" "randabst")

###############################
EMBEDDING_MODEL="text-embedding-ada-002"

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        
        # GPUS=4; MEM="200GB"; TIME="1-00:00:00"
        # #CONSTRAINT="vram40"
        CONSTRAINT="[a40|a16|l4|l40s|a100|h100]"
        GPUS=1; CPUS=2; MEM="30GB"; TIME="1-00:00:00"
        PARTITION="gpu"

        # if [[ "$model" == *Qwen3* || "$model" == *Llama-3* ]]; then
        #     echo "-> ($model) detected - Constraining to A40 or A100 GPUs."
        #     CONSTRAINT="[a40|a100]"
        # fi

        if [[ "$model" == *70B* || "$model" == *72B* ]]; then
            GPUS=5; MEM="250GB"; TIME="2-00:00:00"
        elif [[ "$model" == *32B* || "$model" == *27b* ]]; then
            GPUS=4; MEM="200GB"; TIME="1-12:00:00"
        fi

        for k in "${OG_K_VALUES[@]}"; do
            for abst_type in "${ABST_TYPES[@]}"; do
                JOB_NAME="${model}_${dataset}_k${k}_og_${abst_type}"
                if [ "$dataset" == "medqa" ]; then
                    FEW_SHOT_POOL="${DATA_DIR}/few_shot_pool_medqa_1_train_${abst_type}.json"
                else
                    FEW_SHOT_POOL="${DATA_DIR}/few_shot_pool_amboss_train_test_alldiff_${abst_type}.json"
                fi
                
                sbatch --nodes=1 --partition=${PARTITION} --constraint=${CONSTRAINT} --gpus=${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
                       --job-name="${JOB_NAME}" \
                       --output="${LOG_DIR}/${JOB_NAME}-%j-${DATE_TAG}.out" \
                       --error="${LOG_DIR}/${JOB_NAME}-%j-${DATE_TAG}.err" \
                       ./submit_job.sh --model "$model" --dataset "$dataset" --k "$k" --abst_type "$abst_type" --few_shot_pool "$FEW_SHOT_POOL" --embedding_model "$EMBEDDING_MODEL" --tracker_file "$TRACKER_FILENAME"
            done
        done

        for k in "${PERTURBED_K_VALUES[@]}"; do
            for abst_type in "${ABST_TYPES[@]}"; do
                JOB_NAME="${model}_${dataset}_k${k}_perturbed_${abst_type}"
                if [ "$dataset" == "medqa" ]; then
                    FEW_SHOT_POOL="${DATA_DIR}/perturbed_blended_few_shot_pool_medqa_1_train_${abst_type}.json"
                else
                    FEW_SHOT_POOL="${DATA_DIR}/perturbed_blended_few_shot_pool_amboss_train_test_alldiff_${abst_type}.json"
                fi

                sbatch --nodes=1 --partition=${PARTITION} --constraint=${CONSTRAINT} --gpus=${GPUS}  --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
                       --job-name="${JOB_NAME}" \
                       --output="${LOG_DIR}/${JOB_NAME}-%j-${DATE_TAG}.out" \
                       --error="${LOG_DIR}/${JOB_NAME}-%j-${DATE_TAG}.err" \
                       ./submit_job.sh --model "$model" --dataset "$dataset" --k "$k" --perturbed --abst_type "$abst_type" --few_shot_pool "$FEW_SHOT_POOL" --embedding_model "$EMBEDDING_MODEL" --tracker_file "$TRACKER_FILENAME"
            done
        done
    done
done

echo "All experimental jobs have been submitted."
