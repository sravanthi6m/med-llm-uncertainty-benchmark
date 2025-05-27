#!/usr/bin/env bash

MODEL_NAMES=(
    "Qwen25-05B-Instruct"
    "Qwen25-15B-Instruct"
    "Qwen25-3B-Instruct"
    "Qwen25-7B-Instruct"
    "Qwen25-14B-Instruct"
    "Llama-32-1B-Instruct"
    "Llama-32-3B-Instruct"
    "Llama-31-8B-Instruct"  
)

PROJECT_ROOT="/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/uncertainty_cp" # logits + cp ops

PROMPT_METHOD="base"
FEW_SHOT=0
CP_PROMPT_METHOD="base"
ICL_METHOD="icl0"
CAL_RATIO=0.5
ALPHA=0.1

DATA_FILES=(
  "medqa_1_noabst.json"
  "medqa_1_abstain_A.json"
  "medqa_1_abstain_B.json"
  "medqa_1_abstain_C.json"
  "medqa_1_abstain_D.json"
  "medqa_1_abstain_E.json"
  "medqa_1_abstain_F.json"
)

echo "Running pipeline for ${#MODEL_NAMES[@]} model(s) x ${#DATA_FILES[@]} data file(s)..."
echo "    Output dir: ${OUTPUT_DIR}"
echo

for MODEL_NAME in "${MODEL_NAMES[@]}"; do

    MODEL_PATH="${PROJECT_ROOT}/models/${MODEL_NAME}"

    echo -e "\n===================="
    echo "MODEL: ${MODEL_NAME}"
    echo "Path : ${MODEL_PATH}"
    echo "===================="

    for FILE in "${DATA_FILES[@]}"; do
        BASENAME="${FILE%.json}"
        RAW_FILE="${DATA_DIR}/${FILE}"

        if [[ "${FILE}" == *abstain* ]]; then
            ABSTAIN_OPTION="true"
        else
            ABSTAIN_OPTION="false"
        fi

        echo "----------"
        echo "1. GENERATE LOGITS  -  ${FILE}"
        python "${PROJECT_ROOT}/generate_logits.py"   \
                --model="${MODEL_PATH}"               \
                --data_path="${DATA_DIR}"             \
                --file="${RAW_FILE}"                  \
                --prompt_method="${PROMPT_METHOD}"    \
                --output_dir="${OUTPUT_DIR}"          \
                --few_shot="${FEW_SHOT}"              \
                --abstain_option="${ABSTAIN_OPTION}"

        echo "2. CONFORMAL PREDN -  ${BASENAME}"
        python "${PROJECT_ROOT}/uncertainty_quantification_via_cp.py"  \
                --model="${MODEL_NAME}"                                \
                --raw_data_dir="${DATA_DIR}"                           \
                --logits_data_dir="${OUTPUT_DIR}"                      \
                --data_names="${BASENAME}"                             \
                --prompt_methods="${CP_PROMPT_METHOD}"                 \
                --icl_methods="${ICL_METHOD}"                          \
                --cal_ratio="${CAL_RATIO}"                             \
                --alpha="${ALPHA}"                                     \
                --abstain_option="${ABSTAIN_OPTION}"

        echo "Completed ${MODEL_NAME}/${FILE}"
    done
done

echo -e "\nAll jobs finished"
