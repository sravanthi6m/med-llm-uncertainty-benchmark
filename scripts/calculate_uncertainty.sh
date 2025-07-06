#!/usr/bin/env bash

MODEL_NAMES=(
    "gemma-3-4b"
    "medgemma-4b-it"
    "Phi-4-mini"
    "Qwen3-06B"
    "Qwen25-05B-Instruct"
    "Llama-32-1B-Instruct"
    "Qwen3-1-7B"
    "Qwen25-3B-Instruct"
    "Llama-32-3B-Instruct"
    "Qwen3-4B"
)

# MODEL_NAMES=(
#     "Llama-31-8B-Instruct"
#     "phi-4"
#     "Qwen25-7B-Instruct"
#     "Qwen3-8B"
#     "Qwen25-14B-Instruct"
#     "Qwen3-14B"
#     "Qwen25-15B-Instruct"
# )

# MODEL_NAMES=(
#     "gemma-3-27b-it"
#     "medgemma-27b-text-it"
#     "phi-4"
#     "Qwen3-32B"
#     "Qwen25-32B-Instruct"
# )

# MODEL_NAMES=(
#     "Qwen25-72B-Instruct"
#     "Llama-31-70B-Instruct"
#     "Llama-33-70B-Instruct"
# )

PROJECT_ROOT="/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/uncertainty_cp" # logits + cp ops

PROMPT_METHOD="shared"
FEW_SHOT=0
CP_PROMPT_METHOD="shared"
ICL_METHOD="icl0"
CAL_RATIO=0.3
ALPHA=0.1

DATA_FILES=(
  "medqa_1_test_noabst.json"
  "medqa_1_test_randabst.json"
#  "amboss_alldiff_train_noabst.json"
#  "amboss_alldiff_train_randabst.json"
)

echo "Running pipeline for ${#MODEL_NAMES[@]} model(s) x ${#DATA_FILES[@]} data file(s)..."
echo "    Output dir: ${OUTPUT_DIR}"
echo

mkdir -p "${OUTPUT_DIR}"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do

    MODEL_PATH="${PROJECT_ROOT}/models/${MODEL_NAME}"
    RAW_FILE="${DATA_DIR}/${FILE}"

    echo -e "\n===================="
    echo "MODEL: ${MODEL_NAME}"
    echo "Path : ${MODEL_PATH}"
    echo "===================="

    COT_MODES_TO_RUN=("false") # Default to no CoT (false)
    if [[ "${MODEL_NAME}" == *Qwen3* ]]; then
        echo "*** ${MODEL_NAME} model detected, running with and without cot"
        COT_MODES_TO_RUN=("true" "false")
    else
        echo "*** Running with default settings (shared, no cot)"
    fi

    for FILE in "${DATA_FILES[@]}"; do
        RAW_FILE="${DATA_DIR}/${FILE}" # Define RAW_FILE here, inside the loop

        for COT_MODE in "${COT_MODES_TO_RUN[@]}"; do

            BASENAME="${FILE%.json}"
            COT_TAG="cot"
            if [[ "${COT_MODE}" == "false" ]]; then
                COT_TAG="nocot"
            fi

            JSON_WITH_ANSWERS="${OUTPUT_DIR}/${BASENAME}_${MODEL_NAME}_${PROMPT_METHOD}_${COT_TAG}_with_answers.json"
            RESULTS_JSON="${OUTPUT_DIR}/${BASENAME}_${MODEL_NAME}_${PROMPT_METHOD}_${COT_TAG}_results.json"

		    echo "----------"
            echo "RUNNING: ${FILE} with CoT: ${COT_MODE}"

		    GEN_LOGITS_CMD=(
                "python" "${PROJECT_ROOT}/med-llm-uncertainty-benchmark/generate_logits.py"
                "--model=${MODEL_PATH}"
                "--dataset_file=${RAW_FILE}"
                "--prompt_methods" "${PROMPT_METHOD}"
                "--out_dir=${OUTPUT_DIR}"
                "--output_json=${JSON_WITH_ANSWERS}"
                "--few_shot=${FEW_SHOT}"
            )
            
            # Conditionally add the --cot flag
            if [[ "${COT_MODE}" == "true" ]]; then
                GEN_LOGITS_CMD+=("--cot")
            fi

            CALC_UNCERT_DATA_NAME="${BASENAME}_${PROMPT_METHOD}_${COT_TAG}"

		    CALC_UNCERT_CMD=(
                "python" "${PROJECT_ROOT}/med-llm-uncertainty-benchmark/calculate_uncertainty.py"
                "--model=${MODEL_NAME}"
                "--raw_data_dir=${DATA_DIR}"
                "--logits_data_dir=${OUTPUT_DIR}"
                "--data_names" "${BASENAME}"
                "--prompt_methods" "${PROMPT_METHOD}"
                "--icl_methods" "icl0"
                "--cal_ratio=${CAL_RATIO}"
                "--alpha=${ALPHA}"
                "--out_json=${RESULTS_JSON}"
            )

            echo "1. GENERATE LOGITS..."
            # Execute the command
            "${GEN_LOGITS_CMD[@]}"

            echo "2. CALCULATE UNCERTAINTY..."
            # Execute the command
            "${CALC_UNCERT_CMD[@]}"

		    echo "Completed ${MODEL_NAME}/${FILE} (CoT: ${COT_MODE})"
		    echo "Results saved to ${RESULTS_JSON}"
        done
    done
done

echo -e "\nAll jobs finished"
