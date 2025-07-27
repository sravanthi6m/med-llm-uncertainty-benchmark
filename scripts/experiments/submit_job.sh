#!/usr/bin/env bash
#SBATCH -p gpu
# Default resources: (can be overridden by master launch script)
#SBATCH -G 1
#SBATCH -c 2  # Number of CPU cores
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00

##############

LOG_DIR="/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/outputs/slurm_master_tuning"
mkdir -p "$LOG_DIR"
DATE_TAG=$(date +"%Y%m%d-%H%M")
#SBATCH -o ${LOG_DIR}/%x-%j-${DATE_TAG}.out
#SBATCH -e ${LOG_DIR}/%x-%j-${DATE_TAG}.err

export SLURM_LOG_FILE="${LOG_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${DATE_TAG}.out"

##############

source /home/smachcha_umass_edu/cp_quant/bin/activate

PROJECT_ROOT="/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/"
LAUNCHER_SCRIPT="${PROJECT_ROOT}/med-llm-uncertainty-benchmark/scripts/experiments/run_experiment.py"

ENV_FILE="${PROJECT_ROOT}/med-llm-uncertainty-benchmark/env/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    export $(cat "$ENV_FILE" | xargs)
else
    echo "Warning: Environment file not found at $ENV_FILE"
fi

echo "========================================================"
echo "Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Log file: $SLURM_LOG_FILE"
echo "Running with arguments: $@"
echo "========================================================"

echo "--- NVIDIA-SMI Info ---"
nvidia-smi
echo "-----------------------"

python -u "$LAUNCHER_SCRIPT" "$@" # "$@" forwards all arguments from sbatch to py script

if [ $? -eq 0 ]; then
    echo "Python script finished successfully."
else
    echo "ERROR: Python script failed. Check the output log for details."
fi
