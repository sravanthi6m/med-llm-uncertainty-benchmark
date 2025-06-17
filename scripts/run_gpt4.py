import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from run_experiment import run_experiment
from quantify_uncertainty.metrics_runner import evaluate_outputs

model_name = "gpt-4o-mini"
backend = "openai"
api_key = ""

dataset_path = "/Users/sushritayerra/Downloads/sample_test.json"
prompt_method = "shared"
few_shot = 0
cot = False

out_dir = "outputs"
run_name = f"{model_name}_medqa_{prompt_method}_fs{few_shot}_cot{cot}_test".lower()
output_path = f"{out_dir}/{run_name}.jsonl"
metrics_path = f"{out_dir}/{run_name}_metrics.json"

# 1. Run inference
run_experiment(
    model_name=model_name,
    backend=backend,
    dataset_path=dataset_path,
    prompt_method=prompt_method,
    few_shot=few_shot,
    cot=cot,
    output_path=output_path,
    api_key=api_key,
)

# 2. Run evaluation and save metrics
evaluate_outputs(output_jsonl_path=output_path, output_metrics_path=metrics_path)
