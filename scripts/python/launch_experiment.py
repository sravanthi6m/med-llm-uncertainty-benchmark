import argparse
import os
import sys
from dotenv import load_dotenv

# Make sure top-level module is accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("project_root is ", project_root)
sys.path.insert(0, project_root)

from run_experiment import run_experiment
from quantify_uncertainty.metrics_runner import evaluate_outputs_with_conformal


def detect_backend(model_name: str) -> str:
    """Determine backend type from model name."""
    return "openai" if "gpt" in model_name.lower() or "openai" in model_name.lower() else "open"

def build_output_path(dataset_name, cot, model_name) -> str:
    print(f"dataset_name is {dataset_name} and cot is {cot} and model_name is {model_name}")

    setting = ""
    if cot:
        setting = "fewshot"
    else:
        setting = "zeroshot"
    
    out_dir = os.path.join("outputs", dataset_name, setting, model_name.replace("/", "_"))
    metrics_dir = os.path.join("outputs", dataset_name, setting)
    print(f"newly constructred out put dir is: {out_dir} and metrics_dir: {metrics_dir}")
    return metrics_dir, out_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (.json)")
    parser.add_argument("--prompt", type=str, default="shared", help="Prompt method: shared/base/task")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--cot", type=int, default=0)
    parser.add_argument("--cal_ratio", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--model_key", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--dataset_type", type=str, default="NoAbst") #NoAbst, Abst, Pert
    args = parser.parse_args()

    model_path = args.model
    dataset_path = args.dataset
    prompt_method = args.prompt
    few_shot = args.few_shot
    cot = args.cot
    cal_ratio = args.cal_ratio
    alpha = args.alpha
    version = args.version
    model_key = args.model_key
    dataset_type = args.dataset_type

    load_dotenv(dotenv_path="env/.env")
    api_key = os.getenv(model_key)

    backend = detect_backend(model_path)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    run_name = f"{model_name}_{dataset_name}_{dataset_type}_{prompt_method}_fs{few_shot}_cot{cot}_{version}".lower()
    print(f"run_name is {run_name}")

    # out_dir = os.path.join("outputs", model_path.replace("/", "_"), model_name)
    met_dir, out_dir = build_output_path(dataset_name, cot, model_name)
    os.makedirs(met_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, f"{run_name}.jsonl")
    failures_path = os.path.join(out_dir, f"{run_name}_failures.jsonl")
    metrics_path = os.path.join(met_dir, f"metrics.json")

    print(f"file paths are: \n")
    print(f"output_path: {output_path}")
    print(f"failures_path: {failures_path}")
    print(f"metrics_path: {metrics_path}")
    
    # Run inference
    print(f"Running experiment!")
    run_experiment(
        model_name=model_path,
        backend=backend,
        dataset_path=dataset_path,
        prompt_method=prompt_method,
        few_shot=few_shot,
        cot=cot,
        output_path=output_path,
        failures_path=failures_path,
        api_key=api_key,
        dataset_type=dataset_type
    )

    # Run evaluation
    print(f"Running evaluation!")
    evaluate_outputs_with_conformal(
        jsonl_path=output_path,
        out_json=metrics_path,
        prompt_method="shared",  # keep consistent
        icl_method="icl0",
        cal_ratio=cal_ratio,
        alpha=alpha,
    )

if __name__ == "__main__":
    main()
