import json
from collections import Counter

import numpy as np

import json, os
from quantify_uncertainty.utils import load_and_split_jsonl

from quantify_uncertainty.metrics.cp_metrics import compute


def evaluate_outputs_with_conformal(
    jsonl_path,
    out_json,
    prompt_method="base",
    icl_method="icl0",
    cal_ratio=0.5,
    alpha=0.1,
):
    cal_raw, test_raw = load_and_split_jsonl(jsonl_path, cal_ratio)

    logits_data_all = {
        f"{prompt_method}_{icl_method}": {
            "cal": build_logits_rows(cal_raw),
            "test": build_logits_rows(test_raw),
        }
    }

    metrics = compute(
        logits_data_all,
        cal_raw,
        test_raw,
        prompt_methods=[prompt_method],
        icl_methods=[icl_method],
        alpha=alpha,
    )

    out = {
        "meta": cal_raw[0].get("meta", {}),
        "num_examples": len(cal_raw) + len(test_raw),
        "metrics": metrics,
    }
    save_metrics_append(out, out_json)
    # os.makedirs(os.path.dirname(out_json), exist_ok=True)
    # with open(out_json, "w") as f:
    #     json.dump(out, f, indent=2)

    # print("Metrics saved →", out_json)


def build_logits_rows(examples):
    """Convert each JSONL row → dict expected by LAC/APS code."""
    rows = []
    for row in examples:
        rows.append(
            {
                "id": row["id"],
                "logprobs_options": np.array(
                    [row["logprobs"][k] for k in row["choices"]]
                ),
                "choices": row["choices"],
            }
        )
    return rows


def evaluate_outputs(output_jsonl_path: str, output_metrics_path: str):
    data = []
    with open(output_jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    acc_numerator = 0
    abstain_count = 0
    for row in data:
        pred = row["output"].strip()
        truth = row["ground_truth"].strip()

        if pred == truth:
            acc_numerator += 1

        # Optional: abstention detection
        if pred.lower() in ["e", "f", "i don't know", "cannot answer"]:
            abstain_count += 1

    metrics = {
        "model": row.get("meta", {}).get("model", "unknown"),
        "prompt_method": row.get("meta", {}).get("prompt_method", "base"),
        "few_shot": row.get("meta", {}).get("few_shot", False),
        "cot": row.get("meta", {}).get("cot", False),
        "num_examples": len(data),
        "metrics": {
            "accuracy": acc_numerator / len(data),
            "abstain_rate": abstain_count / len(data),
        },
    }

    with open(output_metrics_path, "a") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {output_metrics_path}")

def save_metrics_append(out: dict, out_json: str):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    # Load existing if it exists
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            raise ValueError("Expected a list of metrics in the output JSON file.")
    else:
        existing = []

    existing.append(out)

    # Save updated list
    with open(out_json, "w") as f:
        json.dump(existing, f, indent=2)

    print("Metrics appended →", out_json)