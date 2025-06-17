import json
from collections import Counter


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

    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {output_metrics_path}")
