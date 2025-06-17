import argparse
import json
import os
from typing import List
from tqdm import tqdm

from quantify_uncertainty.prompts.prompt_helpers import (
    PROMPT_DISPATCH,
    few_shot_exp_ids,
)
from quantify_uncertainty.models.open_source import OpenSourceHFModel
from quantify_uncertainty.models.openai_model import (
    OpenAIModel,
)
from quantify_uncertainty.data_helpers.loaders import load_all_data


def get_fewshot_examples(raw_data, src: str, n: int) -> List[dict]:
    demo_ids = few_shot_exp_ids.get(src, [])[:n]
    return [ex for ex in raw_data if ex["id"] in demo_ids]


def run_experiment(
    model_name: str,
    backend: str,
    dataset_path: str,
    prompt_method: str = "base",
    few_shot: int = 0,
    cot: bool = False,
    output_path: str = "experiment_outputs.jsonl",
    api_key: str = None,
):
    # Load raw data
    raw_data = load_all_data(
        os.path.dirname(dataset_path),
        os.path.splitext(os.path.basename(dataset_path))[0],
    )
    src = raw_data[0]["source"]

    # Few-shot examples
    fewshot = get_fewshot_examples(raw_data, src, few_shot) if few_shot > 0 else None

    # Prompt formatter
    format_fn = PROMPT_DISPATCH[prompt_method]

    # Load model
    if backend == "open":
        model = OpenSourceHFModel(model_name)
    elif backend == "openai":
        model = OpenAIModel(model_name, api_key=api_key)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for ex in tqdm(raw_data, desc=f"{model_name} | {prompt_method}"):
            formatted = format_fn(
                ex, argparse.Namespace(few_shot=few_shot, cot=cot), fewshot
            )
            prompt = formatted["prompt"]
            print(f"prompt here is {prompt}")
            choices = list(ex["choices"].keys())

            result = {
                "id": ex["id"],
                "source": ex["source"],
                "prompt": prompt,
                "choices": choices,
                "ground_truth": ex["answer"],
            }

            # Run model
            result["output"] = model.generate(prompt)

            # Try to get logits if backend supports it
            logits = model.get_logits(prompt, choices)
            if logits is not None:
                result["logits_options"] = logits.tolist()

            fout.write(json.dumps(result) + "\n")

    print(f"Saved outputs to: {output_path}")
