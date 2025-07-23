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
from quantify_uncertainty.models.openai_reasoning import ReasoningModel
from quantify_uncertainty.data_helpers.loaders import load_all_data
from scripts.python.dynamic_sampler import (
    DynamicSampler,
    SentenceTransformerModel,
    OpenAIEmbeddingModel,
)


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
    failures_path: str = "experiment_failures_record.jsonl",
    api_key: str = None,
    dataset_type: str = None,
    few_shot_pool_1: str = None,
    embedding_model: str = None,
    few_shot_pool_2: str = None,
):
    dynamic_fewshot_map = None
    if few_shot > 0:
        print("Generating dynamic samples")

        if not few_shot_pool_1 or not os.path.exists(few_shot_pool_1):
            raise FileNotFoundError(
                f"Dynamic few-shot requires a valid path to json file at --few_shot_pool_1. Path not found: {args.few_shot_pool_1}"
            )

        if "text-embedding" in embedding_model:
            embedding_model_instance = OpenAIEmbeddingModel(model_name=embedding_model)
        else:
            embedding_model_instance = SentenceTransformerModel(
                model_name=embedding_model
            )

        sampler = DynamicSampler(
            data_file_path=dataset_path,
            few_shot_pool_path_1=few_shot_pool_1,
            few_shot_pool_path_2=few_shot_pool_2,
            embedding_model=embedding_model_instance,
        )
        dynamic_fewshot_map = sampler.get_dynamic_few_shot_examples(
            k=few_shot, pool_1_percentage=1.0
        )

    # Load raw data
    raw_data = load_all_data(
        os.path.dirname(dataset_path),
        os.path.splitext(os.path.basename(dataset_path))[0],
    )
    src = raw_data[0]["source"]

    # # Few-shot examples
    # fewshot = get_fewshot_examples(raw_data, src, few_shot) if few_shot > 0 else None

    # Prompt formatter
    format_fn = PROMPT_DISPATCH[prompt_method]

    # Load model
    if backend == "open":
        model = OpenSourceHFModel(model_name)
    elif backend == "openai":
        model = OpenAIModel(model_name, api_key=api_key)
    elif backend == "reasoning":
        model = ReasoningModel(model_name, api_key=api_key)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a") as fout, open(failures_path, "a") as ferr:
        for ex in tqdm(raw_data, desc=f"{model_name} | {prompt_method}"):
            current_fewshot_examples = (
                dynamic_fewshot_map.get(ex["id"], []) if dynamic_fewshot_map else None
            )
            formatted = format_fn(
                ex,
                argparse.Namespace(few_shot=few_shot, cot=cot),
                current_fewshot_examples,
            )
            prompt = formatted["prompt"]
            # print(f"prompt here is {prompt}")
            choices = list(ex["choices"].keys())

            meta = {
                "model": model_name,
                "few_shot": few_shot,
                "cot": cot,
                "dataset_type": dataset_type,
            }
            result = {
                "id": ex["id"],
                "source": ex["source"],
                "prompt": prompt,
                "choices": choices,
                "answer": ex["answer"],
                "meta": meta,
            }
            try:
                # Run model
                model_response = model.generate(prompt, choices)
                # model_response = {
                #     "output": "A",
                #     "raw_logprobs": {
                #         "A": -0.033782511949539185,
                #         "B": -3.533782482147217,
                #         "C": -9.533782958984375,
                #         "D": -5.533782482147217,
                #         "E": -15.283782958984375,
                #     },
                # }
                result["output"] = model_response["output"]
                result["logprobs"] = model_response["raw_logprobs"]

                fout.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"⚠️ Error on ID {ex['id']}: {e}")
                fail_record = {**result, "error": str(e)}
                ferr.write(json.dumps(fail_record) + "\n")

    print(f"Saved outputs to: {output_path}")
