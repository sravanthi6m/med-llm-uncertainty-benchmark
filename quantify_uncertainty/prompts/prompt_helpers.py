import json
from typing import List, Dict
from . import prompt_templates as pt

few_shot_exp_ids: Dict[str, List[int]] = {
    "MMLU": [1, 3, 5, 7, 9],
    "HellaSwag": [1, 3, 5, 7, 9],
    "CosmosQA": [1, 3, 5, 7, 9],
    "Halu-OpenDialKG": [5, 7, 9],
    "Halu-CNN/DailyMail": [9],
}


def _format_example(example, prompt, with_answer=False):
    """
    Appends one Q-A example (dict) to `prompt` (str)
    Retruns str
    """
    src = example["source"]
    if (src in ("MMLU", "MEDQA_1", "MEDQA_3")) or ("AMBOSS" in src):
        prompt += "Question: " + example["question"] + "\nChoices:\n"
    elif src == "CosmosQA":
        prompt += f"Context: {example['context']}\nQuestion: {example['question']}\nChoices:\n"
    elif src == "HellaSwag":
        prompt += f"Context: {example['context']}\nQuestion: {example['question']}\nChoices:\n"
    elif src == "Halu-OpenDialKG":
        prompt += f"Dialogue: {example['context']}\nQuestion: {example['question']}\nChoices:\n"
    elif src == "Halu-CNN/DailyMail":
        prompt += f"Document: {example['context']}\nQuestion: {example['question']}\nChoices:\n"
    else:
        raise NotImplementedError(f"Dataset {src} not supported.")
    for k, v in example["choices"].items():
        prompt += f"{k}. {v}\n"
    prompt += "Answer:"
    if with_answer:
        prompt += " " + example["answer"] + "\n"
    return prompt


# builders

def _base(example, args, fewshot=None):
    prompt = ""
    if (args.k_few_shot > 0) and fewshot:
        for ex in fewshot:
            prompt = _format_example(ex, prompt, with_answer=True)
    if args.cot:
        prompt = pt.base_cot_prompt
    prompt = _format_example(example, prompt)
    return {"id": example["id"], "choices": example["choices"], "prompt": prompt}


def _shared(example, args, fewshot=None):
    if args.cot:
        prompt = pt.shared_cot_prompt
    else:
        prompt = pt.shared_zero_prompt
    
    if (args.k_few_shot > 0) and fewshot:
        prompt = pt.shared_few_prompt
        for ex in fewshot:
            prompt = _format_example(ex, prompt, with_answer=True)
        if args.cot:
            prompt += pt.shared_cot_prompt
        else:
            prompt += pt.shared_zero_prompt
    
    prompt = prompt.format(num_choices=len(example["choices"]))
    prompt = _format_example(example, prompt)
    return {"id": example["id"], "choices": example["choices"], "prompt": prompt}


def _task(example, args, fewshot=None):
    pt_dict = json.loads(pt.task_zero_prompt, strict=False)
    prompt = pt_dict[example["source"]]
    if (args.k_few_shot > 0) and fewshot:
        pt_dict = json.loads(pt.task_few_prompt, strict=False)
        prompt = pt_dict[example["source"]]
        for ex in fewshot:
            prompt = _format_example(ex, prompt, with_answer=True)
        prompt += (
            "\nNow make your best effort and select the correct answer "
            "for the following question. You only need to output the option.\n\n"
        )

    if args.cot:
        pt_dict = json.loads(pt.task_cot_prompt, strict=False)
        prompt = pt_dict[example["source"]]
    prompt = _format_example(example, prompt)
    return {"id": example["id"], "choices": example["choices"], "prompt": prompt}


PROMPT_DISPATCH = {"base": _base, "shared": _shared, "task": _task}

