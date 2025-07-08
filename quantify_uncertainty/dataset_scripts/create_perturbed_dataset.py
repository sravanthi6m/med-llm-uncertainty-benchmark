from typing import Optional
import warnings
import argparse
import re
from openai import OpenAI, OpenAIError, RateLimitError, APIStatusError, APITimeoutError
from typing import List, Optional
import os
import time
import traceback
import json
import numpy as np
from dotenv import load_dotenv

warnings.filterwarnings("ignore")


class OpenAIModel:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                output = response.choices[0].message.content.strip()
                # print(f"returning: {output}")
                return output

            except (RateLimitError, OpenAIError, APITimeoutError) as e:
                wait_time = backoff_base**attempt
                print(
                    f"[Retry {attempt + 1}/{max_retries}] ⚠️ Retriable error occurred:"
                )
                print(f"  └─ Type: {type(e).__name__}")
                print(f"  └─ Message: {str(e)}")
                traceback.print_exc()
                time.sleep(wait_time)
            except Exception as e:
                print("🚨 Non-retryable error:")
                print(f"  └─ Type: {type(e).__name__}")
                print(f"  └─ Message: {str(e)}")
                traceback.print_exc()
                break

        raise RuntimeError(
            f"Failed to get response from OpenAI after {max_retries} attempts"
        )


def _generate_remove_gold_context_prompt(question, options, correct_answer):
    prompt = f"""### INSTRUCTION
You are given a question and the answer to that. Your goal is to:

1. Identify and extract 3-5 key pieces of information present in the question that are essential for answering it correctly.
2. Determine the most critical piece of information among them. This should be the piece of information that makes the question challenging to answer by the candidate if removed. 
You should refer to the answer to identify it. This piece of information is referred to as the `gold context` henceforth.
3. Remove the gold context from the question to create a `modified question`. You can only remove the gold context, but not add incorrect context. 
4. This modified question should not deviate too much from question but also make it challenging for candidates to answer correctly.
4. Format your response in the following structure:
[Key Information]  
<key_information>
[Gold Context]  
<gold_context>
[Explanation for Selecting Gold Context]
<explanation_for_selecting_gold_context>
[Modified Question]  
<modified_question>

### QUESTION
{question}

### OPTIONS
{options}

### ANSWER
{correct_answer}

### RESPONSE\n
"""
    return prompt


def _run_inference(openaimodel: OpenAIModel, question, options, correct_answer):
    llm_prompt = _generate_remove_gold_context_prompt(question, options, correct_answer)
    response = openaimodel.generate(prompt=llm_prompt)
    return response, llm_prompt


def _parse_llm_resp(resp):
    key_info_pattern = r"\[Key Information\](.*?)\n\[Gold Context\]"
    gold_context_pattern = (
        r"\[Gold Context\](.*?)\n\[Explanation for Selecting Gold Context\]"
    )
    reason_pattern = (
        r"\[Explanation for Selecting Gold Context\](.*?)\n\[Modified Question\]"
    )
    modified_question_pattern = r"\[Modified Question\](.*)"
    key_info = re.search(key_info_pattern, resp, re.DOTALL).group(1).strip()
    gold_context = re.search(gold_context_pattern, resp, re.DOTALL).group(1).strip()
    reason = re.search(reason_pattern, resp, re.DOTALL).group(1).strip()
    modified_question = (
        re.search(modified_question_pattern, resp, re.DOTALL).group(1).strip()
    )
    return key_info, gold_context, reason, modified_question


def process_dataset(openai_model, original_dataset_path, perturbed_dataset_path):
    print(f"processing dataset")
    with open(original_dataset_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    processed_dataset = []
    for item in data:
        processed_item = process_item(openai_model, item)
        processed_dataset.append(processed_item)

    with open(perturbed_dataset_path, "w") as f:
        json.dump(processed_dataset, f, indent=2)
    print(f"dataset processing complete")


def get_data(item):
    return (
        item["question"],
        item["choices"],
        item["answer"],
    )  # TODO: update the logic based on different source datasets


def process_item(openaimodel, item):
    print(f"processing item: {item["id"]}")
    question, options, correct_answer = get_data(item)
    resp, prompt = _run_inference(
        openaimodel=openaimodel,
        question=question,
        options=options,
        correct_answer=correct_answer,
    )
    key_info, gold_context, reason, modified_question = _parse_llm_resp(resp)
    result = {
        "id": item["id"],
        "source": item["source"],
        "question": modified_question,
        "choices": options,
        "answer": correct_answer,
        "original_question": question,
        "meta": {
            "key_information": key_info,
            "gold_context": gold_context,
            "reason": reason,
        },
    }
    return result


def main():
    print(f"starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset (.json)"
    )
    parser.add_argument(
        "--perturbed_dataset", type=str, required=True, help="Path to dataset (.json)"
    )
    parser.add_argument("--model_key", type=str, default="OPENAI_API_KEY")

    args = parser.parse_args()

    model = args.model
    dataset_path = args.dataset
    perturbed_dataset_path = args.perturbed_dataset
    model_key = args.model_key

    load_dotenv(dotenv_path="env/.env")
    api_key = os.getenv(model_key)

    openai_model = OpenAIModel(model_name=model, api_key=api_key)
    print("processing")
    process_dataset(
        openai_model=openai_model,
        original_dataset_path=dataset_path,
        perturbed_dataset_path=perturbed_dataset_path,
    )


if __name__ == "__main__":
    main()
