import requests
import json
import warnings
from datasets import load_from_disk
import csv
import argparse
import pandas as pd
from pathlib import Path
import re

warnings.filterwarnings("ignore")


api_key = "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"
org_key = "org-ewbrRzXdrHxv7hV0WyCFzGdD"


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Parse training parameters.")
    parser.add_argument(
        "--exp_folder", type=str, required=True, help="Name of the experiment folder."
    )
    parser.add_argument(
        "--exp_name", type=str, required=True, help="Name of the experiment."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input File.")
    parser.add_argument(
        "--candidate_file", type=str, required=True, help="Candidate One File."
    )
    args = parser.parse_args()
    return args


def _get_openai_response(prompt, model):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Organization": org_key,
    }

    data = {
        "messages": [{"role": "system", "content": prompt}],
        "model": model,
        "temperature": 0.0,
    }
    response = requests.post(url, headers=headers, json=data)
    print(response)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return ""


def _generate_remove_gold_context_prompt(question, options, candidate_answer):
    prompt = f"""### INSTRUCTION
You are given a question and a candidate answer to that. Your goal is to:

1. Identify and extract 3-5 key pieces of information present in the question that are essential for answering it correctly.
2. Determine the most critical piece of information among them. This should be the piece of information that makes the question challenging to answer by the candidate if removed. 
You should refer to the candidate answer to identify it.
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

### CANDIDATE ANSWER
{candidate_answer}

### RESPONSE\n
"""
    return prompt


def _run_inference(question, options, candidate_answer, model="gpt-4o-mini"):
    llm_prompt = _generate_remove_gold_context_prompt(
        question, options, candidate_answer
    )
    response = _get_openai_response(prompt=llm_prompt, model=model)
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


def _inference(exp_folder, exp_name, input_file, candidate_file):
    directory_path = Path(exp_folder)
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_csv(input_file)
    # dataframe = dataframe[:20]
    candidate_df = pd.read_csv(candidate_file)
    results = []
    for i, row in dataframe.iterrows():
        question = row["question"] + "\n"
        candidate_answer = candidate_df.iloc[i]["answer"]
        resp, prompt = _run_inference(
            question=question,
            options=row["options_text"],
            candidate_answer=candidate_answer,
        )
        key_info, gold_context, reason, modified_question = _parse_llm_resp(resp)
        results.append(
            {
                "question": question,
                "prompt": prompt,
                "response": resp,
                "options": row["options"],
                "options_text": row["options_text"],
                "key_information": key_info,
                "gold_context": gold_context,
                "reason": reason,
                "modified_question": modified_question,
                "correct_option": row["correct_option"],
                "correct_reason": row["correct_reason"],
                "gpt_reason": row["gpt_reason"],
            }
        )
    output_dataframe = pd.DataFrame(results)
    output_file = f"{exp_folder}/{exp_name}__test.csv"
    print(
        f"Saving DataFrame with shape: {str(output_dataframe.shape)} to {output_file}",
    )
    output_dataframe.index.name = "index"
    output_dataframe.to_csv(output_file, index=True)


if __name__ == "__main__":
    args = _parse_arguments()
    _inference(
        exp_folder=args.exp_folder,
        exp_name=args.exp_name,
        input_file=args.input_file,
        candidate_file=args.candidate_file,
    )
