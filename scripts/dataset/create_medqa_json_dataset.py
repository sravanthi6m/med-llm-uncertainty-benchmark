import re
import json
import string
import random
import pyarrow as pa
import pandas as pd

from collections import Counter
from pyarrow import RecordBatchStreamReader


def parse_question_and_options(full_text):
    """
    Splits text: question + options 
    """
    #match = re.search(r'([.?!:])\s+(?=A:\s)', full_text)
    match = re.search(r'(\W)\s+(?=A:\s)', full_text)
    if not match:
        raise ValueError("Could not find start of options (A:) in question.")
    
    split_idx = match.end()
    question_part = full_text[:split_idx].strip()
    options_part = full_text[split_idx:].strip()
    
    # split options_part by sth like A: xxx B: yyy C: zzz ...
    option_matches = re.findall(r'([A-F]):\s(.*?)(?=(?:\s[A-F]:\s)|$)', options_part)
    options = [text.strip() for label, text in option_matches]
    
    return question_part, options


def convert_medqa_qa_column_to_json_abstain(df, output_json, allow_abstain=False, random_seed=None):
    """
    Converts a df columnfrom MEDQA into a JSON file in required format for uncertainty metric calculation
    Also has optional param to include abstention choice
    
    Args:
        df (pd.DataFrame): The input DataFrame containing 'input' and 'output' columns.
        output_json (str): The path for the output JSON file.
        allow_abstain (bool): Whether to add abstention option (at random position) for each question - default False
        random_seed (int, optional)
    """
    output_data = []
    abstain_position_counts = Counter()
    
    # set seed...
    if random_seed is not None:
        random.seed(random_seed)
    
    for idx, row in df.iterrows():
        try:
            question_text, options_list = parse_question_and_options(row["input"])
        except ValueError as e:
            print(f"Skipping row {idx}: {e}")
            continue
        
        # original correct answer letter - single letter provided ('A', 'B', etc)
        correct_answer_letter = row["output"].strip().upper()
        
        if allow_abstain:
            abstain_text = "I am unable to answer this question for certain reasons."
            
            original_answer_idx = ord(correct_answer_letter) - ord('A')
            
            # generate random position to insert abstain option
            num_original_options = len(options_list)
            insertion_idx = random.randint(0, num_original_options)
            
            abstain_position_counts[(chr(ord('A') + insertion_idx))] += 1
            
            options_list.insert(insertion_idx, abstain_text)
            
            # shifting correct answer letter accordingly
            if insertion_idx <= original_answer_idx:
                new_answer_idx = original_answer_idx + 1
                correct_answer_letter = chr(ord('A') + new_answer_idx)
        
        options_dict = {letter: opt for letter, opt in zip(string.ascii_uppercase, options_list)}
        
        entry = {
            "id": idx,
            "source": "MEDQA_1", # MEDQA_1 for og questions, MEDQA_3 for perturbed questions
            "question": question_text,
            "choices": options_dict,
            "answer": correct_answer_letter
        }
        
        output_data.append(entry)
    
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Converted {len(output_data)} items and saved to {output_json}")
    
    if allow_abstain:
        print("\n### Abstain Option Distribution ###")
        if not abstain_position_counts:
            print("Nothing processed - no distribution")
        else:
            # Sort the results by letter for a clean, predictable report
            for letter, count in sorted(abstain_position_counts.items()):
                print(f"Position {letter}: {count} times")
        print("-----------------------------------")


def main():
    arrow_file_path = './data-00000-of-00001.arrow'
    
    data_loader = RecordBatchStreamReader(arrow_file_path)
    df = data_loader.read_pandas()

    output_json = "medqa_1_test_randabst.json"
    
    #convert_medqa_qa_column_to_json_abstain(df, output_json, allow_abstain=False, random_seed=None)
    convert_medqa_qa_column_to_json_abstain(df, output_json, allow_abstain=True, random_seed=42)


if __name__ == "__main__":
    main()

