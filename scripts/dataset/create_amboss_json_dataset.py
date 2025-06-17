# Expected directory structure:
#    ./Amboss/
#    |-- d1/
#    │   |-- test/
#    │   │   |-- data-00000-of-00001.arrow
#    │   |-- train/
#    │       |-- data-00000-of-00001.arrow
#    |-- d2/
#    │   |-- ...
#    |-- ...


import pyarrow as pa
import pandas as pd
import numpy as np
import json
import string
import random
import os

from collections import Counter
from pyarrow import RecordBatchStreamReader

def load_arrow_to_df(path):
    """Loads a pd.df from specified arrow filepath"""
    
    with RecordBatchStreamReader(path) as reader:
        df = reader.read_pandas()
    return df

def process_amboss_data(base_path, output_json_path, allow_abstain=False, random_seed=None):
    """
    Loads Amboss data from difficulty levels 1-5, combines them to total 1000 questions, and formats into required JSON
    
    Args:
        base_path (str): root folder containing the 'd1', 'd2', etc. dirs
        output_json_path (str):  path to save final JSON file
    """
    all_dfs = []
    
    for i in range(1, 6):
        difficulty = f'd{i}'
        print(f"Processing difficulty level: {difficulty}")
        
        """
        test_path = os.path.join(base_path, difficulty, 'test', 'data-00000-of-00001.arrow')
        train_path = os.path.join(base_path, difficulty, 'train', 'data-00000-of-00001.arrow')
        
        df_test = load_arrow_to_df(test_path)
            
        if len(df_test) >= 200:
            df_difficulty = df_test.head(200).copy()
            df_difficulty['source'] = f'AMBOSS_{difficulty}_test'
        else:
            # when test set too small, sample remaining from? ... (train set?)
            num_needed = 200 - len(df_test)
            print(f"Test set has {len(df_test)} rows: Need {num_needed} more")
            
            df_train = load_arrow_to_df(train_path)
            
            if df_train is None or len(df_train) < num_needed:
                print(f"Could not load enough training data for {difficulty} - skip...")
                continue
            
            df_test['source'] = f'AMBOSS_{difficulty}_test'
            
            df_train_sampled = df_train.sample(n=num_needed, random_state=42) 
            df_train_sampled['source'] = f'AMBOSS_{difficulty}_train'
            
            df_difficulty = pd.concat([df_test, df_train_sampled], ignore_index=True)
        
        all_dfs.append(df_difficulty)
    
    if not all_dfs:
        print("No data was processed. Exiting.")
        return
    """
        ###
        train_path = os.path.join(base_path, difficulty, 'train', 'data-00000-of-00001.arrow')

        df_train = load_arrow_to_df(train_path)

        if df_train is None:
            print(f"Could not load train data for {difficulty} - skip...")
            continue
        if len(df_train) < 200:
            print(f"Train data for {difficulty} has only {len(df_train)} rows - ( < 200, skipp)")
            continue
            
        # rand 200 questions ...
        df_difficulty = df_train.sample(n=200, random_state=42).copy()

        df_difficulty['source'] = f'AMBOSS_{difficulty}_train'
        
        all_dfs.append(df_difficulty)

    if not all_dfs:
        print("No data was processed. Exiting.")
        return
    ###
    
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n Total questions collected: {len(final_df)}")
    
    output_data = []
    abstain_position_counts = Counter()
    
    for idx, row in final_df.iterrows():
        try:
            if not isinstance(row.get('options'), np.ndarray): continue
            
            original_options_text_list = [
                opt_dict.get('option', '').split('\n', 1)[1].strip()
                for opt_dict in row['options']
            ]

            original_answer_letter = row['correct_answer'].split('\n', 1)[0].strip()
            original_answer_index = ord(original_answer_letter) - ord('A')
            
            final_options_list = original_options_text_list
            final_answer_letter = original_answer_letter
            
            if allow_abstain:
                abstain_text = "I am unable to answer this question for certain reasons."
                num_original_options = len(original_options_text_list)
                
                insertion_index = random.randint(0, num_original_options)
                
                abstain_letter = chr(ord('A') + insertion_index)
                abstain_position_counts[abstain_letter] += 1
                
                final_options_list = list(original_options_text_list)
                final_options_list.insert(insertion_index, abstain_text)
                
                # update correct answer letter accordingly...
                if insertion_index <= original_answer_index:
                    new_answer_index = original_answer_index + 1
                    final_answer_letter = chr(ord('A') + new_answer_index)

            final_choices_dict = {
                letter: opt for letter, opt in zip(string.ascii_uppercase, final_options_list)
            }

            entry = {
                'id': idx,
                'source': row['source'],
                'question': row['question'],
                'choices': final_choices_dict,
                'answer': final_answer_letter
            }
            output_data.append(entry)

        except (IndexError, AttributeError, ValueError):
            print(f"Warning: skipp row {idx} ({row['source']}) - malformed data")
            continue
        
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Successfully converted {len(output_data)} items and saved to {output_json_path}")
    if allow_abstain:
        print("\n### Abstain Option Distribution ###")
        if not abstain_position_counts:
            print("Nothing processed - no distribution")
        else:
            for letter, count in sorted(abstain_position_counts.items()):
                print(f"Position {letter}: {count} times")
        print("-----------------------------------")


def main():
    base_path = '~/Downloads/Amboss'
    output_json = 'amboss_alldiff_train_noabst.json'
    output_json_abst = 'amboss_alldiff_train_randabst.json'
    
    process_amboss_data(base_path, output_json_abst, allow_abstain=True, random_seed=42)

if __name__ == "__main__":
    main()



