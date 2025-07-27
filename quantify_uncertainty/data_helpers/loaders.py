import json
import os
import pickle
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

ids_to_remove: List[int] = []

def get_raw_data(raw_data_file: str,
                 cal_ratio: float
                ) -> Tuple[List[dict], List[dict]]:
    with open(raw_data_file, "r") as f:
        raw_data = json.load(f)
    raw_data = [d for idx, d in enumerate(raw_data) if idx not in ids_to_remove]
    
    return train_test_split(raw_data, train_size=cal_ratio, random_state=42)


def get_logits_data(logits_pkl_path: str,
                    cal_ratio: float,
                    prompt_methods: List[str],
                    icl_methods: List[str]
                    ) -> Dict[str, Dict[str, list]]:
    logits_all = {}
    for m in prompt_methods:
        for fs in icl_methods:          
            print(f"Loading logits from: {logits_pkl_path}")
            with open(logits_pkl_path, "rb") as fp:
                logits = pickle.load(fp)
            logits = [item for i, item in enumerate(logits) if i not in ids_to_remove]
            cal_logits, test_logits = train_test_split(
                logits, train_size=cal_ratio, random_state=42
            )
            logits_all[f"{m}_{fs}"] = {"cal": cal_logits, "test": test_logits}
    return logits_all


def convert_id_to_ans(raw_data):
    return {str(row["id"]): row["answer"] for row in raw_data}
