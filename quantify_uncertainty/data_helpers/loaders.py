import json
import os
import pickle
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

ids_to_remove: List[int] = []

def get_raw_data(raw_data_dir: str, data_name: str, cal_ratio: float
                 ) -> Tuple[List[dict], List[dict]]:
    path = os.path.join(raw_data_dir, f"{data_name}.json")
    raw_data = json.load(open(path, "r"))
    raw_data = [d for idx, d in enumerate(raw_data) if idx not in ids_to_remove]
    
    return train_test_split(raw_data, train_size=cal_ratio, random_state=42)


def load_all_data(raw_data_dir: str, data_name: str):
    path = os.path.join(raw_data_dir, f"{data_name}.json")
    return json.load(open(path, "r"))


def get_logits_data(model_name: str,
                    data_name: str,
                    cal_raw_data,
                    test_raw_data,
                    logits_data_dir: str,
                    cal_ratio: float,
                    prompt_methods: List[str],
                    icl_methods: List[str],
                    k_few_shot: int,
                    dynamic_few_shot: bool,
                    cot: bool) -> Dict[str, Dict[str, list]]:
    logits_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            cot_tag = "cot" if cot else "nocot"
            few_shot_tag = f"k{k_few_shot}" if k_few_shot > 0 else "k0"
            dynamic_tag = "dynamic" if dynamic_few_shot else "static"
            
            fname = f"{model_name}_{data_name}_{m}_{few_shot_tag}_{dynamic_tag}_{cot_tag}.pkl"
            path = os.path.join(logits_data_dir, fname)
            with open(path, "rb") as fp:
                logits = pickle.load(fp)
            logits = [item for i, item in enumerate(logits) if i not in ids_to_remove]
            cal_logits, test_logits = train_test_split(
                logits, train_size=cal_ratio, random_state=42
            )
            logits_all[f"{m}_{fs}"] = {"cal": cal_logits, "test": test_logits}
    return logits_all


def convert_id_to_ans(raw_data):
    return {str(row["id"]): row["answer"] for row in raw_data}

