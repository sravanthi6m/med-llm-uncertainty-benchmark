import re
import numpy as np
from collections import Counter

from .cp_sets import LAC_CP, APS_CP
from ..data_helpers.loaders import convert_id_to_ans


METRIC_NAME = "conformal_pred_summary"
ABSTENTION_STRING = "I am unable to answer this question for certain reasons."


def _get_abstention_option(choices):
    for choice_key, choice_value in choices.items():
        if ABSTENTION_STRING in choice_value:
            return choice_key
    return None


def _get_accuracy(logits_data_all, test_raw, pm, icl):
    acc, e_ratio, f_ratio, abstention_rate = {}, {}, {}, {}
    for m in pm:
        for fs in icl:
            key = f"{m}_{fs}"
            res, preds, abst = [], [], []
            for logit_row, raw_row in zip(logits_data_all[key]["test"], test_raw):
                print(f"logit_row is {logit_row}")
                opts = logit_row.get("option_keys_for_logits") or list(
                    raw_row["choices"].keys()
                )
                abstention_option = _get_abstention_option(raw_row["choices"])

                truth = raw_row["answer"]
                pred = opts[int(np.argmax(logit_row["logits_options"]))]
                preds.append(pred)
                res.append(int(pred == truth))

                if abstention_option and abstention_option == pred:
                    abst.append(logit_row["id"])
            acc[key] = np.mean(res)
            cts = Counter(preds)
            e_ratio[key] = cts.get("E", 0) / len(preds) if preds else 0
            f_ratio[key] = cts.get("F", 0) / len(preds) if preds else 0
            abstention_rate[key] = len(abst) / len(preds) if abst else 0
    return acc, e_ratio, f_ratio, abstention_rate


def _coverage(pred_sets_all, id2ans, pm, icl):
    cov = {}
    for m in pm:
        for fs in icl:
            key = f"{m}_{fs}"
            cov[key] = np.mean([id2ans[k] in v for k, v in pred_sets_all[key].items()])
    return cov


def _set_size(pred_sets_all, pm, icl):
    sz = {}
    for m in pm:
        for fs in icl:
            key = f"{m}_{fs}"
            sz[key] = np.mean([len(v) for v in pred_sets_all[key].values()])
    return sz


def _set_size_by_difficulty(pred_sets_all, test_raw, pm, icl):
    """
    Only for AMBOSS - average set size for each difficulty level
    """
    id_to_source = {str(item['id']): item.get('source', '') for item in test_raw}
    
    m = pm[0]
    fs = icl[0]
    key = f"{m}_{fs}"

    difficulty_groups = {}
    
    if key not in pred_sets_all:
        return {}

    for item_id_str, pred_set in pred_sets_all[key].items():
        source = id_to_source.get(item_id_str)
        if source and 'AMBOSS_d' in source:
            match = re.search(r'd(\d)', source)
            if match:
                difficulty_level = f"d{match.group(1)}"
                if difficulty_level not in difficulty_groups:
                    difficulty_groups[difficulty_level] = []
                difficulty_groups[difficulty_level].append(len(pred_set))
    
    avg_sz_by_diff = {
        diff: np.mean(sizes) for diff, sizes in difficulty_groups.items()
    }
            
    return avg_sz_by_diff


def _uacc(acc_dict, set_size_dict, avg_k):
    return {
        k: acc_dict[k] * np.sqrt(avg_k) / set_size_dict[k]
        for k in acc_dict
        if set_size_dict[k] > 0
    }


def compute(logits_data_all, cal_raw, test_raw, prompt_methods, icl_methods, alpha=0.1):
    id2ans = convert_id_to_ans(test_raw)
    acc, e_rat, f_rat, abst_score = _get_accuracy(
        logits_data_all, test_raw, prompt_methods, icl_methods
    )

    avg_choices = np.mean([len(x["choices"]) for x in test_raw]) or 1

    ps_lac = LAC_CP(logits_data_all, cal_raw, prompt_methods, icl_methods, alpha)
    ps_aps = APS_CP(logits_data_all, cal_raw, prompt_methods, icl_methods, alpha)

    cov_lac = _coverage(ps_lac, id2ans, prompt_methods, icl_methods)
    cov_aps = _coverage(ps_aps, id2ans, prompt_methods, icl_methods)

    sz_lac = _set_size(ps_lac, prompt_methods, icl_methods)
    sz_aps = _set_size(ps_aps, prompt_methods, icl_methods)

    sz_lac_by_diff = {}
    sz_aps_by_diff = {}
    if test_raw and 'AMBOSS' in test_raw[0].get('source', ''):
        print("AMBOSS dataset detected, calculating set size by difficulty...")
        sz_lac_by_diff = _set_size_by_difficulty(ps_lac, test_raw, prompt_methods, icl_methods)
        sz_aps_by_diff = _set_size_by_difficulty(ps_aps, test_raw, prompt_methods, icl_methods)

    uacc_lac = _uacc(acc, sz_lac, avg_choices)
    uacc_aps = _uacc(acc, sz_aps, avg_choices)

    return {
        "Acc": acc,
        "E_rate": e_rat,
        "F_rate": f_rat,
        "abstention_rate": abst_score,
        "LAC_set_size": sz_lac,
        "APS_set_size": sz_aps,
        "LAC_coverage": cov_lac,
        "APS_coverage": cov_aps,
        "UAcc_LAC": uacc_lac,
        "UAcc_APS": uacc_aps,
        "LAC_set_size_by_difficulty": sz_lac_by_diff,
        "APS_set_size_by_difficulty": sz_aps_by_diff
    }