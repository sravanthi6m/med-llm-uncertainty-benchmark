import numpy as np
from collections import Counter

from .cp_sets import LAC_CP, APS_CP
from ..data_helpers.loaders import convert_id_to_ans


METRIC_NAME = "conformal_pred_summary"


def _get_accuracy(logits_data_all, test_raw, pm, icl):
    acc, e_ratio, f_ratio = {}, {}, {}
    for m in pm:
        for fs in icl:
            key = f"{m}_{fs}"
            res, preds = [], []
            for logit_row, raw_row in zip(logits_data_all[key]["test"], test_raw):
                opts = logit_row.get("option_keys_for_logits") \
                     or list(raw_row["choices"].keys())
                truth = raw_row["answer"]
                pred = opts[int(np.argmax(logit_row["logits_options"]))]
                preds.append(pred)
                res.append(int(pred == truth))
            acc[key] = np.mean(res)
            cts = Counter(preds)
            e_ratio[key] = cts.get("E", 0) / len(preds) if preds else 0
            f_ratio[key] = cts.get("F", 0) / len(preds) if preds else 0
    return acc, e_ratio, f_ratio


def _coverage(pred_sets_all, id2ans, pm, icl):
    cov = {}
    for m in pm:
        for fs in icl:
            key = f"{m}_{fs}"
            cov[key] = np.mean([id2ans[k] in v for k, v in
                                pred_sets_all[key].items()])
    return cov


def _set_size(pred_sets_all, pm, icl):
    sz = {}
    for m in pm:
        for fs in icl:
            key = f"{m}_{fs}"
            sz[key] = np.mean([len(v) for v in pred_sets_all[key].values()])
    return sz


def _uacc(acc_dict, set_size_dict, avg_k):
    return {k: acc_dict[k] * np.sqrt(avg_k) / set_size_dict[k]
            for k in acc_dict if set_size_dict[k] > 0}


def compute(logits_data_all, cal_raw, test_raw,
            prompt_methods, icl_methods, alpha=0.1):
    id2ans = convert_id_to_ans(test_raw)
    acc, e_rat, f_rat = _get_accuracy(
        logits_data_all, test_raw, prompt_methods, icl_methods)

    avg_choices = np.mean([len(x["choices"]) for x in test_raw]) or 1

    ps_lac = LAC_CP(logits_data_all, cal_raw,
                    prompt_methods, icl_methods, alpha)
    ps_aps = APS_CP(logits_data_all, cal_raw,
                    prompt_methods, icl_methods, alpha)

    cov_lac = _coverage(ps_lac, id2ans, prompt_methods, icl_methods)
    cov_aps = _coverage(ps_aps, id2ans, prompt_methods, icl_methods)

    sz_lac = _set_size(ps_lac, prompt_methods, icl_methods)
    sz_aps = _set_size(ps_aps, prompt_methods, icl_methods)

    uacc_lac = _uacc(acc, sz_lac, avg_choices)
    uacc_aps = _uacc(acc, sz_aps, avg_choices)

    return {
        "Acc": acc,
        "E_rate": e_rat,
        "F_rate": f_rat,
        "LAC_set_size": sz_lac,
        "APS_set_size": sz_aps,
        "LAC_coverage": cov_lac,
        "APS_coverage": cov_aps,
        "UAcc_LAC": uacc_lac,
        "UAcc_APS": uacc_aps,
    }

