"""
Create prediction sets - LAC and APS
"""

import numpy as np

from ..utils import softmax

def LAC_CP(logits_data_all, cal_raw, prompt_methods, icl_methods, alpha=0.1):
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            pred_sets_all[key] = {}
            
            cal_scores = []
            for idx, row in enumerate(logits_data_all[key]["cal"]):
                opts = row.get("option_keys_for_logits") \
                       or list(cal_raw[idx]["choices"].keys())
                probs = softmax(row["logits_options"])
                truth = cal_raw[idx]["answer"]
                try:
                    truth_idx = opts.index(truth)
                    cal_scores.append(1 - probs[truth_idx])
                except ValueError:
                    continue
            n = len(logits_data_all[key]["cal"])
            qhat = 1.0 if not cal_scores else np.quantile(
                cal_scores,
                np.ceil((n + 1) * (1 - alpha)) / n,
                method="higher",
            )
            
            for row in logits_data_all[key]["test"]:
                opts = row["option_keys_for_logits"]
                probs = softmax(row["logits_options"])
                ps = [o for o, p in zip(opts, probs) if p >= 1 - qhat]
                if not ps:
                    ps.append(opts[int(np.argmax(probs))])
                pred_sets_all[key][str(row["id"])] = ps
    return pred_sets_all


def APS_CP(logits_data_all, cal_raw, prompt_methods, icl_methods, alpha=0.1):
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            pred_sets_all[key] = {}
            
            cal_scores = []
            for idx, row in enumerate(logits_data_all[key]["cal"]):
                opts = row.get("option_keys_for_logits") \
                       or list(cal_raw[idx]["choices"].keys())
                probs = softmax(row["logits_options"])
                pi = np.argsort(probs)[::-1]
                cumsum = np.take_along_axis(probs, pi, axis=0).cumsum()
                cumsum_r = np.take_along_axis(cumsum, pi.argsort(), axis=0)
                truth = cal_raw[idx]["answer"]
                try:
                    cal_scores.append(cumsum_r[opts.index(truth)])
                except ValueError:
                    continue
            n = len(logits_data_all[key]["cal"])
            qhat = 1e-9 if not cal_scores else np.quantile(
                cal_scores,
                np.ceil((n + 1) * (1 - alpha)) / n,
                method="higher",
            )
            
            for row in logits_data_all[key]["test"]:
                opts = row["option_keys_for_logits"]
                probs = softmax(row["logits_options"])
                pi = np.argsort(probs)[::-1]
                cumsum = np.take_along_axis(probs, pi, axis=0).cumsum()
                ps = [opts[p] for p, s in zip(pi, cumsum) if s <= qhat]
                if not ps:
                    ps.append(opts[int(pi[0])])
                pred_sets_all[key][str(row["id"])] = ps
    return pred_sets_all

