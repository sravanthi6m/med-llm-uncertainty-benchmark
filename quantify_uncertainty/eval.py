import pkgutil
import importlib

from .data_helpers.loaders import (get_raw_data, get_logits_data)
import quantify_uncertainty.metrics as _metrics_pkg

def _discover_metric_modules():
    for info in pkgutil.iter_modules(_metrics_pkg.__path__):
        mod = importlib.import_module(f"{_metrics_pkg.__name__}.{info.name}")
        if hasattr(mod, "compute") and hasattr(mod, "METRIC_NAME"):
            yield mod


def evaluate(cfg):
    results = {}
    metric_mods = list(_discover_metric_modules())

    for dsets in cfg.data_names:
        cal_raw, test_raw = get_raw_data(cfg.raw_data_dir, dsets, cfg.cal_ratio)
        
        logits_all = get_logits_data(
            model_name=cfg.model, 
            data_name=dsets, 
            cal_raw_data=cal_raw, 
            test_raw_data=test_raw,
            logits_data_dir=cfg.logits_data_dir, 
            cal_ratio=cfg.cal_ratio,
            prompt_methods=cfg.prompt_methods, 
            icl_methods=cfg.icl_methods,
            k_few_shot=cfg.k_few_shot,
            dynamic_few_shot=cfg.dynamic_few_shot,
            cot=cfg.cot
        )

        per_dataset = {}
        for mod in metric_mods:
            out = mod.compute(logits_all, cal_raw, test_raw,
                              cfg.prompt_methods, cfg.icl_methods,
                              alpha=cfg.alpha)
            per_dataset.update({f"{mod.METRIC_NAME}.{k}": v
                                for k, v in out.items()})
        results[dsets] = per_dataset
    return results

