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
    metric_mods = list(_discover_metric_modules())

    cal_raw, test_raw = get_raw_data(cfg.raw_data_file, cfg.cal_ratio)
    
    logits_all = get_logits_data(
        logits_pkl_path=cfg.logits_pkl_path,
        cal_ratio=cfg.cal_ratio,
        prompt_methods=cfg.prompt_methods, 
        icl_methods=cfg.icl_methods
    )

    per_dataset_results = {}
    for mod in metric_mods:
        out = mod.compute(logits_all, cal_raw, test_raw,
                            cfg.prompt_methods, cfg.icl_methods,
                            alpha=cfg.alpha)
        per_dataset_results.update({f"{mod.METRIC_NAME}.{k}": v
                            for k, v in out.items()})
    
    return per_dataset_results

