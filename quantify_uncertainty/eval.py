import pkgutil
import importlib

from .data_helpers.loaders import (get_raw_data, get_logits_data)
import quantify_uncertanty.metrics as _metrics_pkg

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
        logits_all = get_logits_data(cfg.model, dsets, cal_raw, test_raw,
                                     cfg.logits_data_dir, cfg.cal_ratio,
                                     cfg.prompt_methods, cfg.icl_methods)

        per_dataset = {}
        for mod in metric_mods:
            out = mod.compute(logits_all, cal_raw, test_raw,
                              cfg.prompt_methods, cfg.icl_methods,
                              alpha=cfg.alpha)
            per_dataset.update({f"{mod.METRIC_NAME}.{k}": v
                                for k, v in out.items()})
        results[dsets] = per_dataset
    return results

