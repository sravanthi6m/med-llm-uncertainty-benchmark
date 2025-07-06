from importlib import import_module
import pkgutil as _pkgutil

import quantify_uncertainty.metrics as _metrics_pkg
for _info in _pkgutil.iter_modules(_metrics_pkg.__path__):
    import_module(f"{_metrics_pkg.__name__}.{_info.name}")

