import importlib
import pkgutil

from core import BENCHMARKS

# Import each module so it self-registers into BENCHMARKS.
for _info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_info.name}")
