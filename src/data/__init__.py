"""
Data module initialization.
Explicitly loads DataCleaner and generator to ensure compatibility across filesystems.
"""
import os
import sys
import importlib.util

curr_dir = os.path.dirname(os.path.abspath(__file__))

def _load_module(mod_name, filename):
    file_path = os.path.join(curr_dir, filename)
    if os.path.exists(file_path):
        spec = importlib.util.spec_from_file_location(f"src.data.{mod_name}", file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"src.data.{mod_name}"] = mod
        spec.loader.exec_module(mod)
        return mod
    return None

cleaner = _load_module("cleaner", "cleaner.py")
if cleaner:
    DataCleaner = cleaner.DataCleaner

generator = _load_module("generator", "generator.py")
