"""Explicit plotting environment for validation-only figures, no production fallback."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def ensure_plot_dependencies():
    root=Path(__file__).resolve().parent
    os.environ.setdefault('MPLCONFIGDIR',str(root/'work/matplotlib_cache'))
    if importlib.util.find_spec('matplotlib') is not None:return
    configuration=root/'plot_runtime.json'
    if not configuration.is_file():raise RuntimeError('matplotlib is declared in FNIT environment.yml; run this figure with the complete Conda environment or explicitly configure plot_runtime.json')
    executable=json.loads(configuration.read_text())['python']
    print(f'Validation figure uses explicitly configured plotting Python: {executable}',flush=True)
    raise SystemExit(subprocess.call([executable,*sys.argv]))
