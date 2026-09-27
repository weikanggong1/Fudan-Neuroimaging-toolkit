"""Run from a Git checkout to deploy pinned public HCP surface templates."""

import importlib.util
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[1] / "src/fnit/fmri/assets_setup.py"
_SPEC = importlib.util.spec_from_file_location("fmri_assets_setup", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


if __name__ == "__main__":
    _MODULE.main()
