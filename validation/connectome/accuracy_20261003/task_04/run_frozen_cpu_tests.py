"""Run existing focused tests against explicitly bound repository modules."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

source = Path(sys.argv[1]).resolve()
for name, directory in (("fnit", source.parent), ("fnit.connectome", source)):
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    package.__spec__ = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)])
    sys.modules[name] = package

names = ("anatomy", "atlas_surface", "freesurfer_subject", "atlas_builder", "atlas_tian")
for name in names:
    spec = importlib.util.spec_from_file_location("fnit.connectome." + name, source / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
print(json.dumps({"actual_source_path": str(source), "files_sha256": {
    name: hashlib.sha256((source / (name + ".py")).read_bytes()).hexdigest()
    for name in names}}, indent=2), flush=True)
raise SystemExit(pytest.main(sys.argv[2:]))
