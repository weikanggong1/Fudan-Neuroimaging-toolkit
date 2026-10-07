"""Load this SHA-bound experiment under a separate FNIT namespace.

Does not replace fnit.robust_register, write production files, register an
image, or select a device. Runtime mathematical files retain as-run bytes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _verify(root, name, expected):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError("manifest path must stay inside its declared root")
    data = (root / relative).read_bytes()
    actual = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if actual != expected:
        raise ValueError("experiment or mature dependency source SHA changed: " + name)
    return actual


def load_candidate():
    """Return experimental package, checking own and mature source SHA first."""
    import fnit

    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'CANDIDATE_SOURCE_MANIFEST.json').read_text())
    for name, expected in manifest['runtime_files'].items():
        _verify(here, name, expected)
    mature_root = Path(fnit.__file__).resolve().parent
    for name, expected in manifest['mature_FNIT_dependency_files'].items():
        _verify(mature_root, name, expected)
    namespace = manifest['namespace']
    if namespace != 'fnit._robust_register_validation_20261006':
        raise ValueError("this loader only supports the separate experiment namespace")
    if namespace in sys.modules:
        return sys.modules[namespace]
    source = here / 'candidate_source' / 'robust_register'
    spec = importlib.util.spec_from_file_location(namespace, source / '__init__.py',
                                                submodule_search_locations=[str(source)])
    if spec is None or spec.loader is None:
        raise ImportError("cannot construct isolated experiment package")
    package = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = package
    try:
        spec.loader.exec_module(package)
    except BaseException:
        sys.modules.pop(namespace, None)
        raise
    return package
