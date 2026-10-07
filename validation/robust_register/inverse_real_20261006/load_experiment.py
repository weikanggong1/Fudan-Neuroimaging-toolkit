"""Load SHA-bound overlay or unchanged B package under two private namespaces.

The five unchanged B files are reused in place, without copying or editing.
No installed fnit.robust_register module, mature inverse or global function
is patched. Only the overlay's native_inverse calls its own local helper.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys


def load_package(plan, *, candidate):
    import fnit
    source = Path(plan['legacy_candidate_directory'])
    namespace = ('fnit._robust_inverse_validation_20261006' if candidate
                 else 'fnit._robust_inverse_score_legacy_20261006')
    if namespace in sys.modules:
        result = sys.modules[namespace]
        if result._validation_source_directory != str(source):
            raise ValueError('namespace was already bound to a different source')
        return result
    directories = ([plan['overlay_directory'], str(source)] if candidate
                   else [str(source)])
    specification = importlib.util.spec_from_file_location(
        namespace, source/'__init__.py', submodule_search_locations=directories)
    if specification is None or specification.loader is None:
        raise ImportError('cannot load the isolated SHA-bound experiment')
    package = importlib.util.module_from_spec(specification)
    package._validation_source_directory = str(source)
    sys.modules[namespace] = package
    try:
        specification.loader.exec_module(package)
    except BaseException:
        sys.modules.pop(namespace, None)
        raise
    return package


def actual_module_bindings():
    records = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(('fnit._robust_inverse_validation_20261006',
                            'fnit._robust_inverse_score_legacy_20261006')):
            filename = getattr(module, '__file__', None)
            if filename is not None:
                data = Path(filename).read_bytes()
                records[name] = {'path': filename, 'bytes': len(data),
                                 'sha256': hashlib.sha256(data).hexdigest()}
    return records
