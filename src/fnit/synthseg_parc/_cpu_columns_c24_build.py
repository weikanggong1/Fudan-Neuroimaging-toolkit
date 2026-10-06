"""Private C24 cache using the accepted runtime, compiler and atomic cache helpers.

C72 build inputs and key remain untouched. The C24 identity adds its exact source
and exported symbols, and uses an independent cache directory and failure map.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading

from . import _cpu_columns_build as mature


_SOURCE_SHA = "75c232fa2c83eb28fb4ce62dab2eecfdfd6f9cdd5ed25b22ee430ca305731bb1"
_MUTEX = threading.RLock()
_BUILDS = {}
_FAILED = {}


def _cache_root():
    selected = os.environ.get("FNIT_SYNTHSEG_C24_CPU_CACHE")
    root = Path(selected).expanduser().absolute() if selected else mature._cache_root() / "c24"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    details = root.lstat()
    if (not stat.S_ISDIR(details.st_mode) or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) & 0o077):
        raise RuntimeError("C24 cache must be an owned private directory")
    return root


def _build_inputs(torch):
    # Reuse the accepted Torch/header/provider/GCC11 gate, without calling its
    # build_artifact or changing any C72 artifact/key/failure accounting.
    torch_root, _, compiler, accepted_identity, accepted_key = mature._build_inputs(torch)
    source = Path(__file__).with_name("_columns_c24.cpp")
    if mature._sha(source) != _SOURCE_SHA:
        raise RuntimeError("C24 source differs from the real-validated copy glue")
    identity = dict(accepted_identity)
    identity.update(source_sha256=_SOURCE_SHA, specialization="C24_depth32",
                    accepted_C72_build_key=accepted_key,
                    exports=["fnit_columns_c24_abi_description", "fnit_copy_columns_c24_f32",
                             "fnit_same_provider_sgemm_c24_f32"])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return torch_root, source, compiler, identity, key


def build_artifact(torch):
    """Return one independent C24 artifact, or fail before candidate mathematics."""
    with _MUTEX:
        torch_root, source, compiler, identity, key = _build_inputs(torch)
        cache = _cache_root()
        token = (key, str(cache))
        if token in _FAILED:
            raise RuntimeError(_FAILED[token])
        if token in _BUILDS:
            return dict(_BUILDS[token])
        library, manifest = cache / (key + ".so"), cache / (key + ".json")
        try:
            with mature._lock(cache, key):
                if not mature._valid_artifact(library, manifest, key):
                    with tempfile.TemporaryDirectory(prefix=".c24-build-", dir=cache) as temporary:
                        compiled = Path(temporary) / "columns_c24.so"
                        command = [compiler["command"], *mature._FLAGS,
                                   "-I" + str(torch_root / "include"),
                                   "-I" + str(torch_root / "include/torch/csrc/api/include"),
                                   str(source), "-L" + str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
                                   "-Wl,--no-undefined", "-Wl,-z,relro", "-Wl,-z,now",
                                   "-Wl,-rpath," + str(torch_root / "lib"),
                                   "-Wl,-rpath-link," + str(torch_root / "lib"),
                                   "-Wl,-rpath-link," + str(Path(sys.prefix) / "lib"), "-o", str(compiled)]
                        mature._compile(command)
                        os.chmod(compiled, 0o600)
                        from .cpu_columns_c24 import _ColumnsC24
                        checked = _ColumnsC24(compiled, mature._PROVIDER_SHA, allow_compute=False)
                        checked._provider_still_matches()
                        record = {"key": key, "abi": 10404, "identity": identity,
                                  "library_bytes": compiled.stat().st_size,
                                  "library_sha256": mature._sha(compiled)}
                        temporary_manifest = Path(temporary) / "artifact.json"
                        temporary_manifest.write_text(json.dumps(record, sort_keys=True) + "\n")
                        os.chmod(temporary_manifest, 0o600)
                        os.replace(compiled, library)
                        os.replace(temporary_manifest, manifest)
                if not mature._valid_artifact(library, manifest, key):
                    raise RuntimeError("C24 artifact failed post-build identity gate")
            result = {"key": key, "library": str(library), "provider_sha256": mature._PROVIDER_SHA}
            _BUILDS[token] = result
            return dict(result)
        except (OSError, RuntimeError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
            _FAILED[token] = str(error)
            raise RuntimeError(str(error)) from None
