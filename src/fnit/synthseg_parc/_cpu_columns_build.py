"""Lazy private CPU build cache for FNIT's own columns-copy glue.

Imported only after the exact CPU inference guard. Existing Torch supplies
the installed headers and loaded LP64 SGEMM; this module never finds or loads
another BLAS. Failed preparation retains the mature convolution before math.
"""
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time


_SOURCE_SHA = "2440abe802f1da2c14bd192f8b0f84c46b6f1f518de1dc9f0dca8429b05a2527"
_TORCH_CPU_SHA = "529f5ab9d2397b671c4768e8ad2865c626352909fd6d76b4b6b57622b39a874c"
_PROVIDER_SHA = "bd259733d7a3044c0656f32325df145e63ca52aba71560db0db2e991a138ee16"
_HEADERS = {
    "ATen/Config.h": "317951cca3faf477f49632e48427436b9b9461c1a12943d1e1f8ba402e5082bd",
    "ATen/Parallel-inl.h": "3f5f0b88547cf2a3a5296b3712ae9ba7f2b7b22e9598dbf2d36918b084f7b473",
    "ATen/Parallel.h": "d45ea50942d5e51f5b90f236e13c57fa6e980c1b4134eeba11f812e934b80513",
    "ATen/ParallelOpenMP.h": "547137c59ba1f88da28c1953069b155080147cca4edd741e6a3309eedd08f9aa",
    "ATen/native/CPUBlas.h": "65d41851492e355c08af2c73be4b2ad7d09f1d0af0ca3706a8a91dce78513899",
    "ATen/native/Unfold3d.h": "bbee13e24d17450d3dace10e38774a1463e74cc59bdf34afaf1e6c6bd4fa9500",
    "c10/core/CPUAllocator.h": "714c17e13a79842d52eb6172a4b8551a48ac7db8d0f969b6ece8f335dda3d9c7",
    "c10/core/impl/alloc_cpu.h": "afbdba37984811ece9442f9c746b92249ec50f5c2d2ef84abd54bc268427b350",
}
_FLAGS = ("-std=c++17", "-shared", "-fPIC", "-O2", "-fno-fast-math",
          "-ffp-contract=off", "-fopenmp", "-D_GLIBCXX_USE_CXX11_ABI=0")
_MUTEX = threading.RLock()
_BUILDS = {}
_FAILED = {}


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def _compiler():
    if os.environ.get("CXX"):
        values = shlex.split(os.environ["CXX"])
        if len(values) != 1:
            raise RuntimeError("CXX must name one compiler executable")
        candidates = values
    else:
        prefixes = (os.environ.get("CONDA_PREFIX"), sys.prefix)
        names = ("x86_64-conda-linux-gnu-c++", "x86_64-conda-linux-gnu-g++")
        candidates = [str(Path(prefix) / "bin" / name) for prefix in prefixes if prefix for name in names]
        candidates += list(names)
    selected = next((shutil.which(value) for value in candidates if shutil.which(value)), None)
    if selected is None:
        raise RuntimeError("CPU columns compiler unavailable; use FNIT Conda cxx-compiler/gxx_linux-64")
    actual = Path(selected).resolve(strict=True)
    version = subprocess.run([selected, "--version"], capture_output=True, text=True, timeout=10, check=False)
    if version.returncode:
        raise RuntimeError("CPU columns compiler version probe failed")
    if re.search(r"\b11\.\d+\.\d+\b", version.stdout) is None:
        raise RuntimeError("CPU columns requires the accepted GCC11 Conda toolchain")
    return {"command": selected, "resolved_path": str(actual),
            "binary_sha256": _sha(actual), "version": version.stdout.strip()}


def _cache_root():
    selection = os.environ.get("FNIT_SYNTHSEG_CPU_CACHE")
    root = Path(selection).expanduser().absolute() if selection else (
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "fnit" / "synthseg_columns")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    details = root.lstat()
    if (not stat.S_ISDIR(details.st_mode) or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) & 0o077):
        raise RuntimeError("columns cache must be an owned private directory")
    return root


@contextmanager
def _lock(root, key):
    import fcntl
    descriptor = os.open(root / (key + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        details = os.fstat(descriptor)
        if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) & 0o077):
            raise RuntimeError("invalid columns build lock")
        deadline = time.monotonic() + 15
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EAGAIN, errno.EACCES, errno.ENOLCK, errno.EINTR):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError("columns build lock timed out") from None
                time.sleep(.05)
        yield
    finally:
        os.close(descriptor)


def _private_bytes(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) & 0o077):
            raise RuntimeError("invalid private columns artifact")
        return stream.read()


def _valid_artifact(library, manifest, key):
    try:
        record = json.loads(_private_bytes(manifest))
        data = _private_bytes(library)
        return (record["key"] == key and record["abi"] == 10404
                and record["library_bytes"] == len(data)
                and record["library_sha256"] == hashlib.sha256(data).hexdigest())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False


def _compile(command):
    # Only this compiler subprocess sees these overrides; caller globals stay intact.
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True, env=environment)
    try:
        output, error = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Never block indefinitely while a compiler is stuck in kernel I/O.
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise TimeoutError("columns compiler timed out") from None
    if process.returncode:
        raise RuntimeError("columns compiler failed: " + error.decode(errors="replace")[-600:])


def _build_inputs(torch):
    if (sys.platform != "linux" or platform.machine() != "x86_64"
            or torch.__version__ != "2.5.1" or torch._C._GLIBCXX_USE_CXX11_ABI):
        raise RuntimeError("columns runtime is outside the accepted Linux/Torch2.5.1 ABI0 scope")
    root = Path(torch.__file__).resolve().parent
    source = Path(__file__).with_name("_columns_reuse.cpp")
    if _sha(source) != _SOURCE_SHA or _sha(root / "lib/libtorch_cpu.so") != _TORCH_CPU_SHA:
        raise RuntimeError("columns source or accepted Torch CPU runtime differs")
    for name, expected in _HEADERS.items():
        if _sha(root / "include" / name) != expected:
            raise RuntimeError("columns installed Torch headers differ")
    # Read only the already-loaded Torch handle; never load another BLAS provider.
    from .cpu_columns import _Columns
    provider = _Columns(None, _PROVIDER_SHA, allow_compute=False)
    provider._provider_still_matches()
    compiler = _compiler()
    identity = {"abi": 10404, "source_sha256": _SOURCE_SHA, "Torch_CPU_sha256": _TORCH_CPU_SHA,
                "Torch_headers": _HEADERS, "provider_sha256": _PROVIDER_SHA,
                "compiler": compiler, "flags": list(_FLAGS), "platform": sys.platform,
                "machine": platform.machine(), "Torch_root": str(root)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return root, source, compiler, identity, key


def build_artifact(torch):
    """Compile/load preparation only; returns an identity-keyed private artifact."""
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
            with _lock(cache, key):
                if not _valid_artifact(library, manifest, key):
                    with tempfile.TemporaryDirectory(prefix=".columns-build-", dir=cache) as temporary:
                        compiled = Path(temporary) / "columns.so"
                        command = [compiler["command"], *_FLAGS,
                                   "-I" + str(torch_root / "include"),
                                   "-I" + str(torch_root / "include/torch/csrc/api/include"),
                                   str(source), "-L" + str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
                                   "-Wl,--no-undefined", "-Wl,-z,relro", "-Wl,-z,now",
                                   "-Wl,-rpath," + str(torch_root / "lib"),
                                   "-Wl,-rpath-link," + str(torch_root / "lib"),
                                   "-Wl,-rpath-link," + str(Path(sys.prefix) / "lib"), "-o", str(compiled)]
                        _compile(command)
                        os.chmod(compiled, 0o600)
                        from .cpu_columns import _Columns
                        # ABI/provider-only binding: allow_compute=False, zero tensor work.
                        checked = _Columns(compiled, _PROVIDER_SHA, allow_compute=False)
                        checked._provider_still_matches()
                        record = {"key": key, "abi": 10404, "identity": identity,
                                  "library_bytes": compiled.stat().st_size, "library_sha256": _sha(compiled)}
                        temporary_manifest = Path(temporary) / "artifact.json"
                        temporary_manifest.write_text(json.dumps(record, sort_keys=True) + "\n")
                        os.chmod(temporary_manifest, 0o600)
                        os.replace(compiled, library)
                        os.replace(temporary_manifest, manifest)
                if not _valid_artifact(library, manifest, key):
                    raise RuntimeError("columns artifact failed post-build identity gate")
            result = {"key": key, "library": str(library), "provider_sha256": _PROVIDER_SHA}
            _BUILDS[token] = result
            return dict(result)
        except (OSError, RuntimeError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
            _FAILED[token] = str(error)
            raise RuntimeError(str(error)) from None
