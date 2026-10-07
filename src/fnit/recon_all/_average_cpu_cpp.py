"""Optional lazy CPU averaging; failed/unsupported builds retain NumBa.

Only the CPU wrapper imports this module. C++ source, flags and actual compiler
identity key a private cache. Each call uses the caller's current NumBa mask.
``backend_info`` reports the most recent call on the current Python thread.
"""
from contextlib import contextmanager
from copy import deepcopy
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np


_FLAGS = ('-O3', '-std=c++17', '-ffp-contract=off', '-fno-fast-math',
          '-fopenmp', '-fPIC', '-shared')
_ABI = 1
_BUILD_TIMEOUT = 60.0
_LOCK_TIMEOUT = 15.0
_COMPILERS = {}
_LIBRARIES = {}
_FAILED = {}
_MUTEX = threading.RLock()
_CALL = threading.local()


def backend_info():
    """Copy private provenance of the latest CPU attempt on this thread."""
    return deepcopy(getattr(_CALL, 'info', {'backend': None, 'reason': 'not called'}))


def _fallback(reason, **details):
    _CALL.info = {'backend': 'numba', 'reason': str(reason), **details}
    return None


def _compiler():
    selection = os.environ.get('CXX')
    if selection:
        arguments = shlex.split(selection)
        if len(arguments) != 1:
            raise RuntimeError('CXX must name one compiler executable')
        candidates = arguments
    else:
        prefixes = [os.environ.get('CONDA_PREFIX'), sys.prefix]
        names = ('x86_64-conda-linux-gnu-c++', 'x86_64-conda-linux-gnu-g++', 'c++', 'g++')
        candidates = [str(Path(prefix) / 'bin' / name) for prefix in prefixes if prefix for name in names]
        candidates += ['c++', 'g++']
    command = next((shutil.which(candidate) for candidate in candidates if shutil.which(candidate)), None)
    if command is None:
        raise RuntimeError('C++ compiler unavailable')
    path = Path(command).resolve(strict=True)
    metadata = path.stat()
    fingerprint = (str(path), metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
    if fingerprint not in _COMPILERS:
        version = subprocess.run([command, '--version'], capture_output=True, text=True,
                                 timeout=10, check=False)
        if version.returncode:
            raise RuntimeError('C++ compiler version probe failed')
        identity = {'command': command, 'resolved_path': str(path),
                    'binary_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'version': version.stdout.strip()}
        _COMPILERS[fingerprint] = identity
    return _COMPILERS[fingerprint]


def _cache_directory():
    selected = os.environ.get('FNIT_AVERAGE_CPU_CACHE')
    if selected:
        directory = Path(selected).expanduser().absolute()
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache')))
        directory = base / 'fnit' / 'average_cpu'
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    details = directory.lstat()
    if (not stat.S_ISDIR(details.st_mode) or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) & 0o077):
        raise RuntimeError('compiled averaging cache must be an owned private directory')
    return directory


@contextmanager
def _build_lock(directory, key):
    import fcntl  # Unsupported platforms return before reaching this import.
    descriptor = os.open(directory / (key + '.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        details = os.fstat(descriptor)
        if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) & 0o077):
            raise RuntimeError('invalid averaging build lock ownership or mode')
        deadline = time.monotonic() + _LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EAGAIN, errno.EACCES, errno.ENOLCK, errno.EINTR):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError('averaging cache build lock timed out (errno '
                                       + str(error.errno) + ')') from None
                time.sleep(.05)
        yield
    finally:
        os.close(descriptor)


def _private_bytes(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        details = os.fstat(stream.fileno())
        if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) & 0o077):
            raise RuntimeError('invalid averaging cache artifact ownership or mode')
        return stream.read()


def _valid_artifact(library, manifest, key):
    try:
        record = json.loads(_private_bytes(manifest))
        payload = _private_bytes(library)
        return (record['key'] == key and record['abi'] == _ABI
                and record['library_bytes'] == len(payload)
                and record['library_sha256'] == hashlib.sha256(payload).hexdigest())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _compile(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        output, error = process.communicate(timeout=_BUILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise TimeoutError('averaging C++ build timed out') from None
    if process.returncode:
        raise RuntimeError('averaging C++ build failed: ' + error.decode(errors='replace')[-600:])


def _bind(library):
    handle = ctypes.CDLL(str(library))
    try:
        abi = handle.fnit_average_persistent_abi_version
        function = handle.fnit_average_persistent
    except AttributeError:
        raise RuntimeError('averaging C++ ABI symbols missing') from None
    abi.argtypes = []; abi.restype = ctypes.c_int
    if abi() != _ABI:
        raise RuntimeError('averaging C++ ABI mismatch')
    function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return handle, function


def _library():
    with _MUTEX:
        source = Path(__file__).with_name('_average_cpu_persistent.cpp')
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        compiler = _compiler()
        identity = {'abi': _ABI, 'source_sha256': source_sha,
                    'compiler': compiler, 'flags': list(_FLAGS),
                    'platform': platform.platform(), 'machine': platform.machine()}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        # A new source/compiler/cache location makes a previously failed build retry.
        location = os.environ.get('FNIT_AVERAGE_CPU_CACHE', os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache')))
        token = (key, location)
        if token in _FAILED:
            raise RuntimeError(_FAILED[token])
        if token in _LIBRARIES:
            return _LIBRARIES[token]
        try:
            directory = _cache_directory()
            library, manifest = directory / (key + '.so'), directory / (key + '.json')
            with _build_lock(directory, key):
                bound = None
                if not _valid_artifact(library, manifest, key):
                    descriptor, temporary = tempfile.mkstemp(prefix='.build-', suffix='.so', dir=directory)
                    os.close(descriptor)
                    temporary = Path(temporary)
                    metadata_path = None
                    try:
                        _compile([compiler['command'], *_FLAGS, str(source), '-o', str(temporary)])
                        os.chmod(temporary, 0o600)
                        bound = _bind(temporary)  # Reject a failed link/ABI before publishing it.
                        payload = _private_bytes(temporary)
                        record = {**identity, 'key': key, 'library_bytes': len(payload),
                                  'library_sha256': hashlib.sha256(payload).hexdigest()}
                        descriptor, metadata_path = tempfile.mkstemp(prefix='.manifest-', suffix='.json', dir=directory)
                        with os.fdopen(descriptor, 'w') as stream:
                            json.dump(record, stream, sort_keys=True)
                            stream.flush(); os.fsync(stream.fileno())
                        os.replace(temporary, library)
                        os.replace(metadata_path, manifest)
                    finally:
                        temporary.unlink(missing_ok=True)
                        if metadata_path is not None:
                            Path(metadata_path).unlink(missing_ok=True)
                handle, function = bound if bound is not None else _bind(library)
            provenance = {**identity, 'cache_key': key, 'library': str(library)}
            result = handle, function, provenance
            _LIBRARIES[token] = result
            return result
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            _FAILED[token] = str(error)
            raise


def average_numpy_if_supported(gradient, neighbors, degrees, reciprocals, iterations, threads):
    """Return a new FP32 result, or None to preserve the original NumBa call."""
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        return _fallback('unsupported platform')
    if len(gradient) < 8192 or iterations < 16:
        return _fallback('small mesh or few averaging rounds')
    arrays = (gradient, neighbors, degrees, reciprocals)
    if (not all(isinstance(value, np.ndarray) for value in arrays)
            or gradient.dtype != np.float32 or neighbors.dtype != np.int64
            or degrees.dtype != np.int64 or reciprocals.dtype != np.float32
            or gradient.ndim != 2 or gradient.shape[1] != 3 or neighbors.ndim != 2
            or neighbors.shape[0] != len(gradient) or degrees.shape != (len(gradient),)
            or reciprocals.shape != degrees.shape
            or not isinstance(iterations, (int, np.integer)) or iterations > np.iinfo(np.int64).max
            or not isinstance(threads, (int, np.integer)) or not 1 <= threads <= np.iinfo(np.int32).max):
        return _fallback('unsupported array contract')
    if not np.isfinite(gradient).all() or not np.isfinite(reciprocals).all():
        return _fallback('nonfinite input retains NumBa bit-pattern policy')
    # NumPy copies preserve values/dtype and give the C ABI aligned contiguous buffers.
    arrays = [np.require(value, requirements=['C', 'A']) for value in arrays]
    try:
        handle, function, provenance = _library()
        output = np.empty_like(arrays[0])
        actual_threads = ctypes.c_int()
        status = function(*(value.ctypes.data for value in arrays), len(gradient), neighbors.shape[1],
                          iterations, threads, output.ctypes.data, ctypes.byref(actual_threads))
        if status:
            reason = 'non-default CPU floating policy' if status == 6 else 'averaging C++ status ' + str(status)
            return _fallback(reason, requested_threads=int(threads), actual_threads=actual_threads.value, **provenance)
        _CALL.info = {'backend': 'cpp', 'reason': 'persistent local OpenMP team',
                      'requested_threads': int(threads), 'actual_threads': actual_threads.value, **provenance}
        return output
    except (OSError, RuntimeError, TimeoutError, ValueError, subprocess.SubprocessError) as error:
        return _fallback(str(error))
