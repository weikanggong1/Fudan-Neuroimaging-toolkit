"""Lazy CPU-only Eigen matrix square root, built from FNIT's own adapter."""
from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid

import numpy as np


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_inputs():
    prefix = Path(os.environ.get('CONDA_PREFIX', sys.prefix))
    include = Path(os.environ.get('FNIT_EIGEN_INCLUDE', prefix / 'include/eigen3')).resolve()
    path_parts = {name.lower() for name in include.parts}
    if 'freesurfer' in path_parts or ('tensorflow' in path_parts and 'site-packages' in path_parts):
        raise ValueError('FNIT CPU Eigen requires independent Conda Eigen headers')
    if not (include / 'Eigen/Core').is_file():
        raise RuntimeError('CPU SynthMorph joint needs Eigen from the FNIT Conda environment')
    macros = include / 'Eigen/src/Core/util/Macros.h'
    text = macros.read_text() if macros.is_file() else ''
    version = tuple(int(match.group(1)) if match else -1 for match in (
        re.search(r'^\s*#\s*define\s+EIGEN_' + part + r'_VERSION\s+(\d+)\b', text, re.M)
        for part in ('WORLD', 'MAJOR', 'MINOR')))
    if version != (3, 4, 0):
        raise RuntimeError('CPU SynthMorph joint requires Eigen 3.4.0; install the FNIT Conda pin eigen=3.4.0 (found ' + '.'.join(map(str, version)) + ')')
    compiler = shlex.split(os.environ['CXX']) if os.environ.get('CXX') else []
    if not compiler:
        for name in ('x86_64-conda-linux-gnu-c++', 'x86_64-conda-linux-gnu-g++', 'g++', 'c++'):
            candidate = prefix / 'bin' / name
            located = str(candidate) if candidate.is_file() else shutil.which(name)
            if located:
                compiler = [located]
                break
    if not compiler:
        raise RuntimeError('CPU SynthMorph joint needs the C++ compiler from the FNIT Conda environment')
    executable = shutil.which(compiler[0])
    if not executable:
        raise RuntimeError('configured CPU SynthMorph C++ compiler was not found')
    compiler[0] = str(Path(executable).resolve())
    source = Path(__file__).with_name('_cpu_eigen_sqrt.cpp')
    # No native-architecture or fast-math flags: small FP32 matrix functions
    # retain the established baseline Eigen evaluation order.
    flags = ['-O2', '-std=c++14', '-fPIC', '-shared', '-DNDEBUG',
             '-fno-fast-math', '-ffp-contract=off']
    headers = {str(path.relative_to(include)): _sha256(path)
               for folder in (include / 'Eigen', include / 'unsupported/Eigen')
               for path in sorted(folder.rglob('*')) if path.is_file()}
    identity = {'source_sha256': _sha256(source), 'compiler_argv': compiler,
                'compiler_sha256': _sha256(compiler[0]),
                'compiler_version': subprocess.check_output([*compiler, '--version'], text=True, timeout=5),
                'flags': flags, 'eigen_headers_sha256': headers}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return include, compiler, source, flags, identity, key


def _cache_lock(file):
    """Bound lock contention and transient shared-filesystem failures."""
    import errno
    import fcntl
    deadline = time.monotonic() + 20
    allocation_failures = 0
    while True:
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as error:
            if error.errno == errno.ENOLCK:
                allocation_failures += 1
                if allocation_failures >= 3:
                    raise
            elif error.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError('CPU SynthMorph Eigen cache lock timed out') from error
            time.sleep(.05 * allocation_failures if error.errno == errno.ENOLCK else .05)


def _private_directory(path):
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    message = 'CPU SynthMorph cache directory must be owned by the current user and not a symlink'
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        raise RuntimeError(message) from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(message)
        # Some Conda Linux builds do not provide chmod(..., follow_symlinks=False).
        # Change the checked directory itself without following a path again.
        os.fchmod(descriptor, 0o700)
        if os.fstat(descriptor).st_mode & 0o777 != 0o700:
            raise RuntimeError('CPU SynthMorph cache filesystem must enforce directory mode 0700')
    finally:
        os.close(descriptor)


def _cache_file(path, flags, mode):
    descriptor = os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        os.close(descriptor)
        raise RuntimeError('CPU SynthMorph cache file must be private, owned and regular')
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, mode)


def _cache_digest(path):
    with _cache_file(path, os.O_RDONLY, 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def _compile(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        _, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired as error:
        # The new session contains only this compiler and its subprocesses.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                # A compiler blocked in kernel I/O may not reap immediately.
                # Keep the caller's timeout bounded; the group is signalled.
                process.stdout.close()
                process.stderr.close()
        raise TimeoutError('CPU SynthMorph Eigen compilation exceeded 60 seconds') from error
    if process.returncode:
        raise RuntimeError('CPU SynthMorph Eigen adapter build failed:\n' + stderr[-4000:])


@functools.lru_cache(maxsize=1)
def _library():
    # This module is imported only when a CPU joint inference needs sqrtm.
    include, compiler, source, flags, identity, key = _build_inputs()
    root = Path(os.environ.get('FNIT_SYNTHMORPH_BUILD_CACHE',
                               Path.home() / '.cache/fnit/synthmorph/cpu_eigen')).expanduser()
    _private_directory(root)
    cache = root / key
    _private_directory(cache)
    binary = cache / 'affine_sqrt.so'
    manifest = cache / 'build.json'
    with _cache_file(cache / 'build.lock', os.O_CREAT | os.O_RDWR, 'r+b') as lock:
        _cache_lock(lock)
        if binary.exists() and manifest.exists():
            with _cache_file(manifest, os.O_RDONLY, 'r') as stream:
                record = json.load(stream)
            if record.get('identity') != identity or record.get('binary_sha256') != _cache_digest(binary):
                raise RuntimeError('CPU SynthMorph build cache identity does not match its files')
        else:
            temporary = cache / ('affine_sqrt.' + uuid.uuid4().hex + '.building.so')
            with _cache_file(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 'wb'):
                pass
            command = [*compiler, *flags, '-I', str(include), str(source), '-o', str(temporary)]
            try:
                _compile(command)
                binary_sha256 = _cache_digest(temporary)
                temporary.replace(binary)
            finally:
                temporary.unlink(missing_ok=True)
            record = {'identity': identity, 'binary_sha256': binary_sha256, 'compile_argv': command}
            temporary_manifest = cache / ('build.' + uuid.uuid4().hex + '.json')
            try:
                with _cache_file(temporary_manifest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 'w') as stream:
                    json.dump(record, stream, indent=2)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary_manifest.replace(manifest)
            finally:
                temporary_manifest.unlink(missing_ok=True)
    library = ctypes.CDLL(str(binary))
    function = library.fnit_cpu_affine_sqrt
    function.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    function.restype = ctypes.c_int
    return library, function


def affine_sqrt(matrix):
    """Return an independent FP32 buffer containing each four-by-four root."""
    values = np.ascontiguousarray(matrix, dtype=np.float32)
    if values.shape[-2:] != (4, 4):
        raise ValueError('CPU joint square root requires four-by-four matrices')
    output = np.empty_like(values)
    _, function = _library()
    pointer = ctypes.POINTER(ctypes.c_float)
    status = function(values.ctypes.data_as(pointer), output.ctypes.data_as(pointer), values.size // 16)
    if status:
        raise ValueError('CPU joint affine has no finite real principal square root')
    return output
