"""Lazy CPU-only Eigen matrix square root, built from FNIT's own adapter."""
from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

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
    flags = ['-O2', '-std=c++14', '-fPIC', '-shared', '-DNDEBUG', '-ffp-contract=off']
    headers = {str(path.relative_to(include)): _sha256(path)
               for folder in (include / 'Eigen', include / 'unsupported/Eigen')
               for path in sorted(folder.rglob('*')) if path.is_file()}
    identity = {'source_sha256': _sha256(source), 'compiler_argv': compiler,
                'compiler_sha256': _sha256(compiler[0]),
                'compiler_version': subprocess.check_output([*compiler, '--version'], text=True),
                'flags': flags, 'eigen_headers_sha256': headers}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return include, compiler, source, flags, identity, key


def _cache_lock(file):
    """Retry only a transient shared-filesystem lock allocation failure."""
    import errno
    import fcntl
    import time
    for attempt in range(3):
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            return
        except OSError as error:
            if error.errno != errno.ENOLCK or attempt == 2:
                raise
            time.sleep(.05 * (attempt + 1))


@functools.lru_cache(maxsize=1)
def _library():
    # This module is imported only when a CPU joint inference needs sqrtm.
    include, compiler, source, flags, identity, key = _build_inputs()
    cache = Path(os.environ.get('FNIT_SYNTHMORPH_BUILD_CACHE',
                               Path.home() / '.cache/fnit/synthmorph/cpu_eigen')) / key
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / 'affine_sqrt.so'
    manifest = cache / 'build.json'
    with (cache / 'build.lock').open('w') as lock:
        _cache_lock(lock)
        if binary.exists() and manifest.exists():
            record = json.loads(manifest.read_text())
            if record.get('identity') != identity or record.get('binary_sha256') != _sha256(binary):
                raise RuntimeError('CPU SynthMorph build cache identity does not match its files')
        else:
            temporary = cache / 'affine_sqrt.building.so'
            command = [*compiler, *flags, '-I', str(include), str(source), '-o', str(temporary)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('CPU SynthMorph Eigen adapter build failed:\n' + result.stderr[-4000:])
            temporary.replace(binary)
            record = {'identity': identity, 'binary_sha256': _sha256(binary), 'compile_argv': command}
            manifest.write_text(json.dumps(record, indent=2) + '\n')
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
