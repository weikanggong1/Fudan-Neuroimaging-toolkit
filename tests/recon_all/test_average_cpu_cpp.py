"""Optional CPU build/cache safety and caller contracts, not benchmarks."""
import json
import errno
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numba
import numpy as np
import pytest
import torch

from fnit.recon_all import _average_cpu_cpp as cpp
from fnit.recon_all import mris_register_average_numba as averaging


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('FNIT_AVERAGE_CPU_CACHE', str(tmp_path / 'cache'))
    cpp._LIBRARIES.clear(); cpp._FAILED.clear()
    previous_torch, previous_numba = torch.get_num_threads(), numba.get_num_threads()
    yield
    torch.set_num_threads(previous_torch); numba.set_num_threads(previous_numba)


def arrays(size=8192):
    vertices = np.arange(size, dtype=np.int64)
    neighbors = np.stack(((vertices + 1) % size, (vertices - 1) % size,
                          np.full(size, -1, dtype=np.int64)), axis=1)
    degrees = np.full(size, 2, dtype=np.int64)
    degrees[::13] = 0; degrees[::17] = 1
    gradient = np.random.default_rng(25).standard_normal((size, 3)).astype(np.float32)
    gradient[::7] *= np.float32(1e8)
    reciprocal = (torch.from_numpy(degrees) + 1).float().reciprocal().numpy()
    return gradient, neighbors, degrees, reciprocal


def require_compiler():
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        pytest.skip('optional compiled path supports Linux x86-64')
    try:
        cpp._compiler()
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        pytest.skip(str(error))


@pytest.mark.parametrize('threads', [1, 2, 4, 8])
@pytest.mark.parametrize('iterations', [16, 17, 256])
def test_compiled_matches_ordered_numba_bits_and_preserves_inputs(threads, iterations):
    require_compiler()
    if threads > numba.config.NUMBA_NUM_THREADS:
        pytest.skip('thread count exceeds configured NumBa pool')
    values = arrays()
    # The C boundary also accepts noncontiguous input without modifying it.
    backing = np.empty((len(values[0]), 6), dtype=np.float32)
    backing[:, ::2] = values[0]
    values = (backing[:, ::2], *values[1:])
    before = [value.copy() for value in values]
    numba.set_num_threads(threads)
    expected = averaging._average_numpy(*values, iterations)
    torch.set_num_threads(threads)
    actual = averaging.average_gradients_exact_cpu(
        *(torch.from_numpy(value) for value in values[:3]), iterations)
    info = cpp.backend_info()
    assert info['backend'] == 'cpp', info
    assert info['requested_threads'] == threads
    assert 1 <= info['actual_threads'] <= threads
    assert np.array_equal(actual.numpy().view(np.uint32), expected.view(np.uint32))
    assert actual.dtype == torch.float32 and actual.is_contiguous()
    assert actual.data_ptr() != torch.from_numpy(values[0]).data_ptr()
    assert all(np.array_equal(value, old) for value, old in zip(values, before))
    assert numba.get_num_threads() == threads


@pytest.mark.parametrize('reason', ['compiler', 'platform', 'few_rounds', 'small_mesh', 'float64'])
def test_unsupported_attempt_returns_original_numba_without_build(monkeypatch, reason):
    values = arrays(32 if reason == 'small_mesh' else 8192)
    iterations = 1 if reason == 'few_rounds' else 16
    if reason == 'compiler':
        monkeypatch.setenv('CXX', '/compiler-that-does-not-exist')
    elif reason == 'platform':
        monkeypatch.setattr(cpp.platform, 'system', lambda: 'Unsupported')
    elif reason == 'float64':
        values = (values[0].astype(np.float64), *values[1:])
    monkeypatch.setattr(cpp, '_compile', lambda command: pytest.fail('fallback tried compiling'))
    observed = []
    def fallback(*args):
        observed.append(numba.get_num_threads())
        return args[0].copy()
    monkeypatch.setattr(averaging, '_average_numpy', fallback)
    torch.set_num_threads(min(3, numba.config.NUMBA_NUM_THREADS))
    previous = numba.get_num_threads()
    actual = averaging.average_gradients_exact_cpu(*(torch.from_numpy(v) for v in values[:3]), iterations)
    assert np.array_equal(actual.numpy(), values[0])
    assert observed == [1 if reason == 'small_mesh' else min(3, numba.config.NUMBA_NUM_THREADS)]
    assert numba.get_num_threads() == previous
    assert cpp.backend_info()['backend'] == 'numba'


def test_failed_build_once_and_flags_invalidate_failure(monkeypatch):
    require_compiler(); values = arrays(); attempts = []
    def failed(command):
        attempts.append(command)
        raise RuntimeError('controlled compiler failure')
    monkeypatch.setattr(cpp, '_compile', failed)
    for _ in range(2):
        assert cpp.average_numpy_if_supported(*values, 16, 1) is None
    assert len(attempts) == 1
    assert 'controlled compiler failure' in cpp.backend_info()['reason']
    monkeypatch.setattr(cpp, '_FLAGS', (*cpp._FLAGS, '-DNEW_CACHE_IDENTITY=1'))
    assert cpp.average_numpy_if_supported(*values, 16, 1) is None
    assert len(attempts) == 2


def test_compiler_identity_invalidates_failed_build(monkeypatch):
    require_compiler(); values = arrays(); identity = dict(cpp._compiler()); attempts = []
    monkeypatch.setattr(cpp, '_compiler', lambda: identity)
    def failed(command):
        attempts.append(command); raise RuntimeError('controlled failure')
    monkeypatch.setattr(cpp, '_compile', failed)
    assert cpp.average_numpy_if_supported(*values, 16, 1) is None
    identity['version'] += ' changed compiler identity'
    assert cpp.average_numpy_if_supported(*values, 16, 1) is None
    assert len(attempts) == 2


@pytest.mark.parametrize('kind', ['public_directory', 'symlink_directory', 'symlink_lock'])
def test_unsafe_cache_falls_back_without_compile(tmp_path, monkeypatch, kind):
    require_compiler(); values = arrays(); directory = tmp_path / 'cache'
    if kind == 'public_directory':
        directory.mkdir(mode=0o755)
    elif kind == 'symlink_directory':
        target = tmp_path / 'target'; target.mkdir(mode=0o700); directory.symlink_to(target)
    else:
        directory.mkdir(mode=0o700)
        identity = {'abi': cpp._ABI, 'source_sha256': __import__('hashlib').sha256(Path(cpp.__file__).with_name('_average_cpu_persistent.cpp').read_bytes()).hexdigest(),
                    'compiler': cpp._compiler(), 'flags': list(cpp._FLAGS),
                    'platform': cpp.platform.platform(), 'machine': cpp.platform.machine()}
        key = __import__('hashlib').sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        (directory / (key + '.lock')).symlink_to(tmp_path / 'foreign')
    monkeypatch.setattr(cpp, '_compile', lambda command: pytest.fail('unsafe cache tried compiling'))
    assert cpp.average_numpy_if_supported(*values, 16, 1) is None
    assert cpp.backend_info()['backend'] == 'numba'


def test_cache_manifest_corruption_rebuilds_atomically_and_info_is_copy(monkeypatch):
    require_compiler(); values = arrays()
    actual = cpp.average_numpy_if_supported(*values, 16, 1)
    info = cpp.backend_info(); assert actual is not None, info
    info['compiler']['version'] = 'external mutation'
    assert cpp.backend_info()['compiler']['version'] != 'external mutation'
    library = Path(info['library']); manifest = library.with_suffix('.json')
    manifest.write_text('{broken')  # Never truncate a library that is currently mapped.
    cpp._LIBRARIES.clear()
    original = cpp._compile; observed = []
    def compile_once(command):
        observed.append(command); original(command)
    monkeypatch.setattr(cpp, '_compile', compile_once)
    again = cpp.average_numpy_if_supported(*values, 16, 1)
    assert again is not None and np.array_equal(actual.view(np.uint32), again.view(np.uint32))
    assert len(observed) == 1
    assert json.loads(manifest.read_text())['key'] == info['cache_key']
    assert not list(library.parent.glob('.build-*'))
    assert not list(library.parent.glob('.manifest-*'))


def test_compile_timeout_kills_own_group(monkeypatch):
    monkeypatch.setattr(cpp, '_BUILD_TIMEOUT', .1)
    with pytest.raises(TimeoutError, match='timed out'):
        cpp._compile([sys.executable, '-c', 'import time; time.sleep(30)'])


@pytest.mark.parametrize('exhausted', [False, True])
def test_gpfs_lock_enolck_retries_with_bounded_deadline(tmp_path, monkeypatch, exhausted):
    import fcntl
    directory = tmp_path / 'lock-cache'; directory.mkdir(mode=0o700)
    original = fcntl.flock; attempts = []
    def transient_lock(descriptor, operation):
        attempts.append(operation)
        if exhausted or len(attempts) < 3:
            raise OSError(errno.ENOLCK, 'controlled GPFS lock shortage')
        return original(descriptor, operation)
    monkeypatch.setattr(fcntl, 'flock', transient_lock)
    if exhausted:
        monkeypatch.setattr(cpp, '_LOCK_TIMEOUT', .01)
        with pytest.raises(TimeoutError, match='errno 37'):
            with cpp._build_lock(directory, 'test'): pytest.fail('unlocked build started')
        assert len(attempts) == 2
    else:
        with cpp._build_lock(directory, 'test'): pass
        assert len(attempts) == 3


def test_cross_process_build_has_one_atomic_publisher(tmp_path, monkeypatch):
    require_compiler()
    compiler = cpp._compiler()['command']; counter = tmp_path / 'compile_count'
    wrapper = tmp_path / 'compiler'
    wrapper.write_text(f'#!{sys.executable}\nimport os,sys\n'
                       f'if "--version" not in sys.argv:\n with open({str(counter)!r},"a") as f:f.write("1\\n")\n'
                       f'os.execv({compiler!r},[{compiler!r},*sys.argv[1:]])\n')
    wrapper.chmod(0o700)
    environment = dict(os.environ, CXX=str(wrapper), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    program = '''import numpy as np
from fnit.recon_all import _average_cpu_cpp as c
n=8192
g=np.zeros((n,3),np.float32); nb=np.empty((n,0),np.int64); d=np.zeros(n,np.int64)
assert c.average_numpy_if_supported(g,nb,d,np.ones(n,np.float32),16,1) is not None,c.backend_info()
assert c.backend_info()['backend']=='cpp'
'''
    workers = [subprocess.Popen([sys.executable, '-c', program], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    try:
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=60)
            assert worker.returncode == 0, (stdout, stderr)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill(); worker.wait()
    assert counter.read_text().splitlines() == ['1']


@pytest.mark.parametrize('flush', [False, True])
def test_special_float_bits_and_flush_policy_in_isolated_process(flush):
    require_compiler()
    # Avoid leaving altered worker floating environments in the parent test pool.
    program = f'''import numpy as np,torch,numba
from fnit.recon_all import _average_cpu_cpp as c
from fnit.recon_all import mris_register_average_numba as a
probe=torch.from_numpy(np.array([1],dtype=np.uint32).view(np.float32))
previous=bool((probe*1).view(torch.int32)[0]==0)
try:
 torch.set_flush_denormal({flush!r});torch.set_num_threads(2);numba.set_num_threads(2)
 n=8192;v=np.arange(n,dtype=np.int64)
 nb=np.stack(((v+1)%n,(v-1)%n),axis=1);d=np.full(n,2,np.int64);r=np.full(n,np.float32(1/3),np.float32)
 bits=np.resize(np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff],np.uint32),(n,3))
 g=bits.view(np.float32);before=g.view(np.uint32).copy()
 expected=a._average_numpy(g,nb,d,r,16)
 result=a.average_gradients_exact_cpu(torch.from_numpy(g),torch.from_numpy(nb),torch.from_numpy(d),16)
 assert np.array_equal(result.numpy().view(np.uint32),expected.view(np.uint32)),c.backend_info()
 assert np.array_equal(g.view(np.uint32),before)
 if {flush!r}:assert c.backend_info()['backend']=='numba',c.backend_info()
 else:assert c.backend_info()['backend']=='cpp',c.backend_info()
 for special in [0x7fc01234,0xffc02345,0x7f800000,0xff800000]:
  special_g=g.copy();special_g.view(np.uint32)[0,0]=special;before=special_g.view(np.uint32).copy()
  expected=a._average_numpy(special_g,nb,d,r,16)
  result=a.average_gradients_exact_cpu(torch.from_numpy(special_g),torch.from_numpy(nb),torch.from_numpy(d),16)
  assert np.array_equal(result.numpy().view(np.uint32),expected.view(np.uint32)),c.backend_info()
  assert np.array_equal(special_g.view(np.uint32),before)
  assert c.backend_info()['backend']=='numba',c.backend_info()
finally:torch.set_flush_denormal(previous)
'''
    subprocess.run([sys.executable, '-c', program], check=True, env=os.environ.copy(), timeout=60)
