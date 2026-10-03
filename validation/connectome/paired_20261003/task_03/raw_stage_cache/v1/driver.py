import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

source = Path(__file__).resolve().parent
os.chdir(source)
sys.path.insert(0, str(source / 'src'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sources():
    paths = sorted(p for p in (source / 'src').rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    identities = {str(p.relative_to(source)): sha(p) for p in paths}
    return {'count': len(identities), 'aggregate_sha256': hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}

selected = ['tests/connectome/test_raw_stage_cache_integrity.py',
            'tests/connectome/test_bids.py',
            'tests/connectome/test_cli_numerical_revision.py',
            'tests/connectome/test_paired_e2e.py']
started = time.perf_counter()
before = sources()
import torch
import fnit
import pytest
assert Path(fnit.__file__).resolve().is_relative_to(source / 'src')
torch.set_num_threads(4)
assert not torch.cuda.is_initialized()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''

report = {'git_commit': 'ac0e7d9d12bd730d33135d36d3f218990444d436',
          'archive_sha256': '99d1293d1979dc59bc197fedc6b430844fb51cb9c43ad653588bf7c77d3747f6',
          'scope': 'CPU raw-stage integrity/ordering regressions; stubbed numerical producers; not a real MRI/GPU benchmark',
          'hostname': socket.gethostname(), 'python': platform.python_version(),
          'torch': torch.__version__, 'fnit_import': str(Path(fnit.__file__).resolve()),
          'cuda_initialized_before': torch.cuda.is_initialized(),
          'environment': {k: os.environ.get(k) for k in (
              'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')},
          'cpu_affinity': sorted(os.sched_getaffinity(0)),
          'torch_num_threads': torch.get_num_threads(),
          'selected_tests': selected,
          'file_sha256': {name: sha(source / name) for name in (
              'src/fnit/connectome/bids.py', *selected)},
          'sources_before': before}
log = io.StringIO()
try:
    test_start = time.perf_counter()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = int(pytest.main(['-q', '-o', 'cache_dir=.raw_stage_pytest_cache', *selected]))
    report['pytest_main_seconds'] = time.perf_counter() - test_start
    report['pytest_exit_code'] = result
finally:
    report['sources_after'] = sources()
    report['sources_unchanged'] = report['sources_before'] == report['sources_after']
    report['cuda_initialized_after'] = torch.cuda.is_initialized()
    report['driver_seconds'] = time.perf_counter() - started
    (source / 'focused_cpu.log').write_text(log.getvalue())
    report['log_sha256'] = sha(source / 'focused_cpu.log')
    (source / 'focused_cpu_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(log.getvalue())
    print(json.dumps(report, sort_keys=True))
raise SystemExit(result if report['sources_unchanged'] and not report['cuda_initialized_after'] else 90)
