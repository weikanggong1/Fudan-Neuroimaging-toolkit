"""Compare complete DicL fits on the same real 2,050-person projections.

python benchmark_dicl_real2050.py PROJECTED_DIR CPU_BASELINE_DIR OUTPUT_DIR

CPU_BASELINE_DIR is a completed prior same-input benchmark; use - to fit a
fresh CPU control. Reused CPU times are explicitly marked. Only aggregate.json is suitable for publication. Dictionaries stay on the
private validation host. This uses the production GPU implementation; only
solver counters are instrumented.
"""
from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import sklearn
import torch

import fnit.dictionary_learning.torch_backend as dicl_module
from fnit.bigflica.pipeline import _fit_flica, _spatial_z
from fnit.dictionary_learning import fit_dicl


MODALITIES = ('vbm', 'fa', 'md')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def gpu_status():
    result = subprocess.run(['nvidia-smi', '-i', '1',
                             '--query-gpu=utilization.gpu,memory.used,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return result.stdout.strip()


def compare(reference, candidate):
    norm = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    return {'relative_frobenius_error': float(np.linalg.norm(reference-candidate)/np.linalg.norm(reference)),
            'max_absolute_error': float(np.abs(reference-candidate).max()),
            'same_index_abs_cosine_min': float(np.abs((reference*candidate).sum(1)/norm).min())}


def correlation(first, second):
    first = np.asarray(first, dtype=np.float64).copy()
    second = np.asarray(second, dtype=np.float64).copy()
    first -= first.mean(0); second -= second.mean(0)
    return ((first*second).sum(0)/(np.linalg.norm(first,axis=0)*np.linalg.norm(second,axis=0))).tolist()


def main():
    projected, cpu_baseline, output = map(Path, sys.argv[1:])
    baseline = (None if str(cpu_baseline) == '-' else
                json.loads((cpu_baseline/'aggregate.json').read_text()))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'dataset': '2050 real subjects; full masks VBM/FA/MD; task excluded',
              'control': 'Identical persisted float32 mMIGP R100 values cast to float64 for CPU sklearn and GPU DicL',
              'parameters': {'dicl_dim': 200, 'max_iter': 1000, 'batch_size': 32,
                             'sparse_iterations': 120, 'alpha': 1., 'random_state': 0, 'cpu_threads': 8},
              'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
              'gpu_name': torch.cuda.get_device_name(), 'shared_gpu': True,
              'implementation_sha256': sha(dicl_module.__file__),
              'cpu_baseline_report_sha256':(sha(cpu_baseline/'aggregate.json') if baseline else None),
              'cpu_times_reused':baseline is not None, 'modalities': {}}
    cpu_dicts = {}; gpu_dicts = {}
    original_solver = dicl_module._SparseCodesBPDN
    instances = []
    class ObservedSolver(original_solver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0
            instances.append(self)

        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)

    dicl_module._SparseCodesBPDN = ObservedSolver

    for name in MODALITIES:
        source = projected/f'{name}_projected.h5'
        with h5py.File(source, 'r') as file:
            shape = file['data'].shape
            if shape[1] != 100 or file['data'].dtype != np.float32:
                raise ValueError('Expected R100 float32 projection')
        if baseline is None:
            with h5py.File(source, 'r') as file:
                data = file['data'][:].astype(np.float64)
            started = time.perf_counter()
            cpu = fit_dicl({name: data}, 200, 1000, 0)[name]
            cpu_seconds = time.perf_counter() - started
            del data
            cpu_file = output / f'{name}_cpu_dictionary.npy'
            np.save(cpu_file, cpu)
        else:
            if sha(source) != baseline['modalities'][name]['projected_sha256']:
                raise ValueError('CPU cache projected input mismatch')
            cpu_file = cpu_baseline / f'{name}_cpu_dictionary.npy'
            cpu = np.load(cpu_file)
            cpu_seconds = baseline['modalities'][name].get(
                'cpu_fit_seconds', baseline['modalities'][name].get('cpu_fit_seconds_reused'))
        cpu_dicts[name] = cpu
        instances.clear()
        torch.cuda.reset_peak_memory_stats()
        before = gpu_status()
        start = time.perf_counter()
        gpu = dicl_module.fit_dicl_gpu_streaming(projected, [name], 200, device='cuda:0')[name]
        torch.cuda.synchronize()
        gpu_seconds = time.perf_counter()-start
        gpu_dicts[name] = gpu
        np.save(output/f'{name}_gpu_dictionary.npy', gpu)
        report['modalities'][name] = {'shape': list(shape), 'projected_sha256': sha(source),
                                    ('cpu_fit_seconds_reused' if baseline else 'cpu_fit_seconds'):cpu_seconds, 'gpu_fit_seconds':gpu_seconds,
                                    'cpu_dictionary_sha256':sha(cpu_file),
                                    'gpu_status_before':before,'gpu_status_after':gpu_status(),
                                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                                    'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                                    'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                                    'solver_calls': {'calls':sum(item.calls for item in instances),
                                                     'fallbacks':sum(item.fallback_count for item in instances)}, 'cpu_gpu':compare(cpu,gpu)}
        (output/'aggregate.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'completed':name,**report['modalities'][name]}),flush=True)
        if report['modalities'][name]['cpu_gpu']['relative_frobenius_error'] > 1e-3:
            raise RuntimeError('Dictionary precision gate failed')
    h_cpu,_ = _fit_flica(cpu_dicts,3,100,output/'cpu_flica','cpu','R')
    h_gpu,_ = _fit_flica(gpu_dicts,3,100,output/'gpu_flica','cuda:0','R')
    u = np.load(projected/'U.npy')
    report['course_same_order_correlations'] = correlation(u@h_cpu,u@h_gpu)
    report['zstat_same_order_correlations'] = {}
    for name in MODALITIES:
        with h5py.File(projected/f'{name}_projected.h5','r') as file:
            data=file['data'][:].astype(np.float64)
        report['zstat_same_order_correlations'][name] = correlation(_spatial_z(h_cpu,data),_spatial_z(h_gpu,data))
    report['c20_diagnostics'] = {}
    for backend, dictionaries in (('cpu', cpu_dicts), ('cuda:0', gpu_dicts)):
        directory = output / ('cpu_c20' if backend == 'cpu' else 'gpu_c20')
        try:
            _fit_flica(dictionaries, 20, 100, directory, backend, 'R')
            error = None
        except ValueError as exception:
            error = str(exception)
        diagnostic = json.loads((directory/'flica_reconstruction.json').read_text())
        report['c20_diagnostics'][backend] = {'error': error, **diagnostic}
    report['status']='complete'
    (output/'aggregate.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':'complete','course':report['course_same_order_correlations'],
                      'zstat':report['zstat_same_order_correlations']}),flush=True)


if __name__=='__main__':main()
