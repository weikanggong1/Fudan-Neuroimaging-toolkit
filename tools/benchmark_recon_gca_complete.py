"""真实同输入完整GCA注册：旧Torch/批量求逆与独立Conda原生配对。

每次包括读入、准备、搜索、EM、LTA写出；GPU前后同步。冷进程导入不计入
单次API或main函数墙钟；完整CLI须由外部启动器计时。不生成原始T1整例。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

import numpy as np
import torch

from fnit.recon_all import mri_em_register_python as registration
from fnit.recon_all import mri_em_register_score_gpu as scorer
from fnit.recon_all import profiling
from fnit.recon_all.profiling import ProcessTreeDeviceSampler


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_matrix(path):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.split() == ['1', '4', '4'])
    return np.array([[float(value) for value in line.split()]
                     for line in lines[start + 1:start + 5]], dtype=np.float64)


def save(path, report):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(report, indent=2))
    pending.replace(path)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('nu', 'mask', 'atlas', 'native-binary', 'assets-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--candidate-chunk', type=int, default=1024)
    parser.add_argument('--code-version', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report_path = args.output / 'report.json'
    report = {
        'status': 'running', 'scope': 'frozen same-input complete registration; not raw-T1 end-to-end',
        'code_version': args.code_version, 'host': platform.node(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'threads': args.threads,
        'torch': torch.__version__, 'device': args.device,
        'precision': {'dtype': 'float32 with source FP64 exceptions', 'tf32': True, 'half': False},
        'inputs_sha256': {name: sha(getattr(args, name)) for name in ('nu', 'mask', 'atlas')},
        'source_sha256': {'registration': sha(registration.__file__), 'scorer': sha(scorer.__file__), 'profiling': sha(profiling.__file__), 'script': sha(__file__)},
        'native_sha256': sha(args.native_binary),
        'native_ldd': subprocess.run(['ldd', str(args.native_binary)], capture_output=True, text=True).stdout,
        'environment': {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'NUMBA_CACHE_DIR', 'PYTORCH_NO_CUDA_MEMORY_CACHING')},
        'timing_scope': {'api': 'synchronized read/preparation/search/EM/LTA write',
                         'main': 'argument parsing, hashes, CUDA initialization and all trials; Python imports excluded',
                         'cold_process_cli': 'not measured by this script; use external launcher'},
        'acceptance_before_test': {'strict_reproduction': 'matrix entries exact; reference repeats checked separately',
                                  'optimization_regression': 'final LTA matrix exact to old Torch; no change to search/EM/output semantics'},
        'overall_metric_equivalence': 'not_assessed', 'trials': [],
    }
    save(report_path, report)
    try:
        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(args.device)
        init = time.perf_counter()
        torch.cuda.synchronize(args.device)
        report['cuda_init_seconds'] = time.perf_counter() - init
        matrices = []
        for kind in ('conda', 'old_torch', 'batched_torch', 'batched_torch', 'old_torch', 'conda'):
            index = len(report['trials'])
            lta = args.output / f'{index}_{kind}.lta'
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            monitor_stop = threading.Event()
            memory = ProcessTreeDeviceSampler(device=args.device, parent_pid=os.getpid(), interval=.5)

            def sample_memory():
                while not monitor_stop.is_set():
                    memory.sample_if_due()
                    monitor_stop.wait(.05)

            monitor = threading.Thread(target=sample_memory, daemon=True)
            monitor.start()
            tick = time.perf_counter()
            result = None
            try:
                if kind == 'conda':
                    environment = dict(os.environ, FREESURFER_HOME=str(args.assets_dir), FNIT_GCA_SCORER='original')
                    environment.pop('FNIT_GCA_QUERY_CAPABILITIES', None)
                    with (args.output / f'{index}_conda.log').open('w') as log:
                        subprocess.run([str(args.native_binary), '-uns', '3', '-mask', str(args.mask),
                                        str(args.nu), str(args.atlas), str(lta)],
                                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                else:
                    result = registration.register_t1(
                        nu_path=args.nu, atlas_path=args.atlas, mask_path=args.mask, output_path=lta,
                        device=args.device, search_backend='torch',
                        candidate_chunk=64 if kind == 'old_torch' else args.candidate_chunk,
                        sample_chunk=8192, reduce_on_device=True,
                        inverse_backend='cpu' if kind == 'old_torch' else 'torch')
                torch.cuda.synchronize(args.device)
                seconds = time.perf_counter() - tick
            finally:
                monitor_stop.set()
                monitor.join()
            matrix = read_matrix(lta)
            matrices.append(matrix)
            cache_disabled = os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') is not None
            report['trials'].append({
                'kind': kind, 'api_wall_seconds_including_read_transfer_write': seconds,
                'matrix': matrix.tolist(), 'lta_sha256': sha(lta), 'api_result': result,
                'allocated_peak_bytes': None if cache_disabled else torch.cuda.max_memory_allocated(args.device),
                'reserved_peak_bytes': None if cache_disabled else torch.cuda.max_memory_reserved(args.device),
                'torch_memory_stats_status': 'unavailable_allocator_cache_disabled' if cache_disabled else 'available',
                'process_memory': memory.report(),
                'jit_cache_scope': 'same declared cache; first calls may compile and are retained',
            })
            save(report_path, report)
        old = matrices[1]
        report['comparisons'] = [{
            'kind': row['kind'], 'matrix_max_abs_to_old_torch': float(np.max(np.abs(matrix-old))),
            'matrix_exact_to_old_torch': bool(np.array_equal(matrix, old)),
            'matrix_max_abs_to_conda': float(np.max(np.abs(matrix-matrices[0]))),
        } for row, matrix in zip(report['trials'], matrices)]
        report['conda_repeat_exact'] = bool(np.array_equal(matrices[0], matrices[-1]))
        report['old_torch_repeat_exact'] = bool(np.array_equal(matrices[1], matrices[-2]))
        report['batched_repeat_exact'] = bool(np.array_equal(matrices[2], matrices[3]))
        report['optimization_regression'] = ('matrix_exact' if all(np.array_equal(matrix, old) for matrix in matrices[1:5]) else 'not_passed')
        report['strict_reproduction_vs_conda'] = ('matrix_exact' if all(np.array_equal(matrix, matrices[0]) for matrix in matrices) else 'not_passed')
        timings = {kind: float(np.median([row['api_wall_seconds_including_read_transfer_write']
                    for row in report['trials'] if row['kind']==kind])) for kind in ('conda','old_torch','batched_torch')}
        report['median_api_wall_seconds'] = timings
        report['speedup_to_old_torch'] = timings['old_torch']/timings['batched_torch']
        report.update(status='complete', main_wall_seconds_excluding_imports=time.perf_counter()-started)
        save(report_path, report)
    except BaseException as error:
        report.update(status='failed',error=repr(error),main_wall_seconds_excluding_imports=time.perf_counter()-started)
        save(report_path, report)
        raise


if __name__ == '__main__':
    main()
