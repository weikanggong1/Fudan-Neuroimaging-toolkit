"""同输入完整GCA缓存隔离诊断，含冷子进程和已初始化CUDA父API。

reference-report只用于诊断比较，不传入注册函数；来源/硬件窗口另列。
进程GPU归属无法从容器确认时记null，整卡总量包括其他任务。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch

from fnit.recon_all import gca_torch_worker, profiling
from fnit.recon_all.profiling import ProcessTreeDeviceSampler, configure_cuda_allocator


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def read_lta_matrix(path):
    """比较实际序列化LTA；不能把未写出API矩阵与已舍入文件混比较。"""
    lines = path.read_text().splitlines()
    begin = next(i for i, line in enumerate(lines) if line.split() == ['1', '4', '4'])
    return np.array([[float(value) for value in line.split()]
                     for line in lines[begin + 1:begin + 5]], dtype=np.float64)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('nu', 'mask', 'atlas', 'reference-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--physical-gpu-uuid', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--candidate-chunk', type=int, default=1024)
    parser.add_argument('--parent-cuda', choices=('uninitialized', 'initialized'), required=True)
    parser.add_argument('--code-version', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report_path = args.output / 'report.json'
    frozen = json.loads(args.reference_report.read_text())
    inputs_sha = {name: sha256(getattr(args, name)) for name in ('nu', 'mask', 'atlas')}
    if inputs_sha != frozen['inputs_sha256']:
        raise ValueError('diagnostic reference must have identical nu/mask/atlas hashes')
    old = next(row for row in frozen['trials'] if row['kind'] == 'old_torch')
    native = next(row for row in frozen['trials'] if row['kind'] == 'conda')
    report = {
        'status': 'running', 'scope': 'complete frozen same-input GCA; not raw-T1 end-to-end',
        'code_version': args.code_version, 'parent_cuda_mode': args.parent_cuda,
        'inputs_sha256': inputs_sha,
        'source_sha256': {'worker': sha256(gca_torch_worker.__file__),
                          'profiling': sha256(profiling.__file__), 'benchmark': sha256(__file__)},
        'reference_report_sha256': sha256(args.reference_report),
        'reference_code_version': frozen['code_version'],
        'reference_device': frozen['device'], 'reference_hardware_and_window': 'read frozen reference; separate execution window',
        'acceptance_before_test': {'optimization_regression': 'LTA entries exact to existing Torch, same inputs and geometry',
                                  'strict_reproduction': 'LTA matrix exact to independently built native'},
        'overall_metric_equivalence': 'not_assessed',
        'timing_scope': 'isolated API includes child imports, validation, input hashes, CUDA init, transfers, registration, writes and child exit; benchmark imports excluded',
    }
    save(report_path, report)
    parent_allocator = configure_cuda_allocator(device=args.device, policy='disabled')
    torch.set_num_threads(args.threads)
    parent_tensor = None
    if args.parent_cuda == 'initialized':
        parent_tensor = torch.zeros(1 << 20, device=args.device, dtype=torch.float32)
        torch.cuda.synchronize(args.device)
    parent_precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    parent_environment = os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')
    memory = ProcessTreeDeviceSampler(device=args.device, parent_pid=os.getpid(), interval=.25)
    memory.uuid = args.physical_gpu_uuid
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            memory.sample_if_due()
            stop.wait(.025)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        result = gca_torch_worker.run_isolated_registration(
            nu_path=args.nu, mask_path=args.mask, atlas_path=args.atlas,
            output_path=args.output / 'talairach.lta', report_path=args.output / 'worker.json',
            device=args.device, threads=args.threads, candidate_chunk=args.candidate_chunk,
            inverse_backend='torch', code_version=args.code_version)
        if args.parent_cuda == 'initialized':
            torch.cuda.synchronize(args.device)
        matrix = read_lta_matrix(args.output / 'talairach.lta')
        raw_api_matrix = np.array(result['matrix'])
        old_matrix, native_matrix = np.array(old['matrix']), np.array(native['matrix'])
        report.update(
            status='complete', worker_result=result,
            comparisons={'old_torch_matrix_exact': bool(np.array_equal(matrix, old_matrix)),
                         'old_torch_matrix_max_abs': float(np.max(np.abs(matrix-old_matrix))),
                         'serialized_lta_matrix': matrix.tolist(),
                         'raw_api_matrix_max_abs_to_serialized_lta': float(np.max(np.abs(raw_api_matrix-matrix))),
                         'native_matrix_exact': bool(np.array_equal(matrix, native_matrix)),
                         'native_matrix_max_abs': float(np.max(np.abs(matrix-native_matrix))),
                         'space': 'type=0 voxel-to-voxel LTA; matrix entries mix scaling and voxel translation, not surface distances'},
            optimization_regression='matrix_exact' if np.array_equal(matrix, old_matrix) else 'not_passed',
            parent_allocator=parent_allocator,
            parent_cache_environment_preserved=os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == parent_environment,
            parent_precision_preserved=parent_precision == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32),
            parent_cuda_initialized_after=torch.cuda.is_initialized(),
            parent_live_tensor_unchanged=None if parent_tensor is None else bool((parent_tensor == 0).all()),
            parent_memory_stats_status='unavailable_cache_disabled',
        )
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        stop.set()
        thread.join()
        memory.sample_if_due(force=True)
        report.update(process_memory=memory.report(), main_wall_seconds_excluding_imports=time.perf_counter()-started)
        save(report_path, report)


if __name__ == '__main__':
    main()
