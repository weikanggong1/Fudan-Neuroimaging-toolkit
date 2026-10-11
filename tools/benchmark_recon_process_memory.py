#!/usr/bin/env python3
"""测量显存查询的完整开销；不运行MRI算法，不代表整例速度或显存验收。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

import torch

from fnit.recon_all.profiling import ProcessTreeDeviceSampler


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_benchmark(*, output_path: Path, device: str, repeats: int = 12,
                  interval: float = .1, held_bytes: int = 128 * 1024**2,
                  code_version: str):
    """ABBA查询NVML与CLI；完整原始快照、首次/后续查询秒数和失败保存JSON。

    output_path为新JSON；device显式CUDA逻辑设备；repeats每组至少2次；
    interval为计划间隔秒；held_bytes为目标卡新uint8缓冲大小；code_version
    为实际提交，另记录模块SHA。只同步新张量目标卡，不改调用方当前卡、
    精度、缓存或已有张量。失败仍写报告并抛异常，不判定未知PID显存为0。
    """
    if output_path.exists():
        raise FileExistsError(output_path)
    if repeats < 2 or interval <= 0 or held_bytes < 2:
        raise ValueError('repeats>=2, interval>0 and held_bytes>=2 are required')
    if torch.device(device).type != 'cuda':
        raise ValueError('this GPU telemetry benchmark requires explicit CUDA device')
    started = time.perf_counter()
    from fnit.recon_all import profiling, nvml_process_memory
    report = dict(
        scope='GPU telemetry only; no MRI stage, whole pipeline or 20 GB acceptance',
        code_version=code_version, source_sha256={
            'profiling.py': sha256(profiling.__file__),
            'nvml_process_memory.py': sha256(nvml_process_memory.__file__),
            'benchmark_recon_process_memory.py': sha256(__file__),
        }, torch_version=torch.__version__, nvml_python_version=importlib.metadata.version('nvidia-ml-py'),
        target_device=device, cpu_affinity=sorted(os.sched_getaffinity(0)),
        initialized_at_entry=torch.cuda.is_initialized(),
        allocator_environment=os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING'),
        matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
        repeats_per_group=repeats, requested_interval_seconds=interval, held_buffer_bytes=held_bytes,
        blocks=[], status='running',
    )
    held = None
    try:
        pre = ProcessTreeDeviceSampler(device=device, parent_pid=os.getpid(), query_backend='nvml')
        try:
            pre.sample_if_due(force=True)
            report['lazy_nvml_sample'] = pre.report()
            report['initialized_after_lazy_nvml'] = torch.cuda.is_initialized()
        finally:
            pre.close()
        if report['initialized_after_lazy_nvml'] != report['initialized_at_entry']:
            raise AssertionError('NVML query changed CUDA initialization state')
        current_before = torch.cuda.current_device() if torch.cuda.is_initialized() else None
        report['current_device_at_entry'] = current_before
        held = torch.full((held_bytes,), 17, dtype=torch.uint8, device=device)
        pointer = held.data_ptr()
        torch.cuda.synchronize(torch.device(device))
        report['current_device_after_tensor'] = torch.cuda.current_device()
        report['target_memory_after_tensor'] = dict(
            allocated_bytes=int(torch.cuda.memory_allocated(device)),
            reserved_bytes=int(torch.cuda.memory_reserved(device)),
        )
        for backend in ('nvidia-smi', 'nvml', 'nvml', 'nvidia-smi'):
            sampler = ProcessTreeDeviceSampler(device=device, parent_pid=os.getpid(),
                                               interval=interval, query_backend=backend)
            durations = []
            tick = time.perf_counter()
            try:
                for index in range(repeats):
                    if index:
                        time.sleep(interval)
                    query_tick = time.perf_counter()
                    sampler.sample_if_due(force=True)
                    durations.append(time.perf_counter() - query_tick)
            finally:
                sampler.close()
                report['blocks'].append(dict(
                    backend=backend, query_call_seconds=durations,
                    block_wall_seconds=time.perf_counter() - tick,
                    first_query_seconds=durations[0] if durations else None,
                    subsequent_query_median_seconds=statistics.median(durations[1:]) if len(durations)>1 else None,
                    process_memory=sampler.report(),
                ))
            if sampler.errors or len(sampler.samples) != repeats:
                raise RuntimeError(f'{backend} has failed queries; raw diagnostics preserved')
        uuids = {b['process_memory']['target_gpu_uuid'] for b in report['blocks']}
        if len(uuids) != 1 or None in uuids:
            raise AssertionError('backends did not map to one target UUID')
        report['target_gpu_uuid'] = uuids.pop()
        report['held_buffer_preserved'] = held.data_ptr() == pointer and int(held[0]) == 17 and int(held[-1]) == 17
        report['current_device_after_queries'] = torch.cuda.current_device()
        report['current_device_preserved'] = (current_before is None or
                                               torch.cuda.current_device() == current_before)
        if not report['held_buffer_preserved'] or not report['current_device_preserved']:
            raise AssertionError('query changed initialized API state')
        report['median_query_seconds'] = {
            name: statistics.median([v for b in report['blocks'] if b['backend']==name
                                     for v in b['query_call_seconds'][1:]])
            for name in ('nvidia-smi','nvml')
        }
        report['status'] = 'complete_telemetry_only'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['benchmark_function_wall_seconds'] = time.perf_counter() - started
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--interval', type=float, default=.1)
    parser.add_argument('--held-bytes', type=int, default=128*1024**2)
    parser.add_argument('--code-version', required=True)
    args = parser.parse_args()
    run_benchmark(output_path=args.output_path, device=args.device, repeats=args.repeats,
                  interval=args.interval, held_bytes=args.held_bytes, code_version=args.code_version)


if __name__ == '__main__':
    main()
