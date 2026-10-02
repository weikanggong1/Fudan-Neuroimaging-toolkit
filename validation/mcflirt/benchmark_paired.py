"""同卡交替测量两份源码的完整真实 BOLD MCFLIRT 调用。

输入为同一四维 BOLD、冻结三维参考及两份含 src/fnit 的源码目录。
默认两个回合的顺序为 baseline/candidate、candidate/baseline。
运行时不调用原软件；公开报告只保存匿名计时、源码与数组哈希。
矩阵、参数及 corrected 数组的 SHA-256 按未舍入内存值计算。
各次调用使用独立进程，包含输入解压、准备、首次编译/graph 捕获、
估计、样条采样及 dtype 转换，不含哈希、数值核对或文件写盘。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot():
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.free,clocks.sm',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        return result.stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []


def worker(args):
    import nibabel as nib
    import numpy as np
    import torch
    from unittest.mock import patch
    from fnit.mcflirt import TorchMCFLIRT
    import fnit.fmri.spatial as spatial

    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    root = Path(importlib.import_module('fnit').__file__).resolve().parent
    paths = list((root / 'mcflirt').glob('*.py')) + [root / 'fmri/spatial.py', root / 'flirt/core.py']
    def source_hashes():
        return {str(path.relative_to(root)): sha256_file(path) for path in sorted(paths)}
    before = source_hashes()
    raw = nib.load(args.bold)
    reference = nib.load(args.reference)
    if raw.ndim != 4 or reference.ndim != 3:
        raise ValueError('BOLD must be 4D and reference must be 3D')
    sampler = spatial.apply_motion_warp
    sampling_times = []
    def wrapped_sampler(*values, **options):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = sampler(*values, **options)
        torch.cuda.synchronize(device)
        sampling_times.append(time.perf_counter() - started)
        return result
    load_before = gpu_snapshot()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with patch.object(spatial, 'apply_motion_warp', wrapped_sampler):
        fit = TorchMCFLIRT(device=device).run(raw, reference, resample=True, interpolation='spline')
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = {'allocated_bytes': torch.cuda.max_memory_allocated(device),
            'reserved_bytes': torch.cuda.max_memory_reserved(device)}
    if len(sampling_times) != 1:
        raise RuntimeError('expected one final sampling call')
    def array_sha256(values):
        array = np.ascontiguousarray(values)
        return hashlib.sha256(array.view(np.uint8)).hexdigest()
    result = {'motion_run_seconds': seconds, 'motion_sampling_seconds': sampling_times[0],
              'motion_minus_sampling_seconds': seconds - sampling_times[0],
              'cost_evaluations': fit.cost_evaluations, 'frames': raw.shape[3],
              'arrays_sha256': {name: array_sha256(array) for name, array in (
                  ('matrices_float64', fit.matrices), ('parameters_float64', fit.parameters),
                  ('corrected', np.asanyarray(fit.corrected.dataobj)))},
              'input_sha256': {'bold': sha256_file(args.bold), 'reference': sha256_file(args.reference)},
              'corrected_dtype': str(fit.corrected.get_data_dtype()), 'sources_sha256': before,
              'sources_unchanged': before == source_hashes(), 'peak_cuda': peak,
              'gpu_before': load_before, 'gpu_after': gpu_snapshot(),
              'software': {'torch': torch.__version__, 'numpy': np.__version__, 'nibabel': nib.__version__},
              'device': torch.cuda.get_device_name(device), 'threads': torch.get_num_threads(),
              'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'tf32': True,
              'half_precision': False}
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--bold', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--candidate-source', type=Path)
    parser.add_argument('--baseline-revision')
    parser.add_argument('--candidate-revision')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.rounds < 1 or args.threads < 1 or not all((args.baseline_source, args.candidate_source, args.output,
                                                     args.baseline_revision, args.candidate_revision)):
        parser.error('provide both source directories and revisions, fresh output, positive rounds and threads')
    if args.output.exists():
        parser.error('output must not already exist')
    trials = []
    for round_index in range(args.rounds):
        order = ('baseline', 'candidate') if round_index % 2 == 0 else ('candidate', 'baseline')
        for label in order:
            source = getattr(args, label + '_source').resolve()
            source = source / 'src' if (source / 'src/fnit').is_dir() else source
            if not (source / 'fnit/mcflirt/core.py').is_file():
                parser.error('source directories must contain fnit/mcflirt/core.py')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(source)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--bold', args.bold,
                       '--reference', args.reference, '--threads', str(args.threads), '--device', args.device]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            if result.returncode:
                # Paths may appear in a traceback; the operator's log is private.
                sys.stderr.write(result.stderr)
                raise RuntimeError('paired MCFLIRT worker failed')
            trial = json.loads(result.stdout.strip().splitlines()[-1])
            trial.update({'label': label, 'round': round_index + 1,
                          'revision': getattr(args, label + '_revision')})
            trials.append(trial)
            print(json.dumps({'completed_trial': len(trials), 'label': label,
                              'motion_run_seconds': trial['motion_run_seconds']}, allow_nan=False), flush=True)
    first = trials[0]
    report = {'subjects': 1, 'full_series': True, 'order': [trial['label'] for trial in trials],
              'all_arrays_bitwise_equal': all(trial['arrays_sha256'] == first['arrays_sha256'] for trial in trials),
              'same_cost_evaluations': all(trial['cost_evaluations'] == first['cost_evaluations'] for trial in trials),
              'same_inputs': all(trial['input_sha256'] == first['input_sha256'] for trial in trials),
              'all_runtime_sources_unchanged': all(trial['sources_unchanged'] for trial in trials),
              'trials': trials,
              'timing_scope': 'Independent processes; full TorchMCFLIRT.run includes array read/decompression, preparation, compilation/graph capture, optimization, spline sampling and dtype cast; excludes writes, hashes and numerical comparisons. Same GPU selected by CUDA_VISIBLE_DEVICES; shared load snapshots are observations.',
              'precision_scope': 'SHA-256 of contiguous unrounded float64 matrices/parameters and all corrected voxel values; bitwise array identity includes signed zero. This compares two FNIT revisions, not FSL.',
              'privacy': 'No image arrays, paths, parameters, matrices or per-frame metrics are included.'}
    report['valid_run'] = all(report[key] for key in ('all_arrays_bitwise_equal', 'same_cost_evaluations',
                                                     'same_inputs', 'all_runtime_sources_unchanged'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not report['valid_run']:
        raise RuntimeError('paired validation failed; inspect private worker log and public aggregate report')


if __name__ == '__main__':
    main()
