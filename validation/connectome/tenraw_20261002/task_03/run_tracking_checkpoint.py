"""本轮新下载pilot自产检查点的tracking验收；与raw整例benchmark单列。"""
import argparse
import cProfile
import pstats
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

import numpy as np
import torch


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def load_tracking(source):
    # Each worker imports exactly two frozen source files, without installed FNIT.
    package = types.ModuleType('tracking_ab')
    package.__path__ = [str(source)]
    sys.modules['tracking_ab'] = package
    for name in ['fod', 'tracking']:
        spec = importlib.util.spec_from_file_location('tracking_ab.' + name, source / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def monitor(stop, records):
    while not stop.is_set():
        started = time.monotonic()
        try:
            response = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                       '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            rows = [row.split(',') for row in response.stdout.strip().splitlines()]
            own = [row for row in rows if row[0].strip() == str(os.getpid())]
            used = sum(int(row[2].strip()) * 1024**2 for row in own)
            records.append({'time': started, 'bytes': used, 'uuid': [row[1].strip() for row in own]})
        except Exception as error:
            records.append({'time': started, 'error': str(error)})
        stop.wait(.25)


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('benchmark output must be fresh/empty; retain previous failure evidence')
    begin = time.perf_counter()
    module = load_tracking(args.source)
    stop = threading.Event()
    samples = []
    sampler = threading.Thread(target=monitor, args=(stop, samples), daemon=True)
    sampler.start()
    try:
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        cpu_read = time.perf_counter() - start
        names = ['wm_sh', 'fod_affine', 'five_tissue', 'five_tissue_affine', 'gmwmi']
        if 'tracking_kwargs' not in checkpoint:
            raise ValueError('requires root direct-hook checkpoint with complete tracking_kwargs')
        tracking_kwargs = dict(checkpoint['tracking_kwargs'])
        if tracking_kwargs['compile_arc'] is not False:
            raise ValueError('default lossless benchmark requires snapshot compile_arc=False')
        for name in ['seed','batch_size']:
            requested = getattr(args,name)
            if requested is not None and requested != tracking_kwargs[name]:
                raise ValueError(name+' must match exact root snapshot; RNG consumption cannot change')
        size_override = None
        if args.n_seeds is not None and args.n_seeds != tracking_kwargs['n_seeds']:
            if args.n_seeds != 1_000_000:
                raise ValueError('only explicitly requested 1M scale overrides seed count')
            size_override = {'original':tracking_kwargs['n_seeds'],'requested':args.n_seeds}
            tracking_kwargs['n_seeds'] = args.n_seeds
        for name in ['fod_affine','five_tissue_affine']:
            if checkpoint[name].dtype != torch.float64:
                raise ValueError(name+' must preserve float64 source affine')
        if tracking_kwargs['five_tissue_spacing_mm'] != checkpoint['five_tissue_spacing_mm']:
            raise ValueError('snapshot header spacing mismatch')
        inputs = {}
        start = time.perf_counter()
        for name in names:
            inputs[name] = checkpoint[name].to('cuda')
        torch.cuda.synchronize()
        h2d = time.perf_counter() - start
        inputs['fa'] = None if checkpoint['fa'] is None else checkpoint['fa'].to('cuda')
        start = time.perf_counter()
        profiler = cProfile.Profile() if args.profile else None
        device_profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]) if args.cuda_profile else None)
        if device_profiler is not None:
            device_profiler.__enter__()
        if profiler is not None:
            profiler.enable()
        tracks = module.probabilistic_tractography(**inputs, **tracking_kwargs)
        torch.cuda.synchronize()
        tracking_wall = time.perf_counter() - start
        if device_profiler is not None:
            device_profiler.__exit__(None,None,None)
            device_profiler.export_chrome_trace(str(output/'tracking_cuda_trace.json'))
            kernel_us = copy_us = 0.
            for event in device_profiler.events():
                if event.device_type != torch.autograd.DeviceType.CUDA:
                    continue
                elapsed = event.time_range.elapsed_us()
                if 'memcpy' in event.name.lower() or 'memset' in event.name.lower():
                    copy_us += elapsed
                else:
                    kernel_us += elapsed
            (output/'tracking_cuda_profile.json').write_text(json.dumps({
                'scope':'instrumented CUDA trace; not formal performance wall',
                'sum_kernel_seconds':kernel_us/1e6,'sum_copy_memset_seconds':copy_us/1e6,
                'interpretation':'CUDA event durations; CPU dispatch/read not attributed as kernel percent'},indent=2)+'\n')
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(output / 'tracking_cpu_dispatch.prof')
            with open(output / 'tracking_cpu_dispatch.txt', 'w') as stream:
                pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(80)
        if not tracks.paths:
            raise RuntimeError('new real pilot tracking accepted no streamlines; not a completed benchmark')
        # Includes all tensors still resident, including the CPU-loaded checkpoint.
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        start = time.perf_counter()
        counts = np.array([len(path) for path in tracks.paths], dtype=np.int64)
        offsets = np.concatenate((np.array([0], dtype=np.int64), counts.cumsum()))
        # Transfer each public view without packing another full GPU tensor.
        points = np.empty((int(offsets[-1]), 3), dtype=np.float32)
        path_storages = {}
        path_storage_larger_than_path_count = 0
        for index, path in enumerate(tracks.paths):
            storage = path.untyped_storage()
            path_storages[storage.data_ptr()] = storage.nbytes()
            path_storage_larger_than_path_count += int(storage.nbytes() > path.numel()*path.element_size())
            points[offsets[index]:offsets[index+1]] = path.cpu().numpy()
        arrays = {'points': points, 'offsets': offsets,
                  'endpoints': tracks.endpoints.cpu().numpy(),
                  'lengths_mm': tracks.lengths_mm.cpu().numpy(),
                  'accepted_seeds': tracks.accepted_seeds.cpu().numpy()}
        if tracks.mean_fa is not None:
            arrays['mean_fa'] = tracks.mean_fa.cpu().numpy()
        torch.cuda.synchronize()
        d2h = time.perf_counter() - start
        start = time.perf_counter()
        for name, data in arrays.items():
            np.save(output / (name + '.npy'), data, allow_pickle=False)
        write_wall = time.perf_counter() - start
        report = {'harness_sha256': sha256(__file__), 'checkpoint_sha256': sha256(args.checkpoint), 'manifest_sha256': sha256(args.manifest),
            'source_commit': args.source_commit, 'cpu_dispatch_profile': args.profile,
            'cuda_profile': args.cuda_profile,
            'source_sha256': {name: sha256(args.source / (name + '.py')) for name in ['fod', 'tracking']},
            'seeds_attempted': tracks.seeds_attempted, 'accepted': len(tracks.paths), 'points': int(offsets[-1]),
            'path_live_storage_bytes': sum(path_storages.values()),
            'path_logical_bytes': int(points.nbytes),
            'path_storage_larger_than_path_count':path_storage_larger_than_path_count,
            'n_seeds': tracking_kwargs['n_seeds'], 'seed': tracking_kwargs['seed'],
            'batch_size': tracking_kwargs['batch_size'], 'tracking_kwargs':tracking_kwargs,
            'size_override':size_override, 'has_fa':inputs['fa'] is not None, 'compile_arc': False,
            'cpu_read_seconds': cpu_read, 'h2d_seconds': h2d, 'tracking_wall_seconds': tracking_wall,
            'd2h_seconds': d2h, 'write_seconds': write_wall, 'total_worker_seconds': time.perf_counter()-begin,
            'allocated_peak_bytes': peak_allocated, 'reserved_peak_bytes': peak_reserved,
            'torch_version': torch.__version__, 'cpu_threads': torch.get_num_threads(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'tf32': torch.backends.cuda.matmul.allow_tf32,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'error_type':type(error).__name__,
            'error':str(error),'checkpoint':str(args.checkpoint),'source_commit':args.source_commit},indent=2)+'\n')
        raise
    finally:
        stop.set()
        sampler.join(timeout=5)
        (output / 'nvml_samples.json').write_text(json.dumps(samples, indent=2)+'\n')
    valid = [sample for sample in samples if 'bytes' in sample]
    report['nvml_peak_bytes'] = max((sample['bytes'] for sample in valid if sample['uuid']), default=None)
    report['nvml_gpu_uuids'] = sorted({uuid for sample in valid for uuid in sample['uuid']})
    report['nvml_failed_samples'] = len(samples)-len(valid)
    report['nvml_max_gap_seconds'] = max((b['time']-a['time'] for a,b in zip(samples,samples[1:])), default=None)
    report['budget_pass'] = all(value is not None and value < 20_000_000_000 for value in
                               [peak_allocated, peak_reserved, report['nvml_peak_bytes']])
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not report['budget_pass']:
        raise SystemExit('20e9 memory budget failed or NVML unavailable; report retained')


def compare(args):
    result = {}
    left_report = json.loads((args.baseline/'report.json').read_text())
    right_report = json.loads((args.candidate/'report.json').read_text())
    for name in ['checkpoint_sha256','manifest_sha256','tracking_kwargs','has_fa']:
        if left_report[name] != right_report[name]:
            raise ValueError('paired runs differ in '+name)
    names = ['offsets', 'points', 'endpoints', 'lengths_mm', 'accepted_seeds']
    if left_report['has_fa']:
        names.append('mean_fa')
    for name in names:
        a = np.load(args.baseline / (name+'.npy'), mmap_mode='r')
        b = np.load(args.candidate / (name+'.npy'), mmap_mode='r')
        same_shape = a.shape == b.shape and a.dtype == b.dtype
        neq = int(np.count_nonzero(a != b)) if same_shape else None
        result[name] = {'baseline_shape': list(a.shape), 'candidate_shape': list(b.shape), 'neq': neq}
        if same_shape and a.size:
            delta = np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)
            result[name].update(max_abs=float(np.max(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta**2))),
                                p99=float(np.percentile(np.abs(delta),99)))
    result['strict_pass'] = all(value['neq'] == 0 for value in result.values())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['strict_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    worker = sub.add_parser('run')
    for name in ['source', 'checkpoint', 'manifest', 'output']:
        worker.add_argument('--'+name, type=Path, required=True)
    worker.add_argument('--source-commit', required=True)
    worker.add_argument('--profile', action='store_true')
    worker.add_argument('--cuda-profile', action='store_true')
    worker.add_argument('--n-seeds', type=int)
    worker.add_argument('--seed', type=int)
    worker.add_argument('--batch-size', type=int)
    comparison = sub.add_parser('compare')
    for name in ['baseline', 'candidate', 'output']:
        comparison.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    run(args) if args.mode == 'run' else compare(args)
