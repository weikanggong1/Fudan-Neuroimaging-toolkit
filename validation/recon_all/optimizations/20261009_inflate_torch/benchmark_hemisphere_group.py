"""公开自产smoothwm完整双侧inflation ABBA；父CUDA已初始化且缓存关闭。

复用完整run_standard_inflate与现有半球exec调度，不改算法；新worker
局部启用缓存。原生与Torch均按同一总线程预算、私有复制和发布语义执行。
输入只允许带SHA的公开自产表面包，--output必须不存在；不读取官方结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def callback(*, subject, hemi, device, threads, operation, backend, native):
    """私有subject→同序inflated/sulc；坐标surface RAS/mm，线程每侧固定。"""
    import torch
    from fnit.recon_all.inflate_standard_run import run_standard_inflate
    surface = Path(subject) / 'surf'
    started = time.perf_counter()
    if backend == 'native':
        command = [native, '-threads', str(threads), str(surface / (hemi + '.smoothwm')),
                   str(surface / (hemi + '.inflated'))]
        with (surface / (hemi + '.native-inflate.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = {'backend': backend, 'command': command}
    else:
        result = run_standard_inflate(input_surface=surface / (hemi + '.smoothwm'),
            inflated_output=surface / (hemi + '.inflated'), sulc_output=surface / (hemi + '.sulc'),
            backend='torch', device=device, profile=False)
    result.update(callback_wall_seconds=time.perf_counter() - started,
        cache_disabled_environment=os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING'),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32, autocast=torch.is_autocast_enabled('cuda'),
        actual_torch_threads=torch.get_num_threads(), operation=operation)
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.threads < 2:
        raise ValueError('two workers require total threads >= 2')
    args.output.mkdir(parents=True)
    report = {'status': 'running', 'scope': 'complete same-input bilateral inflation group ABBA; not raw-T1 whole recon-all',
              'device': args.device, 'threads_total': args.threads, 'workers': 2,
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'rows': [],
              'timing_scope': 'complete group includes fresh worker interpreters, imports/CUDA init/JIT, private copies, API, transfers, I/O and publication',
              'cuda_allocator_selection': 'parent disabled with live tensor; fresh children enabled only',
              'overall_metric_equivalence': 'not assessed', 'production_default_changed': False}
    def save():
        temporary = args.output / 'summary.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output / 'summary.json')
    try:
        # main owns this fresh process; worker imports do not execute this policy.
        os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        import torch
        import numba
        from fnit.recon_all.profiling import configure_cuda_allocator
        from fnit.recon_all.hemisphere_parallel import run_hemisphere_group
        from fnit.recon_all import hemisphere_parallel, inflate_standard_run, inflate_torch
        from benchmark_complete import compare
        device = torch.device(args.device)
        if device.type != 'cuda' or device.index is None or torch.cuda.is_initialized():
            raise ValueError('fresh explicit cuda:N parent required')
        allocator = configure_cuda_allocator(args.device, 'disabled')
        torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
        numba.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        live = torch.arange(4096, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        live_sha = hashlib.sha256(live.cpu().numpy().tobytes()).hexdigest()
        parent_env = os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')
        os.environ['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get('PYTHONPATH', '')
        manifest = json.loads((args.data / 'manifest.json').read_text())
        report.update(data_manifest_sha256=sha(args.data / 'manifest.json'), native_sha256=sha(args.native),
            torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device), parent_allocator=allocator,
            parent_cuda_initialized_before_group=True, parent_live_sha256=live_sha,
            source_sha256={str(path): sha(path) for path in map(Path, (
                __file__, hemisphere_parallel.__file__, inflate_standard_run.__file__, inflate_torch.__file__))})
        grouped = {}
        for entry in manifest['cases']:
            if sha(args.data / entry['surface']) != entry['sha256']:
                raise ValueError('input SHA mismatch')
            grouped.setdefault(entry['case'], {})[entry['hemisphere']] = entry
        if not grouped or any(set(value) != {'lh', 'rh'} for value in grouped.values()):
            raise ValueError('each case requires exactly two hemispheres')
        save()
        for case, entries in grouped.items():
            row = {'case': case, 'input_sha256': {hemi: entry['sha256'] for hemi, entry in entries.items()}, 'runs': []}
            report['rows'].append(row)
            directories = []
            for index, backend in enumerate(('native', 'torch', 'torch', 'native')):
                directory = args.output / case / f'{index + 1:02d}_{backend}'
                subject = directory / 'subject'
                for folder in ('mri', 'surf', 'label', 'stats', 'scripts'):
                    (subject / folder).mkdir(parents=True)
                for hemi, entry in entries.items():
                    shutil.copyfile(args.data / entry['surface'], subject / 'surf' / (hemi + '.smoothwm'))
                tick = time.perf_counter()
                group = run_hemisphere_group(subject=subject, operation='complete_standard_inflate',
                    device=args.device, threads=args.threads, workers=2, profile_stages=True,
                    callable_path='benchmark_hemisphere_group:callback',
                    kwargs={'backend': backend, 'native': str(args.native)},
                    startup_wait_seconds=30, cuda_allocator_cache='enabled')
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - tick
                after_sha = hashlib.sha256(live.cpu().numpy().tobytes()).hexdigest()
                checks = {'parent_live_unchanged': after_sha == live_sha,
                    'parent_disabled_environment_preserved': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == parent_env == '1',
                    'both_workers_enabled': all(group['workers'][hemi]['cuda_allocator']['effective'] == 'enabled' for hemi in ('lh', 'rh')),
                    'actual_threads': all(group['values'][hemi]['actual_torch_threads'] == args.threads // 2 for hemi in ('lh', 'rh')),
                    'precision_preserved': all(group['values'][hemi]['matmul_tf32'] and group['values'][hemi]['cudnn_tf32'] and not group['values'][hemi]['autocast'] for hemi in ('lh', 'rh'))}
                item = {'backend': backend, 'group_wall_seconds': elapsed, 'group': group, 'checks': checks}
                row['runs'].append(item); directories.append(subject / 'surf'); save()
                if not all(checks.values()):
                    raise RuntimeError('parent/worker lifecycle contract failed')
                print('DONE', case, backend, index + 1, elapsed, flush=True)
            row['comparisons_to_native_first'] = [{
                'run': index + 1, 'hemisphere': hemi,
                'comparison': compare(directory, directories[0], hemi)}
                for index, directory in enumerate(directories[1:], 1) for hemi in ('lh', 'rh')]
            row['group_median_seconds'] = {backend: statistics.median(item['group_wall_seconds'] for item in row['runs'] if item['backend'] == backend) for backend in ('native', 'torch')}
            row['group_reduction_percent'] = (1 - row['group_median_seconds']['torch'] / row['group_median_seconds']['native']) * 100
            save()
        report['status'] = 'complete_bilateral_stage_pair'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    report['main_seconds'] = time.perf_counter() - started
    save()
    return 0 if report['status'] == 'complete_bilateral_stage_pair' else 1


if __name__ == '__main__':
    raise SystemExit(main())
