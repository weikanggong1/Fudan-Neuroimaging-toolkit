"""真实CUDA父live tensor/新worker缓存合同；无影像，不能替代科学benchmark。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def callback(subject, hemi, device, threads, operation):
    """私有合同目录→GPU标量及surf marker；保留传入device/threads，无MRI输出。"""
    import torch
    values = torch.arange(1024, dtype=torch.float32, device=device)
    result = {'device': str(values.device), 'dtype': str(values.dtype),
              'threads': torch.get_num_threads(), 'sum': float(values.sum()),
              'cache_disabled_environment': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING'),
              'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
              'cudnn_tf32': torch.backends.cudnn.allow_tf32,
              'autocast': torch.is_autocast_enabled('cuda')}
    marker = Path(subject) / 'surf' / f'{hemi}.allocator-contract.json'
    marker.write_text(json.dumps(result) + '\n')
    return result


def main():
    tick = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scope': 'synthetic lifecycle contract with real CUDA;not-real-data-benchmark',
              'device': args.device, 'threads': args.threads, 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    try:
        import numba
        import torch
        from fnit.recon_all.profiling import configure_cuda_allocator
        from fnit.recon_all.hemisphere_parallel import run_hemisphere_group
        import fnit.recon_all.hemisphere_parallel as source
        selected = torch.device(args.device)
        if selected.type != 'cuda' or selected.index is None or torch.cuda.is_initialized():
            raise ValueError('fresh explicit cuda:N contract process required')
        parent_allocator = configure_cuda_allocator(args.device, 'disabled')
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        numba.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        live = torch.arange(4096, dtype=torch.float32, device=selected)
        torch.cuda.synchronize(selected)
        before = hashlib.sha256(live.cpu().numpy().tobytes()).hexdigest()
        parent_environment = os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')
        # Callback directory must be importable in each fresh worker.
        os.environ['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get('PYTHONPATH', '')
        subject = args.output / 'contract-subject'
        for folder in ('mri', 'surf', 'label', 'stats', 'scripts'):
            (subject / folder).mkdir(parents=True)
        group = run_hemisphere_group(subject=subject, operation='allocator_contract',
                                     device=args.device, threads=args.threads, workers=2,
                                     profile_stages=True,
                                     callable_path='check_hemisphere_cache_live_cuda:callback',
                                     kwargs=None, startup_wait_seconds=30,
                                     cuda_allocator_cache='enabled')
        torch.cuda.synchronize(selected)
        after = hashlib.sha256(live.cpu().numpy().tobytes()).hexdigest()
        checks = {'parent_live_sha_equal': before == after,
                  'parent_environment_preserved': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == parent_environment == '1',
                  'both_workers_enabled': all(group['workers'][hemi]['cuda_allocator']['effective'] == 'enabled' for hemi in ('lh', 'rh')),
                  'worker_env_removed': all(group['values'][hemi]['cache_disabled_environment'] is None for hemi in ('lh', 'rh')),
                  'thread_budget': all(group['values'][hemi]['threads'] == args.threads // 2 for hemi in ('lh', 'rh')),
                  'precision_preserved': all(group['values'][hemi]['matmul_tf32'] and group['values'][hemi]['cudnn_tf32']
                                             and not group['values'][hemi]['autocast'] for hemi in ('lh', 'rh')),
                  'no_standard_mri_output': not list((subject / 'mri').iterdir())}
        if not all(checks.values()):
            raise RuntimeError(f'contract failed: {checks}')
        report.update(status='complete_contract', checks=checks, parent_live_sha256=before,
                      parent_allocator=parent_allocator, parent_cuda_initialized_before_workers=True,
                      group=group, module_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      torch_version=torch.__version__, cuda_runtime=torch.version.cuda)
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    report['contract_main_seconds'] = time.perf_counter() - tick
    (args.output / 'contract.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'complete_contract' else 1


if __name__ == '__main__':
    raise SystemExit(main())
