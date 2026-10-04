"""真实双半球标量启动验证；外层负责GPU锁与监测，不运行影像算法。"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def source_manifest(paths):
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in paths}


def function_gate(group):
    """Missing operation flags remain unknown; never infer safe retry from absence."""
    result = {}
    for hemi in ('lh', 'rh'):
        attempts = group.get('startup_attempts', {}).get(hemi)
        children = ([row.get('worker_report', {}) for row in attempts] if attempts is not None
                    else [group.get('workers', {}).get(hemi, {})])
        flags = [child.get('operation_entered') for child in children]
        entered = sum(flag is True for flag in flags)
        result[hemi] = {
            'operation_entered_flags': flags,
            'reported_function_entries': entered,
            'entry_count_known': bool(flags) and all(isinstance(flag, bool) for flag in flags),
            'callable_results': int(hemi in group.get('values', {})),
            'at_most_once_verified_from_flags': bool(flags)
                and all(isinstance(flag, bool) for flag in flags) and entered <= 1,
            'failure_all_entries_proven_false': bool(flags) and all(flag is False for flag in flags),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='新验证输出目录；subject必须不存在，允许外层监测目录已有其他文件')
    parser.add_argument('--initialized-parent', action='store_true',
                        help='父进程先做一次FP32标量分配和同步；失败直接结束，不重试')
    parser.add_argument('--startup-wait-seconds', type=float, default=None,
                        help='candidate显式等待预算，例如30；baseline省略，完全不传新参数')
    args = parser.parse_args(argv)
    if args.startup_wait_seconds is not None and (
            not math.isfinite(args.startup_wait_seconds) or args.startup_wait_seconds < 0):
        parser.error('--startup-wait-seconds must be finite and >= 0')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'startup_group.json'
    subject = output / 'subject'
    if subject.exists() or receipt.exists():
        raise FileExistsError('validation subject/report already exists; use a fresh output')
    started = time.monotonic()
    report = {'status': 'starting', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'pid': os.getpid(), 'mode': 'baseline_default' if args.startup_wait_seconds is None else 'candidate',
              'initialized_parent_requested': args.initialized_parent,
              'startup_wait_seconds_argument': args.startup_wait_seconds,
              'device_requested': 'cuda:0', 'workers': 2, 'threads': 4,
              'scope': 'scalar CUDA bootstrap only; no image, no algorithm retry; external lock/monitor',
              'precision_requested': {'matmul_tf32': True, 'cudnn_tf32': True, 'bootstrap_dtype': 'float32'},
              'script_sha256': source_manifest([__file__])}
    write_json(receipt, report)
    phase, torch = 'prepare', None
    try:
        for folder in ('mri', 'surf', 'label', 'stats', 'scripts'):
            (subject / folder).mkdir(parents=True, exist_ok=False)
        # Fresh process: policies precede Torch and native library imports.
        os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NUMBA_NUM_THREADS'):
            os.environ[key] = '4'
        target_directory = str(Path(__file__).resolve().parent)
        sys.path.insert(0, target_directory)
        os.environ['PYTHONPATH'] = target_directory + os.pathsep + os.environ.get('PYTHONPATH', '')
        report['environment'] = {key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'PYTORCH_NO_CUDA_MEMORY_CACHING', 'PYTORCH_CUDA_ALLOC_CONF',
            'CUDA_MODULE_LOADING', 'TORCH_SHOW_CPP_STACKTRACES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')}
        phase = 'imports'
        import torch
        import fnit.recon_all.hemisphere_parallel as parallel
        import fnit.recon_all.hemisphere_worker as worker
        import fnit.recon_all.profiling as profiling
        import cuda_bootstrap_target as target
        package = Path(parallel.__file__).resolve().parents[1]
        manifest = source_manifest(sorted(package.rglob('*.py')) + [target.__file__])
        write_json(output / 'source_manifest.json', manifest)
        report['source_manifest_sha256'] = hashlib.sha256((output / 'source_manifest.json').read_bytes()).hexdigest()
        report['critical_source_sha256'] = source_manifest([parallel.__file__, worker.__file__,
                                                            profiling.__file__, target.__file__])
        report['torch_version'], report['torch_cuda_version'] = torch.__version__, torch.version.cuda
        phase = 'policy'
        if torch.cuda.is_initialized():
            raise RuntimeError('fresh script unexpectedly entered with CUDA initialized')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        report['allocator'] = profiling.configure_cuda_allocator('cuda:0', 'disabled')
        report['precision_actual'] = {'matmul_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                                      'cudnn_tf32': bool(torch.backends.cudnn.allow_tf32)}
        report['parent_initialized_before_bootstrap'] = torch.cuda.is_initialized()
        if args.initialized_parent:
            phase = 'parent_bootstrap'
            parent_scalar = torch.empty(1, dtype=torch.float32, device='cuda:0')
            torch.cuda.synchronize(torch.device('cuda:0'))
            properties = torch.cuda.get_device_properties('cuda:0')
            report['parent_actual_cuda_device'] = {'name': properties.name,
                'uuid': str(getattr(properties, 'uuid', 'unavailable'))}
            report['parent_bootstrap_dtype'] = str(parent_scalar.dtype)
        report['parent_initialized_before_group'] = torch.cuda.is_initialized()
        phase = 'hemisphere_group'
        parameters = {'device': 'cuda:0', 'threads': 4, 'workers': 2,
                      'callable_path': 'cuda_bootstrap_target:noop'}
        if args.startup_wait_seconds is not None:
            parameters['startup_wait_seconds'] = args.startup_wait_seconds
        report['group_call_parameters'] = parameters
        write_json(receipt, report)
        group = parallel.run_hemisphere_group(subject, 'scalar_startup', **parameters)
        report['group'] = group
        report['function_entry_gate'] = function_gate(group)
        report['actual_worker_devices'] = {h: child.get('actual_cuda_device')
                                          for h, child in group.get('workers', {}).items()}
        if set(group.get('values', {})) != {'lh', 'rh'}:
            raise RuntimeError('complete group did not return exactly one result per hemisphere')
        if any(g['reported_function_entries'] > 1 for g in report['function_entry_gate'].values()):
            raise RuntimeError('multiple callable entries reported for a hemisphere')
        report['status'] = 'complete'
    except BaseException as error:
        if hasattr(error, 'report'):
            report['group'] = error.report
            report['function_entry_gate'] = function_gate(error.report)
            report['actual_worker_devices'] = {h: child.get('actual_cuda_device')
                                               for h, child in error.report.get('workers', {}).items()}
        report.update(status='failed', failure_phase=phase, error=repr(error), traceback=traceback.format_exc())
        # Pure Python state only after failure; no additional CUDA queries.
        report['parent_cuda_initialized_after_failure'] = torch.cuda.is_initialized() if torch is not None else None
    finally:
        report['total_wall_seconds'] = time.monotonic() - started
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(receipt, report)
    print(json.dumps({'status': report['status'], 'report': str(receipt),
                      'total_wall_seconds': report['total_wall_seconds']}), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
