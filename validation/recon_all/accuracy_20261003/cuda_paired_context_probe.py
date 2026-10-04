"""锁内配对 CUDA context 诊断；默认8对/16个新进程，不运行真实影像。

每对一个 context_driver 和一个 frozen original_worker/direct_priority。
顺序交替两种对照，固定seed随机每对的启动顺序，每个进程线程环境为2。
无首CUDA调用barrier，Popen并发不等于首runtime调用同时；不同瞬间的
成功/失败不能单独确证上下文干预修复。--output须为新的独立诊断目录。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

from cuda_bootstrap_probe import GPU_UUID, digest, gpu_snapshot, now, write
from resource_admission import snapshot_descendants, active_owned, cleanup_owned_tree

THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NUMBA_NUM_THREADS')
ENV_KEYS = THREAD_KEYS + ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_MODULE_LOADING',
                          'CUDA_LAUNCH_BLOCKING', 'PYTORCH_NO_CUDA_MEMORY_CACHING',
                          'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'LD_LIBRARY_PATH',
                          'LD_PRELOAD', 'PYTHONPATH', 'TORCH_SHOW_CPP_STACKTRACES',
                          'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT')


class Interrupted(Exception):
    pass


def proc_receipt(pid):
    """外部只读proc采样；不读取argv/fullenv，不向孩子插入CUDA调用。"""
    root = Path('/proc') / str(pid)
    report = {'utc': now(), 'monotonic': time.monotonic(), 'pid': pid}
    for name in ('maps', 'status', 'limits', 'cgroup'):
        try:
            lines = (root / name).read_text().splitlines()
            if name == 'maps':
                paths = []
                for line in lines:
                    fields = line.split(maxsplit=5)
                    if len(fields) == 6 and fields[5].startswith('/'):
                        path = fields[5]
                        if any(token in Path(path).name for token in
                               ('libcuda', 'libtorch_cuda', 'libc10_cuda', 'libcudnn', 'libcublas')):
                            paths.append(path)
                report['loaded_cuda_paths'] = sorted(set(paths))
            elif name == 'status':
                report['status_selected'] = [line for line in lines if
                                            line.startswith(('VmRSS:', 'VmLck:', 'VmPin:', 'Threads:'))]
            else:
                report[name] = lines
        except OSError as error:
            report[name] = {'status': 'unavailable', 'error': repr(error)}
    return report


def numa_host_receipt():
    """只读内核NUMA空闲/碎片指标；不据此直接归因CUDA返回码。"""
    report = {'utc': now(), 'monotonic': time.monotonic()}
    for key, path in [('buddyinfo', Path('/proc/buddyinfo')),
                      ('node0_meminfo', Path('/sys/devices/system/node/node0/meminfo')),
                      ('node1_meminfo', Path('/sys/devices/system/node/node1/meminfo'))]:
        try:
            report[key] = path.read_text().splitlines()
        except OSError as error:
            report[key] = {'status': 'unavailable', 'error': repr(error)}
    try:
        selected = []
        for line in Path('/proc/zoneinfo').read_text().splitlines():
            fields = line.split()
            if (line.startswith('Node ') or line.strip().startswith('pages free') or
                    fields and fields[0] in ('min', 'low', 'high', 'spanned', 'present', 'managed',
                                              'nr_free_pages', 'nr_free_cma')):
                selected.append(line)
        report['zoneinfo_selected'] = selected
    except OSError as error:
        report['zoneinfo_selected'] = {'status': 'unavailable', 'error': repr(error)}
    report['limitation'] = 'NUMA snapshots are observations; they do not establish CUDA OOM causality.'
    return report


def manifest(source, boundary_script, traced=False, numa_wrapper=None, ioctl_preload=None):
    helper_directory = Path(__file__).resolve().parent
    paths = [Path(sys.executable).resolve(), Path(__file__).resolve(), boundary_script,
             helper_directory / 'cuda_bootstrap_probe.py', helper_directory / 'cuda_bootstrap_target.py',
             helper_directory / 'resource_admission.py',
             source / 'src/fnit/recon_all/hemisphere_worker.py',
             source / 'src/fnit/recon_all/profiling.py', source / 'src/fnit/recon_all/thread_budget.py',
             source / 'src/fnit/recon_all/__init__.py']
    if traced:
        paths.append(Path('/usr/bin/strace'))
    if numa_wrapper is not None:
        paths.append(numa_wrapper.resolve())
    if ioctl_preload is not None:
        paths.append(ioctl_preload.resolve())
    return {str(path): {'bytes': path.stat().st_size, 'sha256': digest(path)} for path in paths}


def controller(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    boundary = args.boundary_script.resolve()
    bindings = manifest(source, boundary, args.strace, args.numa_wrapper, args.ioctl_preload)
    rng = random.Random(args.seed)
    plan = []
    for index in range(args.pairs):
        comparator = 'default_worker' if args.numa_wrapper else ('original_worker' if index % 2 == 0 else 'direct_priority')
        intervention = 'node1_worker' if args.numa_wrapper else 'context_driver'
        order = [comparator, intervention]
        rng.shuffle(order)
        plan.append({'pair': index, 'comparator': comparator, 'intervention': intervention, 'launch_order': order})
    env = dict(os.environ)
    env.update({key: '2' for key in THREAD_KEYS})
    env.update(CUDA_VISIBLE_DEVICES=GPU_UUID, PYTORCH_NO_CUDA_MEMORY_CACHING='1',
               PYTHONPATH=str(source / 'src') + os.pathsep + str(Path(__file__).resolve().parent),
               TORCH_SHOW_CPP_STACKTRACES='1')
    report = {'status': 'waiting_lock', 'started_utc': now(), 'pid': os.getpid(),
              'host': os.uname().nodename, 'source': str(source), 'python': sys.executable,
              'gpu_uuid': GPU_UUID, 'seed': args.seed, 'planned_pairs': plan,
              'source_script_manifest': bindings, 'child_environment': {key: env.get(key) for key in ENV_KEYS},
              'scope': 'scalar-only paired context diagnosis; no T1, no pipeline/production change',
              'ioctl_preload':str(args.ioctl_preload.resolve()) if args.ioctl_preload else None,
              'ioctl_scope':'fixed RM headers from own fresh scalar children only; not production' if args.ioctl_preload else 'disabled',
              'numa_wrapper': str(args.numa_wrapper.resolve()) if args.numa_wrapper else None,
              'numa_scope': 'Default-retained vs BIND-node1 before Torch import; diagnostic only; no production policy change' if args.numa_wrapper else 'disabled',
              'strace': {'enabled': args.strace, 'scope': 'Only newly created owned child trees; ioctl/mmap/munmap/mlock/munlock/brk; no file IO or full environment',
                         'timing_limitation': 'Traced timings include ptrace overhead and are not production timings'},
              'minimum_free_bytes': args.minimum_free_bytes,
              'admission_scope': 'scalar diagnosis only; not a lowered whole-case memory threshold',
              'threads': {'children_per_pair': 2, 'native_threads_per_child': 2, 'total_declared': 4,
                          'limitation': 'library thread settings do not bound aggregate process thread count'},
              'precision': 'FP32 scalar; TF32 true; no autocast; CUDA cache disabled',
              'timing_limitation': 'No first-call barrier. Child CPU import paths and intervention cost differ; '
                                   'Popen timestamps are not CUDA-call timestamps. Concurrent paired outcomes '
                                   'may still reflect differing moments of shared device state.',
              'inference_limitation': 'Discordant paired outcomes are diagnostic evidence, not proof of fix. '
                                      'No failed arm is cleared/retried in this controller.',
              'trials': []}
    write(root / 'manifest.json', {'created_utc': now(), 'source_script_manifest': bindings})
    write(root / 'summary.json', report)
    lock = args.lock.open('a+')
    held = False
    children, owned, streams = [], {}, []
    cancelled = []

    def cancelled_handler(signum, frame):
        # Do not interrupt the short spawn→ownership-registration critical section.
        cancelled.append({'signal': signum, 'utc': now()})

    def check_cancelled():
        if cancelled:
            report['cancellation_signals'] = list(cancelled)
            raise Interrupted(f"signal {cancelled[-1]['signal']}")

    def cleanup_trial():
        if not children:
            return
        snapshot_descendants([child.pid for child in children] + list(owned), owned)
        while active_owned(owned):
            try:
                cleanup_owned_tree(children[0], owned, report,
                                   lambda: write(root / 'summary.json', report))
            except Exception as error:
                # Cleanup errors cannot authorize release of the shared lock with active computation.
                report['cleanup_retry_error'] = repr(error)
                write(root / 'summary.json', report)
                time.sleep(.1)
                snapshot_descendants([child.pid for child in children] + list(owned), owned)
        for child in children:
            child.poll()
        for stream in streams:
            stream.close()
        streams.clear()

    handlers = {signum: signal.signal(signum, cancelled_handler)
                for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        started_wait = time.monotonic()
        while not held:
            check_cancelled()
            sample = gpu_snapshot()
            report['waiting_gpu'] = sample
            write(root / 'summary.json', report)
            if sample['free_bytes'] >= args.minimum_free_bytes:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = True
                except BlockingIOError:
                    pass
                if held:
                    sample = gpu_snapshot()
                    if sample['free_bytes'] < args.minimum_free_bytes:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                        held = False
            if not held:
                if time.monotonic() - started_wait >= args.maximum_wait_seconds:
                    raise TimeoutError('shared lock/scalar resource admission timed out')
                time.sleep(min(1.0, max(.01, args.maximum_wait_seconds - (time.monotonic() - started_wait))))
        report.update(status='running', under_lock_gpu=gpu_snapshot(),
                      admission_wait_seconds=time.monotonic() - started_wait)
        write(root / 'summary.json', report)
        for pair in plan:
            check_cancelled()
            # Check frozen source and actual diagnostic/helper binaries before each paired trial.
            if manifest(source, boundary, args.strace, args.numa_wrapper, args.ioctl_preload) != bindings:
                raise ValueError('source/script/interpreter manifest changed during matrix')
            trial = root / f"{pair['pair']:03d}_{pair['comparator']}_vs_{pair['intervention']}"
            trial.mkdir()
            row = {**pair, 'started_utc': now(), 'started_monotonic': time.monotonic(),
                   'gpu_before': gpu_snapshot(), 'before_host': numa_host_receipt(), 'children': []}
            report['trials'].append(row)
            children, owned, streams = [], {}, []
            # Prepare both requests before spawning, minimizing mode-dependent preparation gaps.
            commands = []
            for slot, mode in enumerate(pair['launch_order']):
                childroot = trial / f'{slot}_{mode}'
                output = childroot / 'boundary.json'
                policy_output = None
                if mode in ('original_worker', 'default_worker', 'node1_worker'):
                    childroot.mkdir()
                    request, output = childroot / 'request.json', childroot / 'worker.json'
                    write(request, {'callable': 'cuda_bootstrap_target:noop', 'operation': 'startup_probe',
                                    'kwargs': {}, 'device': 'cuda:0', 'threads': 2,
                                    'precision': {'matmul_tf32': True, 'cudnn_tf32': True},
                                    'profile_stages': True, 'allocator_policy': 'disabled'})
                    if args.numa_wrapper:
                        policy_output = childroot / 'policy.json'
                        command = [sys.executable, str(args.numa_wrapper.resolve()), '--policy',
                                   'default' if mode == 'default_worker' else 'node1',
                                   '--request', str(request), '--report', str(output),
                                   '--policy-report', str(policy_output)]
                    else:
                        command = [sys.executable, '-m', 'fnit.recon_all.hemisphere_worker',
                                   str(request), str(output)]
                else:
                    command = [sys.executable, str(boundary), '--mode', mode, '--output', str(childroot)]
                trace_path = trial / f'{slot}_{mode}.strace.log' if args.strace else None
                if args.strace:
                    command = ['/usr/bin/strace', '-f', '-ttt', '-T', '-e',
                               'trace=ioctl,mmap,munmap,mlock,munlock,brk', '-o', str(trace_path), *command]
                log_path = trial / f'{slot}_{mode}.log'
                stream = log_path.open('x')
                streams.append(stream)
                commands.append((mode, command, output, stream, log_path, trace_path, policy_output))
            for mode, command, output, stream, log_path, trace_path, policy_output in commands:
                check_cancelled()
                before = time.monotonic()
                started_utc = now()
                child_env=dict(env)
                if args.ioctl_preload:
                    if child_env.get('LD_PRELOAD'):raise ValueError('refusing to override existing LD_PRELOAD')
                    child_env['LD_PRELOAD']=str(args.ioctl_preload.resolve())
                    child_env['FNIT_NV_IOCTL_TRACE_PATH']=str(log_path.with_suffix('.ioctl.jsonl'))
                child = subprocess.Popen(command, env=child_env, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=str(source))
                # Always register ownership before honoring cancellation.
                children.append(child)
                snapshot_descendants([child.pid], owned)
                row['children'].append({'mode': mode, 'pid': child.pid, 'command': command,
                                        'report': str(output), 'raw_log': str(log_path),
                                        'ioctl_trace':child_env.get('FNIT_NV_IOCTL_TRACE_PATH'),
                                        'policy_report': str(policy_output) if policy_output is not None else None,
                                        'strace_log': str(trace_path) if trace_path is not None else None,
                                        'popen_before_utc': started_utc, 'popen_before_monotonic': before,
                                        'popen_after_monotonic': time.monotonic(), 'proc_receipts': []})
                check_cancelled()
            row['popen_start_gap_seconds'] = (row['children'][1]['popen_before_monotonic'] -
                                               row['children'][0]['popen_before_monotonic'])
            write(root / 'summary.json', report)
            deadline = time.monotonic() + args.child_timeout_seconds
            last_paths = {}
            while any(child.poll() is None for child in children):
                check_cancelled()
                snapshot_descendants([child.pid for child in children] + list(owned), owned)
                for child, item in zip(children, row['children']):
                    if child.poll() is None:
                        # With strace the direct child is the tracer; sample traced descendants too.
                        observed = proc_receipt(child.pid)
                        if args.strace:
                            observed['owned_descendant_receipts'] = [proc_receipt(identity['pid'])
                                for identity in active_owned(owned) if identity['pid'] != child.pid
                                and identity['pgid'] == child.pid]
                        key = (observed.get('loaded_cuda_paths', []),
                               [(item['pid'], item.get('loaded_cuda_paths', []))
                                for item in observed.get('owned_descendant_receipts', [])])
                        # Keep CPU receipt on first sample and library changes; no GPU API in children.
                        if child.pid not in last_paths or key != last_paths[child.pid]:
                            item['proc_receipts'].append(observed)
                            last_paths[child.pid] = key
                if time.monotonic() > deadline:
                    raise TimeoutError('paired diagnostic child timeout; no retry')
                time.sleep(.1)
            cleanup_trial()  # Shared lock is held until all descendants have exited.
            for child, item in zip(children, row['children']):
                item['exit_code'] = child.returncode
                output, log_path = Path(item['report']), Path(item['raw_log'])
                if output.exists():
                    item['result'] = json.loads(output.read_text())
                    item['report_sha256'] = digest(output)
                else:
                    item['report_status'] = 'missing'
                if item.get('policy_report'):
                    policy_path = Path(item['policy_report'])
                    if policy_path.exists():
                        item['policy_result'] = json.loads(policy_path.read_text())
                        item['policy_report_sha256'] = digest(policy_path)
                    else:
                        item['policy_report_status'] = 'missing'
                    policy = item.get('policy_result', {})
                    item['failure_category'] = ('policy_admission_failed' if policy.get('policy_admission') != 'passed'
                                                else 'policy_restoration_failed' if policy.get('restoration') == 'failed'
                                                else 'worker_failed' if item['exit_code'] != 0
                                                else 'none')
                if item.get('ioctl_trace'):
                    ioctl_path=Path(item['ioctl_trace'])
                    if ioctl_path.is_file():
                        item['ioctl_sha256']=digest(ioctl_path);item['ioctl_bytes']=ioctl_path.stat().st_size
                        try:
                            events=[json.loads(line) for line in ioctl_path.read_text().splitlines()]
                            item['ioctl_records']=len(events)
                            item['ioctl_uncopied_headers']=sum(not x.get('header_copied') for x in events)
                            item['ioctl_nonzero_statuses']=[e for e in events if e.get('status')]
                        except (ValueError,OSError) as ioerror:item['ioctl_parse_error']=repr(ioerror)
                    else:item['ioctl_trace_status']='unavailable; interception coverage not established'
                item['raw_log_sha256'] = digest(log_path)
                item['raw_log_bytes'] = log_path.stat().st_size
                if item.get('strace_log'):
                    trace_path = Path(item['strace_log'])
                    if trace_path.exists():
                        item['strace_log_sha256'] = digest(trace_path)
                        item['strace_log_bytes'] = trace_path.stat().st_size
                    else:
                        item['strace_status'] = 'missing'
                # Raw tracebacks/API returns remain in individual receipts/logs, never replaced by counts.
                item['passed'] = item['exit_code'] == 0 and item.get('result', {}).get('status') == 'complete'
            results = {item['mode']: item['passed'] for item in row['children']}
            comparator = pair['comparator']
            intervention = pair['intervention']
            row['outcome'] = ('both_pass' if all(results.values()) else 'both_fail' if not any(results.values())
                              else ('node1_only_pass' if args.numa_wrapper else 'driver_only_pass')
                              if results[intervention] else 'comparator_only_pass')
            row.update(finished_utc=now(), finished_monotonic=time.monotonic(),
                       gpu_after=gpu_snapshot(), after_host=numa_host_receipt())
            write(root / 'summary.json', report)
        if manifest(source, boundary, args.strace, args.numa_wrapper, args.ioctl_preload) != bindings:
            raise ValueError('source/script/interpreter manifest changed at final boundary')
        report['arm_results'] = {mode: {'passed': sum(item['passed'] for row in report['trials']
                                                     for item in row['children'] if item['mode'] == mode),
                                       'failed': sum(not item['passed'] for row in report['trials']
                                                     for item in row['children'] if item['mode'] == mode)}
                                 for mode in (('default_worker', 'node1_worker') if args.numa_wrapper else
                                              ('original_worker', 'direct_priority', 'context_driver'))}
        report['pair_outcomes'] = {comparator: {outcome: sum(row['outcome'] == outcome for row in report['trials']
                                                          if row['comparator'] == comparator)
                                               for outcome in ('both_pass', 'both_fail',
                                                               'node1_only_pass' if args.numa_wrapper else 'driver_only_pass',
                                                               'comparator_only_pass')}
                                   for comparator in (('default_worker',) if args.numa_wrapper else
                                                      ('original_worker', 'direct_priority'))}
        report['status'] = 'matrix_complete'
        report['interpretation'] = 'Paired diagnostic matrix completed; production fix and root cause not confirmed.'
        return 0
    except BaseException as error:
        report.update(status='failed_or_interrupted', error=repr(error), traceback=traceback.format_exc())
        return 1
    finally:
        cleanup_trial()
        if held:
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        report['finished_utc'] = now()
        report['finished_monotonic'] = time.monotonic()
        write(root / 'summary.json', report)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boundary-script', type=Path,
                        default=Path(__file__).with_name('cuda_runtime_boundary.py'))
    parser.add_argument('--ioctl-preload',type=Path,help='独立诊断.so，仅附着自己的新scalar进程')
    parser.add_argument('--numa-wrapper', type=Path, help='可选NUMA标量wrapper；对照default vs BIND-node1')
    parser.add_argument('--lock', type=Path, default=Path('/tmp/fnit-shared-benchmark.lock'))
    parser.add_argument('--pairs', type=int, choices=(8, 12), default=8)
    parser.add_argument('--strace', action='store_true', help='仅跟踪本轮自有子树的指定系统调用')
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--minimum-free-bytes', type=int, default=2_000_000_000)
    parser.add_argument('--maximum-wait-seconds', type=float, default=900)
    parser.add_argument('--child-timeout-seconds', type=float, default=90)
    args = parser.parse_args()
    if args.ioctl_preload and not args.ioctl_preload.is_file():parser.error('--ioctl-preload must exist')
    if args.ioctl_preload and os.environ.get('LD_PRELOAD'):parser.error('existing LD_PRELOAD is unsupported')
    if args.numa_wrapper and not args.numa_wrapper.is_file():
        parser.error('--numa-wrapper must exist')
    if args.strace and not Path('/usr/bin/strace').is_file():
        parser.error('--strace requires /usr/bin/strace')
    if (not args.source.is_dir() or not args.boundary_script.is_file() or
            args.minimum_free_bytes < 2_000_000_000 or args.maximum_wait_seconds < 0 or
            not 1 <= args.child_timeout_seconds <= 180):
        parser.error('source/boundary must exist; scalar free>=2GB; nonnegative wait; child timeout1..180s')
    return controller(args)


if __name__ == '__main__':
    raise SystemExit(main())
