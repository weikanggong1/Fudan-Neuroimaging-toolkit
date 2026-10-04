"""仅诊断：import Torch 前保留默认 NUMA 策略或 BIND-node1，再运行冻结 worker。

--policy default 保留原线程策略，不强制改成 MPOL_DEFAULT；node1 设置
MPOL_BIND，核验get_mempolicy返回值后才运行worker，不静默回退。
仅改变本进程主线程及之后创建线程的内存策略，不改CPU affinity、系统
策略、页缓存或GPU配置。finally恢复主线程原策略；辅助线程继承的策略
仅在其退出时消失。本包装器不拿共享锁；调度方必须持锁至进程树退出。
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import traceback

LIBNUMA = Path('/usr/lib64/libnuma.so.1.0.0')
MAXNODE = 1024
MPOL_DEFAULT = 0
MPOL_BIND = 2
MPOL_LOCAL = 4


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def host():
    report = {'pid': os.getpid(), 'ppid': os.getppid(),
              'native_thread_id': threading.get_native_id(), 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    for name in ('status', 'limits', 'cgroup', 'maps'):
        try:
            lines = (Path('/proc/self') / name).read_text().splitlines()
            if name == 'status':
                lines = [line for line in lines if line.startswith(
                    ('Cpus_allowed', 'Mems_allowed', 'VmRSS:', 'VmLck:', 'VmPin:', 'Threads:'))]
            elif name == 'maps':
                lines = [line for line in lines if any(text in line for text in ('libnuma', 'libcuda', 'libtorch'))]
            report[name] = lines
        except OSError as error:
            report[name] = {'status': 'unavailable', 'error': repr(error)}
    return report


class Receipt:
    def __init__(self, path, policy):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x') as stream:
            stream.write('{}\n')
        self.data = {'status': 'starting', 'policy': policy, 'started_utc': now(), 'events': [],
                     'before_library': host(), 'scope': 'NUMA scalar diagnostic only; no production policy change',
                     'mask_layout': {'maxnode_bits': MAXNODE,
                                     'unsigned_long_bits': ctypes.sizeof(ctypes.c_ulong) * 8,
                                     'words': (MAXNODE + ctypes.sizeof(ctypes.c_ulong) * 8 - 1) //
                                              (ctypes.sizeof(ctypes.c_ulong) * 8)},
                     'restoration_scope': 'Restore original main OS thread policy only; inherited policies of '
                                          'runtime helper threads persist until those threads/process exit.',
                     'timing_limitation': 'No shared first-CUDA-call barrier; paired calls can occur at different times.'}
        self.flush()

    def flush(self):
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(self.path)

    def event(self, name, **fields):
        self.data['events'].append({'event': name, 'utc': now(), 'monotonic': time.monotonic(),
                                   'native_thread_id': threading.get_native_id(), **fields})
        self.flush()


def nodes(mask):
    width = ctypes.sizeof(ctypes.c_ulong) * 8
    return [node for node in range(MAXNODE) if mask[node // width] & (1 << (node % width))]


def run(args):
    receipt = Receipt(args.policy_report, args.policy)
    report = receipt.data
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    Mask = ctypes.c_ulong * ((MAXNODE + bits - 1) // bits)
    original_mask, original_mode = Mask(), ctypes.c_int()
    library = get_policy = set_policy = None
    original_known = False
    attempted_bind = False
    original_exit = 1
    stage = 'request_validation'
    original_argv = list(sys.argv)
    try:
        # JSON syntax validation only; importing fnit/Torch before policy intervention is prohibited.
        request = json.loads(args.request.read_text())
        if not isinstance(request, dict) or not {'device', 'callable', 'kwargs', 'precision', 'threads'} <= request.keys():
            raise ValueError('invalid frozen-worker request object')
        if (request['callable'] != 'cuda_bootstrap_target:noop' or request['kwargs'] != {} or
                request['device'] != 'cuda:0' or request['threads'] != 2):
            raise ValueError('only declared scalar noop/cuda:0/two-thread request is supported')
        if args.report.exists():
            raise FileExistsError('refusing existing worker output')
        if args.report.resolve() == args.policy_report.resolve():
            raise ValueError('worker and policy reports must differ')
        report['request_sha256'] = file_hash(args.request)
        report['request_path'] = str(args.request.resolve())
        report['worker_report_path'] = str(args.report.resolve())
        stage = 'load_actual_libnuma'
        # Fixed, previously verified server library, never guessed fallback libraries.
        resolved = LIBNUMA.resolve(strict=True)
        library = ctypes.CDLL(str(resolved), use_errno=True)
        report['libnuma'] = {'requested_path': str(LIBNUMA), 'actual_path': str(resolved),
                             'sha256': file_hash(resolved), 'bytes': resolved.stat().st_size}
        ULongPointer = ctypes.POINTER(ctypes.c_ulong)
        get_policy = library.get_mempolicy
        get_policy.argtypes = [ctypes.POINTER(ctypes.c_int), ULongPointer, ctypes.c_ulong,
                               ctypes.c_void_p, ctypes.c_ulong]
        get_policy.restype = ctypes.c_int
        set_policy = library.set_mempolicy
        set_policy.argtypes = [ctypes.c_int, ULongPointer, ctypes.c_ulong]
        set_policy.restype = ctypes.c_int

        def get_receipt(name, mode, mask):
            receipt.event('before_' + name, maxnode=MAXNODE, addr=None, flags=0)
            ctypes.set_errno(0)
            rc = int(get_policy(ctypes.byref(mode), mask, MAXNODE, None, 0))
            error = ctypes.get_errno()
            receipt.event('after_' + name, returncode=rc, errno=error,
                          errno_string=os.strerror(error) if error else None,
                          mode=mode.value if rc == 0 else None,
                          nodes=nodes(mask) if rc == 0 else None,
                          mask_words=list(mask) if rc == 0 else None)
            if rc != 0:
                raise OSError(error, f'{name} failed; no fallback')

        stage = 'get_original_policy'
        get_receipt('get_mempolicy_original', original_mode, original_mask)
        original_known = True
        report['original_policy'] = {'mode': original_mode.value, 'nodes': nodes(original_mask),
                                     'mask_words': list(original_mask)}
        report['before_policy_intervention'] = host()
        if args.policy == 'node1':
            stage = 'set_bind_node1'
            target_mask = Mask()
            target_mask[1 // bits] = 1 << (1 % bits)
            receipt.event('before_set_mempolicy_node1', mode=MPOL_BIND, maxnode=MAXNODE,
                          nodes=[1], mask_words=list(target_mask))
            ctypes.set_errno(0)
            attempted_bind = True
            rc = int(set_policy(MPOL_BIND, target_mask, MAXNODE))
            error = ctypes.get_errno()
            receipt.event('after_set_mempolicy_node1', returncode=rc, errno=error,
                          errno_string=os.strerror(error) if error else None)
            if rc != 0:
                raise OSError(error, 'set_mempolicy BIND-node1 failed; worker not launched; no fallback')
        else:
            receipt.event('default_policy_retained', mode=original_mode.value, nodes=nodes(original_mask))
        stage = 'verify_effective_policy'
        effective_mode, effective_mask = ctypes.c_int(), Mask()
        get_receipt('get_mempolicy_effective', effective_mode, effective_mask)
        expected_mode = MPOL_BIND if args.policy == 'node1' else original_mode.value
        expected_nodes = [1] if args.policy == 'node1' else nodes(original_mask)
        if effective_mode.value != expected_mode or nodes(effective_mask) != expected_nodes:
            raise RuntimeError('effective NUMA policy mismatch; worker not launched; no fallback')
        report['effective_policy'] = {'mode': effective_mode.value, 'nodes': nodes(effective_mask)}
        report['before_worker_cpu'] = host()
        report['policy_admission'] = 'passed'
        report['worker_launched'] = True
        receipt.event('before_run_frozen_worker', module='fnit.recon_all.hemisphere_worker')
        stage = 'frozen_worker'
        sys.argv = ['fnit.recon_all.hemisphere_worker', str(args.request.resolve()), str(args.report.resolve())]
        try:
            result = runpy.run_module('fnit.recon_all.hemisphere_worker', run_name='__main__', alter_sys=True)
            report['worker_module_path'] = result.get('__file__')
            original_exit = 0
        except SystemExit as error:
            original_exit = error.code if isinstance(error.code, int) else 0 if error.code is None else 1
            report['worker_system_exit'] = repr(error.code)
        except BaseException as error:
            original_exit = 1
            report['worker_unhandled_exception'] = repr(error)
            report['worker_unhandled_traceback'] = traceback.format_exc()
        report['worker_exit_code'] = original_exit
        report['status'] = 'complete' if original_exit == 0 else 'worker_failed'
        receipt.event('after_run_frozen_worker', exit_code=original_exit)
    except BaseException as error:
        report.update(status='policy_failed' if stage != 'frozen_worker' else 'worker_failed',
                      failure_stage=stage, error=repr(error), traceback=traceback.format_exc())
        if report['status'] == 'policy_failed':
            report['policy_admission'] = 'failed'
            report['worker_launched'] = False
        print(report['traceback'], file=sys.stderr, flush=True)
        original_exit = 1
    finally:
        sys.argv = original_argv
        report['after_worker_host'] = host()  # CPU/proc only, never CUDA observations or error clearing.
        if original_known:
            try:
                # A retained-default arm did not mutate policy, but still restore/verify the original thread policy.
                base_mode = original_mode.value & 0x7  # Preserve flags in set call; inspect base mode only.
                use_null = base_mode in (MPOL_DEFAULT, MPOL_LOCAL)
                restore_mask = None if use_null else original_mask
                restore_maxnode = 0 if use_null else MAXNODE
                receipt.event('before_restore_original_mempolicy', mode=original_mode.value,
                              maxnode=restore_maxnode, nodes=nodes(original_mask),
                              attempted_bind=attempted_bind)
                ctypes.set_errno(0)
                rc = int(set_policy(original_mode.value, restore_mask, restore_maxnode))
                error = ctypes.get_errno()
                receipt.event('after_restore_original_mempolicy', returncode=rc, errno=error,
                              errno_string=os.strerror(error) if error else None)
                if rc != 0:
                    raise OSError(error, 'original main-thread policy restoration failed')
                restored_mode, restored_mask = ctypes.c_int(), Mask()
                get_receipt('get_mempolicy_restored', restored_mode, restored_mask)
                if restored_mode.value != original_mode.value or nodes(restored_mask) != nodes(original_mask):
                    raise RuntimeError('restored policy differs from original policy')
                report['restoration'] = 'original_main_thread_policy_restored'
            except BaseException as error:
                report['status_before_restore_failure'] = report['status']
                report.update(status='policy_failed', restoration='failed', restoration_error=repr(error),
                              restoration_traceback=traceback.format_exc())
                original_exit = 1
        else:
            report['restoration'] = 'not_applicable_original_policy_unavailable_no_set_attempt'
        report['exit_code'] = original_exit
        report['finished_utc'] = now()
        receipt.flush()
    return original_exit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=('default', 'node1'), required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--policy-report', type=Path, required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
