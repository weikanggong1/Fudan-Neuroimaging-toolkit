"""独立 CUDA runtime 边界诊断；只供锁内新进程调用，不改变生产源码。

--mode 指定首 runtime/context 干预，--output 必须是不存在的诊断目录。
仅分配一个 FP32 标量；保留 TF32，不启 autocast。GPU 可见 UUID 由父进程
CUDA_VISIBLE_DEVICES 指定。本脚本不获取锁，父调度须全程持有共享锁。
所有 ctypes 库均取自本进程 maps，使用 RTLD_NOLOAD，拒绝猜测 stub 路径。
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time
import traceback
import uuid

MODES = ('direct_priority', 'context_sync', 'context_driver', 'sticky_probe')
ENV_KEYS = (
    'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_MODULE_LOADING',
    'CUDA_LAUNCH_BLOCKING', 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
    'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT', 'CUDA_MPS_PIPE_DIRECTORY',
    'PYTORCH_NO_CUDA_MEMORY_CACHING', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF',
    'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'TORCH_SHOW_CPP_STACKTRACES',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NUMBA_NUM_THREADS',
)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def mapped_paths():
    result = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = fields[5]
            if any(token in Path(path).name for token in
                   ('libcuda', 'libcudnn', 'libcublas', 'libtorch_cuda', 'libc10_cuda')):
                result.append(path)
    return sorted(set(result))


def host_snapshot():
    """只读 CPU/proc 状态；不调用 CUDA/NVML，不初始化 Numba 线程池。"""
    result = {'pid': os.getpid(), 'ppid': os.getppid(),
              'python_thread_id': threading.get_ident(),
              'native_thread_id': threading.get_native_id(),
              'resource_environment': {key: os.environ.get(key) for key in ENV_KEYS}}
    for key, path in [('loaded_cuda_libraries', '/proc/self/maps'),
                      ('status', '/proc/self/status'), ('cgroup', '/proc/self/cgroup')]:
        try:
            if key == 'loaded_cuda_libraries':
                result[key] = mapped_paths()
            elif key == 'status':
                result[key] = [line for line in Path(path).read_text().splitlines()
                               if line.startswith(('VmRSS:', 'VmLck:', 'VmPin:', 'Threads:'))]
            else:
                result[key] = Path(path).read_text().splitlines()
        except OSError as error:
            result[key] = {'status': 'unavailable', 'error': repr(error)}
    result['limits'] = {name: list(resource.getrlimit(getattr(resource, name)))
                        for name in ('RLIMIT_AS', 'RLIMIT_DATA', 'RLIMIT_MEMLOCK',
                                     'RLIMIT_NPROC', 'RLIMIT_NOFILE')}
    try:
        result['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    except OSError as error:
        result['cpu_affinity'] = {'error': repr(error)}
    try:
        result['meminfo'] = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                             for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.split(':', 1)[0] in
                             ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree', 'Mlocked')}
    except OSError as error:
        result['meminfo'] = {'error': repr(error)}
    # v2 cgroup receipt only; no claim about other/ancestor limits when unavailable.
    for entry in result.get('cgroup', []) if isinstance(result.get('cgroup'), list) else []:
        if entry.startswith('0::'):
            base = Path('/sys/fs/cgroup') / entry[3:].lstrip('/')
            receipt = {'path': str(base), 'scope': 'own v2 cgroup; ancestor limits not inferred'}
            for name in ('memory.current', 'memory.max', 'memory.events', 'pids.current', 'pids.max'):
                try:
                    receipt[name] = (base / name).read_text().strip()
                except OSError as error:
                    receipt[name] = {'status': 'unavailable', 'error': repr(error)}
            result['cgroup_v2'] = receipt
    return result


def hash_library(path):
    """末尾 CPU 读取；不把较慢 hash 放在首 CUDA 干预之前。"""
    try:
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return {'sha256': digest.hexdigest(), 'bytes': Path(path).stat().st_size}
    except OSError as error:
        return {'status': 'unavailable', 'error': repr(error)}


def attach_loaded(kind):
    paths = mapped_paths()
    if kind == 'runtime':
        candidates = [path for path in paths if Path(path).name.startswith('libcudart.so')]
    else:
        candidates = [path for path in paths if Path(path).name.startswith('libcuda.so')]
    if len(candidates) != 1:
        raise RuntimeError(f'{kind}: require exactly one actually mapped library, got {candidates}')
    path = candidates[0]
    if path.endswith(' (deleted)'):
        raise RuntimeError(f'{kind}: mapped library deleted; refusing path substitution')
    if not hasattr(os, 'RTLD_NOLOAD'):
        raise RuntimeError('RTLD_NOLOAD unavailable; refusing ordinary library load')
    # Reuse the actual mapped object, never find_library or a guessed CUDA path.
    return ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_NOLOAD), path


def bind(library, name, arguments, result=ctypes.c_int):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = result
    return function


class CUDAError(RuntimeError):
    pass


class Receipt:
    def __init__(self, directory, mode):
        directory.mkdir(parents=True, exist_ok=False)
        self.path = directory / 'boundary.json'
        self.data = {'status': 'starting', 'mode': mode, 'started_utc': utc(),
                     'events': [], 'before_import': host_snapshot(),
                     'scope': 'independent runtime diagnostic; no T1; not a production fix',
                     'precision': 'float32 scalar; TF32 true; autocast not enabled',
                     'lock_scope': 'caller must hold shared GPU lock through child cleanup'}
        self.flush()

    def flush(self):
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(self.path)

    def event(self, boundary, **fields):
        event = {'boundary': boundary, 'utc': utc(), 'monotonic': time.monotonic(),
                 'native_thread_id': threading.get_native_id(), **fields}
        self.data['events'].append(event)
        self.flush()
        return event

    def api(self, name, function, *arguments, outputs=None, require_success=True):
        self.event('before_' + name)
        code = int(function(*arguments))
        details = outputs() if outputs is not None and code == 0 else None
        self.event('after_' + name, errorcode=code, outputs=details)
        if code != 0 and require_success:
            raise CUDAError(f'{name} returned errorcode={code}; no error clearing attempted')
        return code


def run(args):
    receipt = Receipt(args.output, args.mode)
    report = receipt.data
    phase = 'before_import'
    driver = runtime = None
    driver_path = runtime_path = None
    retained_device = None
    retained_context = ctypes.c_void_p()
    old_context = ctypes.c_void_p()
    restore_current = False
    scalar = None
    try:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible.startswith('GPU-') or ',' in visible:
            raise ValueError('diagnostic requires exactly one full GPU UUID in CUDA_VISIBLE_DEVICES')
        expected_uuid = str(uuid.UUID(visible[4:]))
        report['declared_device'] = {'logical': 'cuda:0', 'visible_uuid': visible}
        phase = 'import_torch'
        receipt.event('before_import_torch')
        import torch
        from fnit.recon_all.profiling import configure_cuda_allocator
        from fnit.recon_all.thread_budget import thread_budget  # Same pre-CUDA Numba import as worker.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_num_interop_threads(1)
        report['allocator'] = configure_cuda_allocator('cuda:0', 'disabled')
        report['torch'] = {'version': torch.__version__, 'cuda_build': torch.version.cuda,
                           'module_path': str(Path(torch.__file__).resolve()),
                           'intraop_threads': torch.get_num_threads(),
                           'interop_threads': torch.get_num_interop_threads(),
                           'numba_prebootstrap': 'imported, thread pool not observed or initialized'}
        report['before_cuda'] = host_snapshot()
        report['initialized_before_explicit_init'] = torch.cuda.is_initialized()
        if report['initialized_before_explicit_init']:
            raise RuntimeError('CUDA already initialized before designated first boundary')
        phase = 'torch_cuda_init'
        receipt.event('before_torch_cuda_init')
        torch.cuda.init()
        receipt.event('after_torch_cuda_init')
        phase = 'attach_actual_loaded_libraries'
        runtime, runtime_path = attach_loaded('runtime')
        driver, driver_path = attach_loaded('driver')
        report['library_paths'] = {'runtime': runtime_path, 'driver': driver_path}
        receipt.event('attached_loaded_libraries', paths=report['library_paths'])
        int_pointer = ctypes.POINTER(ctypes.c_int)
        priority = bind(runtime, 'cudaDeviceGetStreamPriorityRange', [int_pointer, int_pointer])
        low, high = ctypes.c_int(), ctypes.c_int()

        if args.mode == 'context_sync':
            phase = 'torch_cuda_synchronize_first'
            receipt.event('before_torch_cuda_synchronize_first')
            torch.cuda.synchronize(torch.device('cuda:0'))
            receipt.event('after_torch_cuda_synchronize_first')
        elif args.mode == 'context_driver':
            # Every driver call is an explicit intervention after torch.cuda.init.
            phase = 'driver_device_mapping'
            device = ctypes.c_int()
            device_get = bind(driver, 'cuDeviceGet', [int_pointer, ctypes.c_int])
            receipt.api('cuDeviceGet', device_get, ctypes.byref(device), 0,
                        outputs=lambda: {'device': device.value})
            class CUuuid(ctypes.Structure):
                _fields_ = [('bytes', ctypes.c_ubyte * 16)]
            actual = CUuuid()
            uuid_name = 'cuDeviceGetUuid_v2' if hasattr(driver, 'cuDeviceGetUuid_v2') else 'cuDeviceGetUuid'
            device_uuid = bind(driver, uuid_name, [ctypes.POINTER(CUuuid), ctypes.c_int])
            receipt.api(uuid_name, device_uuid, ctypes.byref(actual), device.value,
                        outputs=lambda: {'uuid': str(uuid.UUID(bytes=bytes(actual.bytes)))})
            actual_uuid = str(uuid.UUID(bytes=bytes(actual.bytes)))
            report['driver_device_uuid'] = actual_uuid
            if actual_uuid != expected_uuid:
                raise ValueError(f'driver device UUID mismatch: {actual_uuid} != {expected_uuid}')
            get_current = bind(driver, 'cuCtxGetCurrent', [ctypes.POINTER(ctypes.c_void_p)])
            receipt.api('cuCtxGetCurrent', get_current, ctypes.byref(old_context),
                        outputs=lambda: {'context': old_context.value})
            phase = 'driver_primary_ctx_retain'
            retain = bind(driver, 'cuDevicePrimaryCtxRetain',
                          [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int])
            receipt.api('cuDevicePrimaryCtxRetain', retain, ctypes.byref(retained_context), device.value,
                        outputs=lambda: {'context': retained_context.value})
            retained_device = device.value  # Release only this successful, owned retain.
            set_current = bind(driver, 'cuCtxSetCurrent', [ctypes.c_void_p])
            phase = 'driver_set_current'
            restore_current = True  # Also restore if set-current reports a failure.
            receipt.api('cuCtxSetCurrent', set_current, retained_context)

        if args.mode == 'sticky_probe':
            report['sticky_probe_scope'] = (
                'Explicit diagnostic API sequence; cudaGetLastError clears this OS thread error state. '
                'No scalar recovery or production success inferred.')
            peek = bind(runtime, 'cudaPeekAtLastError', [])
            get_last = bind(runtime, 'cudaGetLastError', [])
            name = bind(runtime, 'cudaGetErrorName', [ctypes.c_int], ctypes.c_char_p)
            description = bind(runtime, 'cudaGetErrorString', [ctypes.c_int], ctypes.c_char_p)
            phase = 'sticky_diagnostic_sequence'
            codes = []
            codes.append(receipt.api('cudaPeekAtLastError_before', peek, require_success=False))
            codes.append(receipt.api('cudaDeviceGetStreamPriorityRange', priority,
                                     ctypes.byref(low), ctypes.byref(high),
                                     outputs=lambda: {'least_priority': low.value, 'greatest_priority': high.value},
                                     require_success=False))
            codes.append(receipt.api('cudaPeekAtLastError_after', peek, require_success=False))
            receipt.event('before_explicit_thread_error_clear', intervention='cudaGetLastError')
            codes.append(receipt.api('cudaGetLastError', get_last, require_success=False))
            for index, code in enumerate(codes):
                # Error strings decode codes; these calls are logged separately, not GPU health checks.
                receipt.event('decode_runtime_error', sequence_index=index, errorcode=code,
                              error_name=(name(code) or b'').decode('utf-8', errors='replace'),
                              error_string=(description(code) or b'').decode('utf-8', errors='replace'))
            device = ctypes.c_int()
            device_get = bind(driver, 'cuDeviceGet', [int_pointer, ctypes.c_int])
            code = receipt.api('cuDeviceGet_for_state', device_get, ctypes.byref(device), 0,
                               outputs=lambda: {'device': device.value}, require_success=False)
            state_codes = [code]
            if code == 0:
                flags, active = ctypes.c_uint(), ctypes.c_int()
                state = bind(driver, 'cuDevicePrimaryCtxGetState',
                             [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), int_pointer])
                state_codes.append(receipt.api('cuDevicePrimaryCtxGetState', state, device.value,
                                               ctypes.byref(flags), ctypes.byref(active),
                                               outputs=lambda: {'flags': flags.value, 'active': active.value},
                                               require_success=False))
            report['status'] = 'diagnostic_errors_observed' if any(codes + state_codes) else 'complete'
            report['observed_runtime_errorcodes'] = codes
            report['observed_driver_errorcodes'] = state_codes
        else:
            if args.mode in ('direct_priority', 'context_driver'):
                phase = 'direct_priority_range'
                receipt.api('cudaDeviceGetStreamPriorityRange', priority,
                            ctypes.byref(low), ctypes.byref(high),
                            outputs=lambda: {'least_priority': low.value, 'greatest_priority': high.value})
            phase = 'first_float32_allocation'
            receipt.event('before_empty_float32')
            scalar = torch.empty(1, dtype=torch.float32, device='cuda:0')
            receipt.event('after_empty_float32', dtype=str(scalar.dtype), device=str(scalar.device))
            phase = 'postallocation_synchronize'
            receipt.event('before_postallocation_synchronize')
            torch.cuda.synchronize(torch.device('cuda:0'))
            receipt.event('after_postallocation_synchronize')
            # Device/precision observations occur only after the whole tested sequence succeeded.
            props = torch.cuda.get_device_properties(torch.device('cuda:0'))
            actual_uuid = str(getattr(props, 'uuid', 'unavailable')).removeprefix('GPU-')
            report['actual_device'] = {'name': props.name, 'uuid': actual_uuid,
                                       'total_memory': props.total_memory}
            if actual_uuid != 'unavailable' and str(uuid.UUID(actual_uuid)) != expected_uuid:
                raise ValueError('actual Torch device UUID mismatch')
            report['actual_precision'] = {
                'dtype': str(scalar.dtype), 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                'autocast_enabled': bool(torch.is_autocast_enabled())}
            report['torch_memory_stats_status'] = 'unavailable_cache_disabled'
            report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', failure_phase=phase,
                      error=repr(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        # No postfailure CUDA observations. Driver restore/release only balance owned intervention.
        report['after_sequence_host'] = host_snapshot()
        if retained_device is not None:
            try:
                if restore_current:
                    function = bind(driver, 'cuCtxSetCurrent', [ctypes.c_void_p])
                    code = receipt.api('cleanup_restore_cuCtxSetCurrent', function, old_context,
                                       require_success=False)
                    if code:
                        report['cleanup_error'] = f'context restore returned {code}'
                function = bind(driver, 'cuDevicePrimaryCtxRelease_v2' if
                                hasattr(driver, 'cuDevicePrimaryCtxRelease_v2') else
                                'cuDevicePrimaryCtxRelease', [ctypes.c_int])
                code = receipt.api('cleanup_release_owned_primary_ctx', function, retained_device,
                                   require_success=False)
                report['owned_primary_ctx_release_errorcode'] = code
                if code:
                    report['cleanup_error'] = f'owned context release returned {code}'
            except BaseException as error:
                report['cleanup_error'] = repr(error)
        if report.get('cleanup_error'):
            report['status_before_cleanup_error'] = report['status']
            report['status'] = 'cleanup_failed'
        # Do not call any runtime API to delete/recover a scalar after a failed boundary.
        # The disposable process owns the scalar and exits immediately after its final receipt.
        paths = report.get('library_paths', {})
        report['library_hashes'] = {kind: hash_library(path) for kind, path in paths.items()}
        report['finished_utc'] = utc()
        receipt.flush()
    return 0 if report['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
