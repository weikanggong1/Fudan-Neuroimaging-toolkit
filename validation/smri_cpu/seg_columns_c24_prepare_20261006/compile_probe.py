"""One separately authorized C24 compile/load probe: no tensors, copy or SGEMM."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from bindings import check_sources, check_runtime, flags, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'workspace', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--approved-contracts', action='store_true', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, exist_ok=False)
    report = {'schema': 'fnit_C24_compile_load/v1', 'status': 'started_not_accepted',
              'copy_calls': 0, 'SGEMM_calls': 0, 'MRI_calls': 0, 'native_calls': 0,
              'model_forward_calls': 0, 'GPU_calls': 0, 'tensor_allocation_calls': 0,
              'global_allocator_changed': False, 'compile_calls': 0, 'completed': False}
    started = time.monotonic()
    torch = None
    process = None
    try:
        plan = json.loads((args.workspace / 'PLAN.json').read_text())
        report['PLAN'] = identity(args.workspace / 'PLAN.json')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '8' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
            raise RuntimeError('CPU-only eight-thread environment required')
        if any(k in os.environ for k in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','OPENBLAS_CORETYPE')):
            raise RuntimeError('loader overrides forbidden')
        if sorted(os.sched_getaffinity(0)) != plan['affinity']:
            raise RuntimeError('frozen CPU affinity required')
        resource.setrlimit(resource.RLIMIT_AS, (4_000_000_000, 4_000_000_000))
        report['sources_before'] = check_sources(args.root, args.workspace, plan)
        sys.path.insert(0, str(args.root / 'repo/src'))
        import torch as loaded_torch
        torch = loaded_torch
        from prototype import ColumnsC24
        check_runtime(torch, plan)
        report['flags_before'] = flags(torch)
        if report['flags_before']['CUDA_initialized']:
            raise RuntimeError('unexpected CUDA initialization')
        cxx = Path(plan['compiler']['command'])
        if str(cxx.resolve(strict=True)) != plan['compiler']['resolved_path'] or identity(cxx.resolve())['sha256'] != plan['compiler']['binary_sha256']:
            raise RuntimeError('accepted GCC11 compiler path/bytes changed')
        version = subprocess.run([str(cxx), '--version'], capture_output=True, text=True, timeout=10, check=False)
        if version.returncode or version.stdout.strip() != plan['compiler']['version']:
            raise RuntimeError('accepted compiler version changed')
        report['compiler'] = plan['compiler']
        report['compiler_version_probe_calls'] = 1
        torch_root = Path(torch.__file__).resolve().parent
        library = args.output / 'columns_c24.so'
        command = [str(cxx), *plan['compile_flags'], '-I'+str(torch_root/'include'),
                   '-I'+str(torch_root/'include/torch/csrc/api/include'), str(args.workspace/'columns_c24.cpp'),
                   '-L'+str(torch_root/'lib'), '-ltorch_cpu', '-lc10', '-Wl,--no-undefined',
                   '-Wl,-z,relro', '-Wl,-z,now', '-Wl,-rpath,'+str(torch_root/'lib'),
                   '-Wl,-rpath-link,'+str(torch_root/'lib'), '-Wl,-rpath-link,'+str(Path(sys.prefix)/'lib'),
                   '-o', str(library)]
        report['compiler_argv'] = command
        report['compile_calls'] = 1
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, env=os.environ.copy())
        try:
            stdout, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=5)
            raise TimeoutError('one compiler child exceeded120s; no retry')
        (args.output/'compiler.stdout.log').write_bytes(stdout)
        (args.output/'compiler.stderr.log').write_bytes(stderr)
        report['compiler_rc'] = process.returncode
        if process.returncode:
            raise RuntimeError('new C24 compile failed; no retry')
        report['binary'] = identity(library)
        helper = ColumnsC24(library, plan['provider_sha256'], allow_compute=False)
        if helper.allow_compute or helper.allow_bounded_contracts:
            raise RuntimeError('metadata helper compute unexpectedly open')
        if ctypes.cast(ctypes.CDLL(None).sgemm_, ctypes.c_void_p).value != helper.sgemm_address:
            raise RuntimeError('global and loaded Torch SGEMM addresses differ')
        helper._provider_still_matches()
        report.update({'provider_sha256': helper.provider_sha256, 'provider_basename': helper.provider.name,
                       'Torch_handle_and_global_SGEMM_same_address': True, 'abi_description': 10404,
                       'new_exports': ['fnit_columns_c24_abi_description','fnit_copy_columns_c24_f32','fnit_same_provider_sgemm_c24_f32'],
                       'status': 'compiled_loaded_interface_only', 'completed': True})
    except BaseException as error:
        report['status'] = 'interface_failed_stopped'
        report['error_type'], report['error'] = type(error).__name__, str(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        try:
            report['sources_after'] = check_sources(args.root, args.workspace, plan)
            report['sources_unchanged'] = report.get('sources_before') == report['sources_after']
            report['compiler_bytes_unchanged'] = identity(Path(plan['compiler']['resolved_path']))['sha256'] == plan['compiler']['binary_sha256']
        except Exception as error:
            report['sources_unchanged'] = False
            report['postcondition_error'] = str(error)
        if torch is not None:
            report['flags_after'] = flags(torch)
            report['flags_unchanged'] = report.get('flags_before') == report['flags_after']
        report['worker_observation_seconds'] = time.monotonic()-started
        report['RSS_maximum_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['compiler_maxRSS_bytes'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024
        report['binary_unchanged'] = report.get('binary') == identity(args.output/'columns_c24.so') if (args.output/'columns_c24.so').is_file() else False
        report['valid_interface'] = bool(report['completed'] and report['binary_unchanged'] and report.get('sources_unchanged') and report.get('compiler_bytes_unchanged') and report.get('flags_unchanged') and not report.get('flags_after',{}).get('CUDA_initialized',True))
        (args.output/'COMPILE.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not report['valid_interface']:
        raise RuntimeError('interface final gate failed')
    print(json.dumps({'status': report['status'], 'valid_interface': True, 'copy_calls': 0, 'SGEMM_calls': 0, 'MRI_calls': 0}))


if __name__ == '__main__':
    main()
