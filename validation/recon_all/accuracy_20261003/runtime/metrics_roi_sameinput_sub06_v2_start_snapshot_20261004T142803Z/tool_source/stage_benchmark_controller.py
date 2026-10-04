"""显式JSON/argv阶段benchmark外层：核验、10GB准入、共享锁、监测和自有树清理。

仅用于annotation阶段或startup标量，不用于整例。--config指定固定解释器、
源码、脚本、argv、输入清单和新产物/诊断目录；不执行shell/eval，不重试算法。
所有已登记自有后代退出后才释放锁；失败、取消、超时不会因清理成功改成complete。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
import uuid

from resource_admission import snapshot_descendants, active_owned, cleanup_owned_tree, digest, now

THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
               'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NUMBA_NUM_THREADS')
ENV_KEYS = set(THREAD_KEYS) | {'CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE',
                             'PYTORCH_NO_CUDA_MEMORY_CACHING', 'NUMBA_CACHE_DIR', 'PYTHONPATH',
                             'TORCH_SHOW_CPP_STACKTRACES', 'FS_LICENSE', 'FREESURFER_HOME'}
MINIMUM_FREE_BYTES = 10_000_000_000


class Interrupted(Exception):
    pass


def write(path, report):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def finite_seconds(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(name + ' must be finite and >= ' + str(minimum))
    return float(value)


def validate_config(raw):
    required = {'python', 'source', 'script', 'args', 'manifest_files', 'output', 'diagnostic_root',
                'lock', 'gpu_uuid', 'minimum_free_bytes', 'threads', 'benchmark_kind'}
    if not isinstance(raw, dict) or not required <= raw.keys():
        raise ValueError('missing explicit stage benchmark config fields')
    if raw['benchmark_kind'] not in ('annotation_stage', 'startup_scalar'):
        raise ValueError('controller is stage/scalar only; whole case uses its20GB protocol')
    if not isinstance(raw['threads'], int) or raw['threads'] != 4 or isinstance(raw['threads'], bool):
        raise ValueError('total native thread budget must be4')
    if (not isinstance(raw['minimum_free_bytes'], int) or isinstance(raw['minimum_free_bytes'], bool)
            or raw['minimum_free_bytes'] < MINIMUM_FREE_BYTES):
        raise ValueError('stage admission free budget must be>=10,000,000,000 bytes')
    if not isinstance(raw['gpu_uuid'], str) or not raw['gpu_uuid'].startswith('GPU-') or ',' in raw['gpu_uuid']:
        raise ValueError('one explicit full GPU UUID required')
    uuid.UUID(raw['gpu_uuid'][4:])
    if not isinstance(raw['args'], list) or any(not isinstance(arg, str) for arg in raw['args']):
        raise ValueError('args must be a list of literal argv strings')
    if os.environ.get('LD_PRELOAD', '').strip():
        raise ValueError('regular stage benchmark requires inherited LD_PRELOAD empty; diagnostic preload must be separate')
    config = dict(raw)
    for key in ('python', 'source', 'script', 'output', 'diagnostic_root', 'lock'):
        if not isinstance(raw[key], str) or not Path(raw[key]).is_absolute():
            raise ValueError(key + ' must be an absolute path')
        config[key] = Path(raw[key]).resolve()
    if not config['source'].is_dir() or not (config['source'] / 'src').is_dir():
        raise ValueError('source must contain the declared src directory')
    for key in ('python', 'script'):
        if not config[key].is_file():
            raise FileNotFoundError(key)
    if not os.access(config['python'], os.X_OK) or config['script'].suffix != '.py':
        raise ValueError('executable Python and .py script required')
    if config['script'].name in ('execute_whole_case.py', 'run_whole.py'):
        raise ValueError('whole-case runner is outside this controller scope')
    monitor = Path(raw.get('monitor_script', str(config['source'] /
                      'validation/recon_all/python_gpu_port/run_monitored.py'))).resolve()
    if not monitor.is_file():
        raise FileNotFoundError('run_monitored.py')
    config['monitor_script'] = monitor
    actual_outputs = []
    for index, argument in enumerate(raw['args']):
        if argument == '--output':
            if index + 1 >= len(raw['args']):
                raise ValueError('--output needs its declared path')
            actual_outputs.append(Path(raw['args'][index + 1]).resolve())
        elif argument.startswith('--output='):
            actual_outputs.append(Path(argument.split('=', 1)[1]).resolve())
    if actual_outputs != [config['output']]:
        raise ValueError('argv must bind exactly one --output to config.output')
    if config['output'].exists() or config['diagnostic_root'].exists():
        raise FileExistsError('output and diagnostic_root must both be new directories')
    if overlap(config['output'], config['diagnostic_root']):
        raise ValueError('stage output and diagnostic_root must be separate')
    entries = raw['manifest_files']
    if not isinstance(entries, list) or not entries:
        raise ValueError('manifest_files must explicitly bind input/program files')
    declared = []
    for entry in entries:
        if not isinstance(entry, dict) or not {'path', 'size_bytes', 'sha256'} <= entry.keys():
            raise ValueError('each manifest file needs path,size_bytes,sha256')
        if not isinstance(entry['path'], str) or not Path(entry['path']).is_absolute():
            raise ValueError('manifest file path must be absolute')
        path = Path(entry['path']).resolve()
        if 'license' in path.name.lower() and path.suffix != '.py':
            raise ValueError('license contents must not be included in manifest inventory')
        checksum = entry['sha256']
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('invalid SHA256')
        if not isinstance(entry['size_bytes'], int) or isinstance(entry['size_bytes'], bool) or entry['size_bytes'] < 0:
            raise ValueError('invalid manifest file size')
        declared.append({**entry, 'path': str(path)})
    if len({entry['path'] for entry in declared}) != len(declared):
        raise ValueError('duplicate manifest file')
    config['manifest_files'] = declared
    environment = raw.get('env', {})
    if not isinstance(environment, dict) or any(key not in ENV_KEYS or not isinstance(value, str)
                                              for key, value in environment.items()):
        raise ValueError('env requires permitted explicit string overrides')
    config['env'] = environment
    # No new directory may overlap declared inputs, source, assets or program files.
    protected = [config['source'], config['script'], config['python'], monitor,
                 Path(__file__).resolve(), Path(__file__).with_name('resource_admission.py').resolve(),
                 *[Path(entry['path']) for entry in declared]]
    for key in ('assets', 'checkpoint', 'protected_paths'):
        value = raw.get(key)
        if value is not None:
            paths = value if isinstance(value, list) else [value]
            protected.extend(Path(path).resolve() for path in paths)
    for key in ('FREESURFER_HOME', 'FS_LICENSE'):
        if environment.get(key):
            protected.append(Path(environment[key]).resolve())
    # Protect original subject/assets roots stated in actual child argv as well.
    for index, argument in enumerate(raw['args']):
        if argument in ('--checkpoint', '--assets', '--weights', '--input-manifest') and index + 1 < len(raw['args']):
            protected.append(Path(raw['args'][index + 1]).resolve())
        elif any(argument.startswith(key + '=') for key in ('--checkpoint', '--assets', '--weights', '--input-manifest')):
            protected.append(Path(argument.split('=', 1)[1]).resolve())
    for entry in declared:
        path = Path(entry['path'])
        if path.parent.name in ('mri', 'surf', 'label', 'stats', 'average'):
            protected.append(path.parent.parent)
        elif path.parent.name == 'bem' and path.parent.parent.name == 'lib':
            protected.append(path.parents[2])
    for destination in (config['output'], config['diagnostic_root']):
        if any(overlap(destination, original) for original in protected):
            raise ValueError('new output overlaps declared inputs/source/assets/program')
    if any(overlap(config['lock'], destination) for destination in
           (config['output'], config['diagnostic_root'])):
        raise ValueError('shared lock cannot be inside new output directories')
    config['maximum_wait_seconds'] = finite_seconds(raw.get('maximum_wait_seconds', 3600), 'maximum_wait_seconds')
    config['timeout_seconds'] = finite_seconds(raw.get('timeout_seconds', 3600), 'timeout_seconds', 1)
    config['query_timeout_seconds'] = finite_seconds(raw.get('query_timeout_seconds', 5), 'query_timeout_seconds', .1)
    if config['query_timeout_seconds'] > 60:
        raise ValueError('query_timeout_seconds must be<=60')
    return config


def inventory(config):
    bindings = {}
    for entry in config['manifest_files']:
        path = Path(entry['path'])
        actual = {'size_bytes': path.stat().st_size, 'sha256': digest(path)}
        if actual != {'size_bytes': entry['size_bytes'], 'sha256': entry['sha256']}:
            raise ValueError('declared manifest differs: ' + str(path))
        bindings[str(path)] = actual
    # Only source .py and explicit program/input files; never rescan all weights/assets/licenses.
    programs = [config['python'], config['script'], config['monitor_script'], Path(__file__).resolve(),
                Path(__file__).with_name('resource_admission.py').resolve()]
    programs.extend(path for path in (config['source'] / 'src').rglob('*.py')
                    if '.git' not in path.parts and '__pycache__' not in path.parts)
    for path in programs:
        path = path.resolve()
        if str(path) not in bindings:
            bindings[str(path)] = {'size_bytes': path.stat().st_size, 'sha256': digest(path)}
    return bindings


def gpu_query(gpu_uuid, timeout):
    result = subprocess.run(['nvidia-smi', '--id=' + gpu_uuid,
                             '--query-gpu=uuid,memory.total,memory.used,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            timeout=timeout, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError('exactly one target GPU required')
    observed, total, used, free = [value.strip() for value in lines[0].split(',')]
    if observed != gpu_uuid:
        raise ValueError('physical GPU UUID mismatch')
    values = {'total_bytes': int(total) * 1048576, 'used_bytes': int(used) * 1048576,
              'free_bytes': int(free) * 1048576}
    if min(values.values()) < 0:
        raise ValueError('invalid GPU memory query')
    return {'utc': now(), 'monotonic': time.monotonic(), 'gpu_uuid': observed, **values,
            'measurement': 'integer MiB converted to bytes; separate from process-app sample'}


def cpu_receipt():
    def usage(who):
        value = resource.getrusage(who)
        return {'user_seconds': value.ru_utime, 'system_seconds': value.ru_stime,
                'maxrss_kib_linux': value.ru_maxrss, 'minor_faults': value.ru_minflt, 'major_faults': value.ru_majflt}
    return {'utc': now(), 'self': usage(resource.RUSAGE_SELF), 'reaped_children': usage(resource.RUSAGE_CHILDREN),
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'limits': {name: list(resource.getrlimit(getattr(resource, name)))
                       for name in ('RLIMIT_AS', 'RLIMIT_DATA', 'RLIMIT_MEMLOCK', 'RLIMIT_NOFILE', 'RLIMIT_NPROC')},
            'scope': 'rusage self/reaped children; not a continuous complete CPU process-tree profile'}


def run(config_path):
    start = time.monotonic()
    initial_utc = now()
    initial_cpu = cpu_receipt()
    raw = json.loads(config_path.read_text())
    config = validate_config(raw)
    bindings = inventory(config)  # Verify original receipts before creating diagnostic output or querying GPU.
    diag = config['diagnostic_root']
    diag.mkdir(parents=True, exist_ok=False)
    report_path = diag / 'controller.json'
    write(diag / 'inventory.json', bindings)
    report = {'status': 'waiting_resources', 'started_utc': initial_utc, 'pid': os.getpid(),
              'config_path': str(config_path.resolve()), 'config_sha256': digest(config_path),
              'benchmark_kind': config['benchmark_kind'], 'source': str(config['source']),
              'output': str(config['output']), 'diagnostic_root': str(diag),
              'gpu_uuid': config['gpu_uuid'], 'minimum_free_bytes': config['minimum_free_bytes'],
              'budget_scope': '10GB-or-higher stage/scalar admission only; whole-case20GB protocol unchanged',
              'inventory_sha256': digest(diag / 'inventory.json'), 'gpu_admission_samples': [],
              'cpu_before': initial_cpu, 'initial_validation_inventory_seconds': time.monotonic() - start,
              'algorithm_retry': False,
              'sampling_requested_seconds': .5, 'tree_snapshot_requested_seconds': .1}
    environment = dict(os.environ)
    environment.update(config['env'])
    environment.update({key: '4' for key in THREAD_KEYS})
    environment.update(CUDA_VISIBLE_DEVICES=config['gpu_uuid'], PYTORCH_NO_CUDA_MEMORY_CACHING='1',
                       PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=str(config['source'] / 'src') + os.pathsep + str(config['script'].parent))
    environment['TORCH_SHOW_CPP_STACKTRACES'] = config['env'].get('TORCH_SHOW_CPP_STACKTRACES', '1')
    environment['NUMBA_CACHE_DIR'] = config['env'].get('NUMBA_CACHE_DIR', str(diag / 'numba_cache'))
    report['child_environment'] = {key: environment.get(key) for key in sorted(ENV_KEYS)}
    write(report_path, report)
    lock = config['lock'].open('a+')
    held = False
    child = None
    owned = {}
    cancelled = []
    result_code = 1
    stream = None
    tree_times = []

    def signal_handler(signum, frame):
        cancelled.append({'signal': signum, 'utc': now()})

    def check_cancelled():
        if cancelled:
            report['cancellation_signals'] = list(cancelled)
            raise Interrupted('signal ' + str(cancelled[-1]['signal']))

    def write_report():
        write(report_path, report)

    handlers = {signum: signal.signal(signum, signal_handler) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        waiting_start = time.monotonic()
        while not held:
            check_cancelled()
            sample = gpu_query(config['gpu_uuid'], config['query_timeout_seconds'])
            report['gpu_admission_samples'].append({**sample, 'phase': 'before_lock'})
            write_report()
            if sample['free_bytes'] >= config['minimum_free_bytes']:
                # Recheck inventory only when lock acquisition is plausible, not every polling tick.
                if inventory(config) != bindings:
                    raise ValueError('inventory changed before lock')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = True
                except BlockingIOError:
                    pass
                if held:
                    sample = gpu_query(config['gpu_uuid'], config['query_timeout_seconds'])
                    report['gpu_admission_samples'].append({**sample, 'phase': 'under_lock'})
                    if sample['free_bytes'] < config['minimum_free_bytes']:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                        held = False
            if not held:
                if time.monotonic() - waiting_start >= config['maximum_wait_seconds']:
                    raise TimeoutError('resource admission timeout; command not launched')
                time.sleep(1)
        check_cancelled()
        if inventory(config) != bindings:
            raise ValueError('inventory changed under lock before launch')
        if config['output'].exists():
            raise FileExistsError('child output appeared during admission; refusing overwrite')
        command = [str(config['python']), str(config['monitor_script']), '--gpu-uuid', config['gpu_uuid'],
                   '--output', str(diag / 'monitor'), '--interval', '0.5', '--query-timeout',
                   str(config['query_timeout_seconds']), '--', str(config['python']), str(config['script']),
                   *config['args']]
        report.update(status='running', command=command, admission_wait_seconds=time.monotonic() - waiting_start,
                      command_started_utc=now(), command_started_monotonic=time.monotonic())
        write_report()
        stream = (diag / 'monitor_launcher.log').open('x')
        child = subprocess.Popen(command, env=environment, cwd=str(config['source']), stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        # Cancellation only sets a flag so ownership registration cannot be skipped after Popen.
        snapshot_descendants([child.pid], owned)
        report['monitor_pid'] = child.pid
        deadline = time.monotonic() + config['timeout_seconds']
        last_report = time.monotonic()
        while child.poll() is None:
            check_cancelled()
            snapshot_descendants([child.pid, *owned], owned)
            tree_times.append(time.monotonic())
            if time.monotonic() >= deadline:
                raise TimeoutError('stage command timeout; no algorithm retry')
            if time.monotonic() - last_report >= 10:
                report['active_owned'] = active_owned(owned)
                write_report()
                last_report = time.monotonic()
            time.sleep(.1)
        report.update(command_exit_code=child.returncode, command_finished_utc=now(),
                      monitored_process_wall_seconds=time.monotonic() - report['command_started_monotonic'])
        report['status'] = 'command_exited_successfully_pending_cleanup' if child.returncode == 0 else 'command_failed'
        result_code = child.returncode if child.returncode else 0
    except Interrupted as error:
        report.update(status='interrupted', error=repr(error)); result_code = 130
    except TimeoutError as error:
        report.update(status='timed_out', error=repr(error)); result_code = 124
    except BaseException as error:
        report.update(status='validation_query_or_launch_failed', error=repr(error), traceback=traceback.format_exc())
        result_code = 1
    finally:
        if cancelled:
            report['cancellation_signals'] = list(cancelled)
        cleanup_start = time.monotonic()
        if child is not None:
            snapshot_descendants([child.pid, *owned], owned)
            report['residual_owned_after_direct_exit'] = active_owned(owned)
            if (report['status'] == 'command_exited_successfully_pending_cleanup' and
                    report['residual_owned_after_direct_exit']):
                report['status'] = 'residual_owned_terminated'
                result_code = 1
            while active_owned(owned):
                try:
                    cleanup_owned_tree(child, owned, report, write_report)
                except Exception as error:
                    report['cleanup_error_while_lock_held'] = repr(error)
                    write_report(); time.sleep(.1)
                    snapshot_descendants([child.pid, *owned], owned)
            child.poll()
        if stream is not None:
            stream.close()
        report['all_tracked_owned_exited'] = not active_owned(owned)
        report['cleanup_seconds'] = time.monotonic() - cleanup_start
        if cancelled and result_code == 0:
            report.update(status='interrupted', cancellation_signals=list(cancelled)); result_code = 130
        # Cleanup success preserves failure/termination states and never upgrades them to complete.
        if report['status'] == 'command_exited_successfully_pending_cleanup':
            try:
                monitor_path = diag / 'monitor/monitor.json'
                monitor = json.loads(monitor_path.read_text())
                report['monitor_result'] = monitor
                report['monitor_report_sha256'] = digest(monitor_path)
                if monitor.get('exit_code') != 0 or not monitor.get('monitor_thread_finished'):
                    raise RuntimeError('monitor completion missing/failed despite direct wrapper exit')
                if not config['output'].is_dir():
                    raise FileNotFoundError('declared stage output missing despite zero exit')
                if inventory(config) != bindings:
                    raise ValueError('source/input/program inventory changed by final boundary')
                report['status'] = 'complete'
                result_code = 0
            except BaseException as error:
                report.update(status='completion_unverified', completion_error=repr(error)); result_code = 1
        else:
            try:
                report['inventory_unchanged_after_failure'] = inventory(config) == bindings
            except BaseException as error:
                report['final_inventory_error'] = repr(error)
        gaps = [b - a for a, b in zip(tree_times, tree_times[1:])]
        report['maximum_tree_snapshot_gap_seconds'] = max(gaps, default=None)
        report['cpu_after'] = cpu_receipt()
        report['outer_wall_seconds_through_cleanup_and_validation'] = time.monotonic() - start
        # A simultaneous child failure must not erase the user's cancellation evidence.
        if cancelled:
            report['cancellation_signals'] = list(cancelled)
            if result_code == 0:
                report['status'] = 'interrupted'
                result_code = 130
        report['exit_code'] = result_code
        report['finished_utc'] = now()
        report['scope'] = 'Stage/scalar execution and monitor completion only; no whole-case/official-equivalence claim'
        write_report()
        if held:
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return result_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    return run(args.config.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
