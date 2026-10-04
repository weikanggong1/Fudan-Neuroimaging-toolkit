"""冻结整例重跑的资源准入包装器；标准库，无生产算法修改。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

GPU_UUID = 'GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e'
MINIMUM_FREE_BYTES = 20_000_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def admitted(sample):
    return sample['gpu_uuid'] == GPU_UUID and sample['free_bytes'] >= MINIMUM_FREE_BYTES


def query_gpu(timeout=5):
    result = subprocess.run(['nvidia-smi', '--id=' + GPU_UUID,
                             '--query-gpu=uuid,memory.total,memory.used,memory.free',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True, timeout=timeout)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError('expected exactly one GPU')
    uuid, total, used, free = [part.strip() for part in lines[0].split(',')]
    sample = dict(time_utc=now(), gpu_uuid=uuid, total_bytes=int(total)*1048576,
                  used_bytes=int(used)*1048576, free_bytes=int(free)*1048576,
                  measurement='nvidia-smi integer MiB converted to bytes')
    if uuid != GPU_UUID or min(sample[k] for k in ('total_bytes','used_bytes','free_bytes')) < 0:
        raise ValueError('wrong GPU or invalid memory query')
    return sample


def retry_config(original, retry_root):
    """仅修改两项输出路径；原始文件保持不变，目标必须尚不存在。"""
    if original['gpu_uuid'] != GPU_UUID:
        raise ValueError('original GPU UUID differs from fixed protocol')
    if original['invocation'] not in ('cli', 'initialized_cuda_api'):
        raise ValueError('unsupported original invocation')
    root = Path(retry_root).resolve()
    if root.exists():
        raise FileExistsError('retry root must not exist')
    for key in ('output', 'diagnostic_root', 'code_root', 'weights', 'assets', 'native_bin_dir'):
        if key not in original:
            continue
        old = Path(original[key]).resolve()
        if root == old or old in root.parents or root in old.parents:
            raise ValueError('retry root overlaps original output')
    return {**original, 'output': str(root/'subject'), 'diagnostic_root': str(root/'diagnostics')}


def inventory(config):
    """记录实际输入、冻结源码、权重、模板及独立程序；不读取许可证。"""
    paths = {str(Path(config['input']).resolve())}
    for key in ('code_root', 'weights', 'assets', 'native_bin_dir'):
        root = Path(config[key])
        if not root.is_dir():
            raise FileNotFoundError(key)
        paths.update(str(p.resolve()) for p in root.rglob('*') if p.is_file()
                     and '.git' not in p.parts and '__pycache__' not in p.parts
                     and p.suffix not in ('.pyc', '.nbc', '.nbi')
                     and 'license' not in p.name.lower())
    if 'benchmark_tools' in config:
        paths.update(str(p.resolve()) for p in benchmark_tool_paths(config).values())
    return {path: digest(path) for path in sorted(paths)}


def benchmark_tool_paths(config):
    """绑定源码树外的验证驱动；缺省保留历史树内路径，不修改生产源树。"""
    source = Path(config['code_root'])
    if 'benchmark_tools' not in config:
        return {'monitor': source/'validation/recon_all/python_gpu_port/run_monitored.py',
                'whole_case_driver': source/'validation/recon_all/optimizations/20261002_parallel/execute_whole_case.py'}
    rows = config['benchmark_tools']
    if not isinstance(rows, dict) or set(rows) != {'monitor', 'whole_case_driver'}:
        raise ValueError('benchmark_tools requires exactly monitor and whole_case_driver')
    paths = {}
    for role, row in rows.items():
        if not isinstance(row, dict) or set(row) != {'path', 'sha256'}:
            raise ValueError('benchmark tool requires explicit path and sha256: ' + role)
        path = Path(row['path'])
        if not path.is_absolute() or path.suffix != '.py' or not path.is_file():
            raise ValueError('benchmark tool must be an existing absolute .py file: ' + role)
        if not isinstance(row['sha256'], str) or re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is None:
            raise ValueError('benchmark tool requires frozen SHA256: ' + role)
        if digest(path) != row['sha256']:
            raise ValueError('benchmark tool SHA changed: ' + role)
        paths[role] = path.resolve()
    return paths


def benchmark_command(config, retry_root):
    """返回固定设备的字面 argv；只选择监控工具，影像参数仍由配置绑定。"""
    tools = benchmark_tool_paths(config)
    return [config['python'], str(tools['monitor']), '--gpu-uuid', GPU_UUID,
            '--output', str(Path(retry_root)/'monitor'), '--', config['python'],
            str(tools['whole_case_driver']), '--config', str((Path(retry_root)/'retry_config.json').resolve())]


class Interrupted(Exception):
    pass


def process_identity(pid):
    """只读取/proc/stat身份，不读取argv、环境；僵尸不算活跃计算。"""
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], pgid=int(fields[2]),
                    starttime=int(fields[19]))
    except (OSError, ValueError, IndexError):
        return None


def snapshot_descendants(roots, owned):
    pending=list(roots)
    visited=set()
    while pending:
        pid=pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        identity=process_identity(pid)
        if identity is None:
            continue
        existing=owned.get(pid)
        if existing is not None and existing['starttime'] != identity['starttime']:
            continue  # PID was reused: never follow or signal its new owner.
        owned[pid]=identity
        try:
            for task in (Path('/proc')/str(pid)/'task').iterdir():
                try:
                    pending.extend(int(p) for p in (task/'children').read_text().split())
                except (OSError, ValueError):
                    pass
        except OSError:
            pass


def active_owned(owned):
    result=[]
    for original in owned.values():
        current=process_identity(original['pid'])
        if current and current['starttime']==original['starttime'] and current['state'] not in ('Z','X'):
            result.append(current)
    return result


def signal_owned(identity, signum):
    """pidfd绑定具体进程，避免身份检查之后PID重用误杀。"""
    if not callable(getattr(os,'pidfd_open',None)) or not callable(getattr(signal,'pidfd_send_signal',None)):
        current=process_identity(identity['pid'])
        if current and current['starttime']==identity['starttime'] and current['pgid']==identity['pgid'] and current['state'] not in ('Z','X'):
            try:
                os.kill(identity['pid'],signum)
            except ProcessLookupError:
                pass
        return
    try:
        descriptor=os.pidfd_open(identity['pid'])
    except ProcessLookupError:
        return
    try:
        current=process_identity(identity['pid'])
        if current and current['starttime']==identity['starttime'] and current['pgid']==identity['pgid'] and current['state'] not in ('Z','X'):
            signal.pidfd_send_signal(descriptor,signum)
    except ProcessLookupError:
        pass
    finally:
        os.close(descriptor)


def cleanup_owned_tree(child, owned, report, write_report, term_seconds=10):
    """停止本次后代跨session计算；没有活跃后代时才允许调用者退锁。"""
    # Prevent repeated interruption from bypassing cleanup and releasing the lock.
    handlers={s:signal.signal(s,signal.SIG_IGN) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        snapshot_descendants([child.pid, *owned],owned)
        # Stop ancestors before collecting children again, closing ordinary fork races.
        for identity in active_owned(owned):
            signal_owned(identity,signal.SIGSTOP)
        while True:
            old=set(owned)
            snapshot_descendants(list(owned),owned)
            for pid in set(owned)-old:
                signal_owned(owned[pid],signal.SIGSTOP)
            if set(owned)==old:
                break
        report['cleanup_snapshot']=list(owned.values())
        report['cleanup_status']='term_waiting_lock_held'
        write_report()
        for identity in active_owned(owned):
            signal_owned(identity,signal.SIGTERM)
            signal_owned(identity,signal.SIGCONT)
        deadline=time.monotonic()+term_seconds
        while active_owned(owned) and time.monotonic()<deadline:
            snapshot_descendants(list(owned),owned)
            time.sleep(.05)
        survivors=active_owned(owned)
        report['cleanup_kill_required']=bool(survivors)
        # KILL is bounded after TERM. Uninterruptible D-state may persist: retain
        # the benchmark lock and a clear receipt until all active owners disappear.
        while survivors:
            snapshot_descendants(list(owned),owned)
            for identity in active_owned(owned):
                signal_owned(identity,signal.SIGKILL)
            report['cleanup_status']='kill_waiting_lock_held'
            report['cleanup_active']=active_owned(owned)
            write_report()
            time.sleep(.05)
            survivors=active_owned(owned)
        child.poll()  # Reap direct child if available; orphan zombies are not computing.
        report.update(cleanup_status='all_owned_computation_exited',cleanup_active=[])
        write_report()
    finally:
        for signum,handler in handlers.items():
            signal.signal(signum,handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--retry-root', type=Path)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--query-timeout', type=float, default=5)
    parser.add_argument('--maximum-wait-seconds', type=float, default=3600)
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 60 or not 0 < args.query_timeout <= 60 or args.maximum_wait_seconds < 0:
        parser.error('poll/query intervals must be >0 and <=60; wait must be nonnegative')
    original = None
    if not args.preflight_only:
        if not all((args.config, args.retry_root, args.lock)):
            parser.error('execution requires config, retry-root and shared lock')
        original = json.loads(args.config.read_text())
        target = args.report.resolve()
        for key in ('output', 'diagnostic_root', 'code_root', 'weights', 'assets', 'native_bin_dir'):
            if key in original:
                protected = Path(original[key]).resolve()
                if target == protected or protected in target.parents:
                    parser.error('report cannot be inside original outputs or resources')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive report creation prevents overwriting an earlier receipt.
    with args.report.open('x') as stream:
        stream.write('{}\n')
    report = dict(status='starting', started_utc=now(), gpu_uuid=GPU_UUID,
                  minimum_free_bytes=MINIMUM_FREE_BYTES, samples=[],
                  timing_scope='admission waiting excluded from monitored algorithm time',
                  limitation='preflight and cooperative flock cannot exclude later external competitors')
    lock = None
    child = None
    owned = {}
    start = time.monotonic()
    def interrupted(signum, frame):
        raise Interrupted('signal ' + str(signum))
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.preflight_only:
            sample = query_gpu(args.query_timeout)
            report['samples'].append(dict(sample, phase='read_only'))
            report['status'] = 'preflight_admitted' if admitted(sample) else 'preflight_insufficient'
            return 0 if admitted(sample) else 75
        if not all((args.config, args.retry_root, args.lock)):
            raise ValueError('execution requires config, retry-root and shared lock')
        config = retry_config(original, args.retry_root)
        if digest(original['input']) != original['input_sha256']:
            raise ValueError('original input SHA mismatch')
        launch = json.loads((Path(original['diagnostic_root'])/'launch.json').read_text())
        for key, value in original.items():
            if launch.get(key) != value:
                raise ValueError('original config/launch binding differs: ' + key)
        if digest(Path(original['code_root'])/'src/fnit/recon_all/native_free.py') != launch['candidate_native_free_sha256']:
            raise ValueError('frozen program SHA differs from original launch')
        bindings = inventory(original)
        report.update(original_config=str(args.config.resolve()), original_config_sha256=digest(args.config),
                      original_launch_sha256=digest(Path(original['diagnostic_root'])/'launch.json'),
                      config=config, resource_sha256=bindings)
        args.retry_root.mkdir(parents=True, exist_ok=False)
        config_path=args.retry_root/'retry_config.json'
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        lock = args.lock.open('a+')
        while True:
            sample = query_gpu(args.query_timeout)
            report['samples'].append(dict(sample, phase='before_lock'))
            if admitted(sample):
                if inventory(original) != bindings:
                    raise ValueError('resource SHA changed during admission waiting')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    report['samples'].append(dict(time_utc=now(), phase='lock_busy'))
                else:
                    sample = query_gpu(args.query_timeout)
                    report['samples'].append(dict(sample, phase='under_lock'))
                    if admitted(sample):
                        break
                    fcntl.flock(lock, fcntl.LOCK_UN)
            if time.monotonic()-start >= args.maximum_wait_seconds:
                report['status']='wait_timeout'
                return 75
            report['status']='waiting'
            args.report.write_text(json.dumps(report, indent=2)+'\n')
            time.sleep(min(args.poll_seconds, max(0, args.maximum_wait_seconds-(time.monotonic()-start))))
        command=benchmark_command(config,args.retry_root)
        report.update(status='running', admission_wait_seconds=time.monotonic()-start,
                      algorithm_started_utc=now(), command=command)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        # Own process group permits interruption without touching unrelated GPU tasks.
        child=subprocess.Popen(command,start_new_session=True)
        snapshot_descendants([child.pid],owned)
        while child.poll() is None:
            snapshot_descendants([child.pid,*owned],owned)
            try:
                child.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        code=child.returncode
        report.update(status='complete' if code==0 else 'child_failed', exit_code=code)
        return code
    except Interrupted as error:
        report.update(status='interrupted', error=str(error))
        return 130
    except Exception as error:
        report.update(status='query_or_validation_failed',error=repr(error))
        return 1
    finally:
        if child is not None:
            snapshot_descendants([child.pid,*owned],owned)
            while active_owned(owned):
                try:
                    cleanup_owned_tree(child,owned,report,
                                       lambda:args.report.write_text(json.dumps(report,indent=2)+'\n'))
                except BaseException as error:
                    # Fail closed: a cleanup problem must never release the lock.
                    for signum in (signal.SIGINT,signal.SIGTERM):
                        signal.signal(signum,signal.SIG_IGN)
                    report.update(cleanup_status='cleanup_error_lock_held',cleanup_error=repr(error))
                    args.report.write_text(json.dumps(report,indent=2)+'\n')
                    time.sleep(1)
        if lock is not None:
            fcntl.flock(lock,fcntl.LOCK_UN)
            lock.close()
        report.update(finished_utc=now(), wrapper_seconds=time.monotonic()-start)
        if 'admission_wait_seconds' not in report:
            report['admission_wait_seconds']=time.monotonic()-start
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        for s, handler in previous.items():
            signal.signal(s,handler)


if __name__=='__main__':
    raise SystemExit(main())
