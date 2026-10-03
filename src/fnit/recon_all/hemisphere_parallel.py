"""双侧 exec 进程、私有文件和阶段屏障；父进程确定性发布输出。"""
from __future__ import annotations
import json
from numbers import Integral
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HEMISPHERES = ('lh', 'rh')
# 旧流程先左后右，最后保留右侧的 placement 诊断体积。
SHARED_LAST_RIGHT = {'mri/mrisps.wpa.mgz', 'mri/mrisps.white.mgz'}


class HemisphereGroupError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def validate_hemisphere_workers(workers, threads):
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers not in (1, 2):
        raise ValueError('hemisphere_workers must be 1 or 2')
    if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < workers:
        raise ValueError('threads must be an integer >= hemisphere_workers')


def _snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def _owned(relative, hemi):
    path = Path(relative)
    return (path.parts[0] in ('surf', 'label', 'stats')
            and (path.name.startswith(hemi + '.')
                 or path.name == f'autodet.gw.stats.{hemi}.dat')) or (
                     relative == f'mri/filled-pretess{255 if hemi == "lh" else 127}.mgz')


def _normalize_paths(value, private, public, published_paths=None):
    if isinstance(value, str):
        if published_paths and value in published_paths:
            return published_paths[value]
        return value.replace(str(private), str(public))
    if isinstance(value, list):
        return [_normalize_paths(v, private, public, published_paths) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_paths(v, private, public, published_paths) for k, v in value.items()}
    return value


def resolve_worker_device(device):
    """将CUDA当前设备固定成带索引名称；worker不能继承父current_device。"""
    import torch
    selected = torch.device(device)
    if selected.type == 'cuda' and selected.index is None:
        selected = torch.device('cuda', torch.cuda.current_device() if torch.cuda.is_initialized() else 0)
    return str(selected)


def inherited_allocator_policy(environ):
    """按新进程环境选实际策略，不推断已初始化父API的实际缓存状态。"""
    return 'disabled' if 'PYTORCH_NO_CUDA_MEMORY_CACHING' in environ else 'enabled'


def release_idle_parent_cuda_cache(device):
    """在新 worker 启动前释放父进程空闲缓存；保留 live tensor 与 allocator 策略。

    CPU 或尚未初始化的 CUDA 不建立 context。计数只属于父进程选定设备的
    PyTorch allocator，不包含 CUDA context、子进程或其他库的显存。
    empty_cache 可释放父进程其他设备的空闲缓存；不从全局初始化状态
    推断目标设备是否已经建立 context。
    """
    import torch
    selected = torch.device(device)
    if selected.type != 'cuda' or not torch.cuda.is_initialized():
        return {'status': 'not_applicable', 'cuda_context_created': False}
    tick = time.perf_counter()
    torch.cuda.synchronize(selected)
    before = {'allocated_bytes': int(torch.cuda.memory_allocated(selected)),
              'reserved_bytes': int(torch.cuda.memory_reserved(selected))}
    torch.cuda.empty_cache()
    after = {'allocated_bytes': int(torch.cuda.memory_allocated(selected)),
             'reserved_bytes': int(torch.cuda.memory_reserved(selected))}
    return {'status': 'complete', 'device': str(selected), 'seconds': time.perf_counter() - tick,
            'before': before, 'after': after, 'cuda_context_created': None,
            'cuda_context_measurement': 'not measured; global initialization does not prove target-device context exists',
            'scope': 'parent PyTorch allocator counters; CUDA contexts, workers and external allocators excluded',
            'method': 'synchronize selected device then empty_cache on parent allocator; live tensors and allocator policy unchanged'}


def _live_group(group):
    # leader已退出仍可能有原生孙进程；zombie不作为仍运行计算。
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            if path.stat().st_uid != os.getuid():
                continue
            fields = path.read_text().rsplit(') ', 1)[1].split()
            if int(fields[2]) == group and int(fields[3]) == group and fields[0] not in ('Z', 'X'):
                return True
        except (OSError, ValueError, IndexError):
            pass
    return False


def _cancel(processes):
    """只终止本调度new-session创建的组，包括已退出leader的原生孙进程。"""
    groups = [process.pid for process in processes]
    for group in groups:
        if _live_group(group):
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and any(_live_group(group) for group in groups):
        time.sleep(.05)
    for group in groups:
        if _live_group(group):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        process.wait()


def run_hemisphere_group(subject, operation, *, device, threads, workers=2,
                         profile_stages=False, callable_path='fnit.recon_all.native_free:_hemisphere_operation',
                         kwargs=None):
    """在私有完整拷贝中执行双侧任务，成功屏障后逐文件原子发布。

    subject 是自产被试目录；operation 标识阶段；kwargs 为 JSON 可序列化
    公共参数。threads 是总预算，workers=1/2 时分别分配全部/一半预算
    （奇数向下取整）。device 是显式 CPU/CUDA 设备；不 fork CUDA 状态。
    返回 values[lh/rh]、独立 workers 报告、组墙钟、重叠及同期显存记录。
    启动 worker 前同步 CUDA 并释放父进程空闲缓存，不改变 live tensor 或缓存策略。
    发布失败不会生成成功状态；失败子树终止，保留 worker 日志和失败报告。
    不发布缺陷体积：父调用者必须按 lh、rh 顺序串行累计。
    """
    from .profiling import ProcessTreeDeviceSampler, parallel_intervals
    from .thread_budget import native_thread_environment
    import torch
    validate_hemisphere_workers(workers, threads)
    caller_device = str(device)
    device = resolve_worker_device(device)
    subject = Path(subject).resolve()
    group_tick = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix=f'hemi-{operation}-', dir=subject / 'scripts'))
    report = {'operation': operation, 'workers_requested': workers,
              'total_thread_budget': threads, 'worker_threads': threads // workers,
              'status': 'running', 'workers': {}, 'values': {}, 'published': [],
              'caller_device': caller_device, 'worker_device': device,
              'worker_allocator_selection': {'basis': 'environment at fresh exec',
                  'selected_policy': inherited_allocator_policy(os.environ),
                  'parent_preinitialized_actual_state': 'not_inferred_from_environment'}}
    precision = {'matmul_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                 'cudnn_tf32': bool(torch.backends.cudnn.allow_tf32)}
    processes, streams, private_roots, snapshots, pending = [], [], {}, {}, []
    sampler = ProcessTreeDeviceSampler(device=device, parent_pid=os.getpid())
    failure = None
    try:
        preparation_tick = time.monotonic()
        for hemi in HEMISPHERES:
            private = root / hemi / subject.name
            private.mkdir(parents=True)
            for folder in ('mri', 'surf', 'label', 'stats'):
                source = subject / folder
                if source.exists():
                    # 不使用 hardlink/symlink：原生 in-place 写入不能污染父目录。
                    shutil.copytree(source, private / folder)
                else:
                    (private / folder).mkdir()
            (private / 'scripts').mkdir()
            private_roots[hemi] = private
            snapshots[hemi] = _snapshot(private)
        report['private_copy_seconds'] = time.monotonic() - preparation_tick
        # Previously used volume/network buffers should not overlap the fresh
        # hemisphere CUDA contexts when they are only held as idle cache.
        report['parent_idle_cuda_cache'] = release_idle_parent_cuda_cache(device)
        for offset in range(0, 2, workers):
            active = []
            for hemi in HEMISPHERES[offset:offset + workers]:
                private = private_roots[hemi]
                request_path, report_path = root / f'{hemi}.request.json', root / f'{hemi}.report.json'
                child_kwargs = dict(kwargs or {}, subject=str(private), hemi=hemi,
                                    device=device, threads=threads // workers,
                                    operation=operation)
                request = {'callable': callable_path, 'operation': operation,
                           'kwargs': child_kwargs, 'device': device,
                           'threads': threads // workers, 'precision': precision,
                           'profile_stages': profile_stages,
                           'allocator_policy': inherited_allocator_policy(os.environ)}
                request_path.write_text(json.dumps(request))
                env, environment_report = native_thread_environment(threads=threads // workers)
                # API 调用者可从 sys.path 导入候选源码；worker 必须导入同一包。
                source_root = str(Path(__file__).resolve().parents[2])
                env['PYTHONPATH'] = source_root + os.pathsep + env.get('PYTHONPATH', '')
                log_path = subject / 'scripts' / f'{operation}.{hemi}.worker.log'
                stream = log_path.open('w')
                streams.append(stream)
                process = subprocess.Popen([sys.executable, '-m',
                    'fnit.recon_all.hemisphere_worker', str(request_path), str(report_path)],
                    env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                sampler.add_worker(process.pid)
                active.append((hemi, process, report_path, environment_report))
            while any(p.poll() is None for _, p, _, _ in active):
                sampler.sample_if_due()
                if any(p.poll() not in (None, 0) for _, p, _, _ in active):
                    raise RuntimeError('hemisphere worker failed; sibling cancelled')
                time.sleep(.05)
            for hemi, process, report_path, environment_report in active:
                if report_path.is_file():
                    child_report = json.loads(report_path.read_text())
                    child_report['environment'] = environment_report
                    report['workers'][hemi] = child_report
                if process.returncode or not report_path.is_file() or child_report['status'] != 'complete':
                    raise RuntimeError(f'{hemi} worker failed or produced no complete report')
                report['values'][hemi] = child_report['value']
        report.update(parallel_intervals(report['workers']))
        # 先审计两侧全部改动，再发布任何文件。
        for hemi in HEMISPHERES:
            private = private_roots[hemi]
            after = _snapshot(private)
            if set(snapshots[hemi]) - set(after):
                raise RuntimeError(f'{hemi} worker deleted input files')
            for relative, version in sorted(after.items()):
                if snapshots[hemi].get(relative) == version:
                    continue
                if _owned(relative, hemi):
                    pending.append((private / relative, subject / relative))
                elif relative in SHARED_LAST_RIGHT:
                    if hemi == 'rh':
                        pending.append((private / relative, subject / relative))
                elif relative.startswith('scripts/'):
                    name = Path(relative).name
                    pending.append((private / relative, subject / 'scripts' / f'{operation}.{hemi}.{name}'))
                else:
                    raise RuntimeError(f'undeclared shared write by {hemi}: {relative}')
        publish_tick = time.monotonic()
        for source, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.fnit-publish-tmp')
            shutil.copyfile(source, temporary)
            temporary.replace(target)
            report['published'].append(str(target.relative_to(subject)))
        report['publish_seconds'] = time.monotonic() - publish_tick
        published_paths = {str(source): str(target) for source, target in pending}
        for hemi in HEMISPHERES:
            normalized = _normalize_paths(report['values'][hemi], private_roots[hemi], subject, published_paths)
            report['values'][hemi] = normalized
            report['workers'][hemi]['private_subject'] = str(private_roots[hemi])
            report['workers'][hemi]['value'] = normalized
        report['status'] = 'complete'
    except BaseException as error:
        _cancel(processes)
        # 失败 worker 报告也可诊断，不能丢失失败阶段。
        for hemi in HEMISPHERES:
            path = root / f'{hemi}.report.json'
            if path.is_file():
                report['workers'][hemi] = json.loads(path.read_text())
        report.update(status='failed', error=repr(error))
        failure = HemisphereGroupError(f'{operation} hemisphere group failed: {error}', report)
        raise failure from error
    finally:
        final_errors = []
        try:
            _cancel(processes)
        except BaseException as error:
            final_errors.append(('process_reaping', error))
        for stream in streams:
            try:
                stream.close()
            except BaseException as error:
                final_errors.append(('worker_log_close', error))
        try:
            sampler.sample_if_due(force=True)
            report['device_process_tree'] = sampler.report()
        except BaseException as error:
            final_errors.append(('device_sampling_finalization', error))
        # 私有影像不进入交付；清理失败也属于当前组失败，不保留complete。
        cleanup_tick = time.monotonic()
        try:
            shutil.rmtree(root)
        except BaseException as error:
            final_errors.append(('private_directory_cleanup', error))
            report['private_directory_retained'] = str(root)
        report['cleanup_seconds'] = time.monotonic() - cleanup_tick
        report['group_wall_seconds'] = time.monotonic() - group_tick
        if final_errors:
            report['finalization_errors'] = [{'phase': phase, 'error': repr(error)}
                                             for phase, error in final_errors]
            report['status'] = 'failed'
            report.setdefault('error', repr(final_errors[0][1]))
            report['failed_finalization_phase'] = final_errors[0][0]
        path = subject / 'scripts' / f'{operation}.hemisphere-group.json'
        try:
            # 先写临时元数据再替换，避免留下截断的成功JSON。
            temporary_report = path.with_suffix('.json.tmp')
            temporary_report.write_text(json.dumps(report, indent=2))
            temporary_report.replace(path)
        except BaseException as error:
            try:
                temporary_report.unlink(missing_ok=True)
            except OSError as temporary_error:
                error.add_note(f'Incomplete metadata temporary cleanup failed: {temporary_error!r}')
            final_errors.append(('group_report_publication', error))
            report.update(status='failed', failed_finalization_phase='group_report_publication')
            report.setdefault('error', repr(error))
            report.setdefault('finalization_errors', []).append(
                {'phase': 'group_report_publication', 'error': repr(error)})
        if final_errors:
            message = '; '.join(f'{phase}: {error!r}' for phase, error in final_errors)
            if failure is not None:
                failure.add_note('Hemisphere group finalization also failed: ' + message)
            else:
                raise HemisphereGroupError(f'{operation} hemisphere finalization failed: {message}', report) from final_errors[0][1]
    return report
