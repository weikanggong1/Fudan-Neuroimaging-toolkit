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


def _normalize_paths(value, private, public):
    if isinstance(value, str):
        return value.replace(str(private), str(public))
    if isinstance(value, list):
        return [_normalize_paths(v, private, public) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_paths(v, private, public) for k, v in value.items()}
    return value


def _cancel(processes):
    """终止整棵 worker 原生子树，再回收；不影响外部进程。"""
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 3
    for process in processes:
        try:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_hemisphere_group(subject, operation, *, device, threads, workers=2,
                         profile_stages=False, callable_path='fnit.recon_all.native_free:_hemisphere_operation',
                         kwargs=None):
    """在私有完整拷贝中执行双侧任务，成功屏障后逐文件原子发布。

    subject 是自产被试目录；operation 标识阶段；kwargs 为 JSON 可序列化
    公共参数。threads 是总预算，workers=1/2 时分别分配全部/一半预算
    （奇数向下取整）。device 是显式 CPU/CUDA 设备；不 fork CUDA 状态。
    返回 values[lh/rh]、独立 workers 报告、组墙钟、重叠及同期显存记录。
    发布失败不会生成成功状态；失败子树终止，保留 worker 日志和失败报告。
    不发布缺陷体积：父调用者必须按 lh、rh 顺序串行累计。
    """
    from .profiling import ProcessTreeDeviceSampler, parallel_intervals
    from .thread_budget import native_thread_environment
    import torch
    validate_hemisphere_workers(workers, threads)
    subject = Path(subject).resolve()
    group_tick = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix=f'hemi-{operation}-', dir=subject / 'scripts'))
    report = {'operation': operation, 'workers_requested': workers,
              'total_thread_budget': threads, 'worker_threads': threads // workers,
              'status': 'running', 'workers': {}, 'values': {}, 'published': []}
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
                           'profile_stages': profile_stages}
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
                report['values'][hemi] = _normalize_paths(child_report['value'], private_roots[hemi], subject)
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
