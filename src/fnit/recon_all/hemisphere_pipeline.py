"""一次私有拷贝/一次 exec 的半球连续链，父进程逐阶段审计和发布。"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .hemisphere_parallel import (
    HEMISPHERES, SHARED_LAST_RIGHT, HemisphereGroupError, _snapshot, _owned,
    _normalize_paths, _cancel, resolve_worker_device, release_idle_parent_cuda_cache,
    validate_hemisphere_workers, worker_cuda_cache_environment,
)
from .hemisphere_pipeline_worker import atomic_json


def _operations(operations, callable_path, kwargs):
    if not isinstance(operations, (list, tuple)) or not operations:
        raise ValueError('operations must be a nonempty ordered list')
    result = []
    reserved = {'subject', 'hemi', 'device', 'threads', 'operation'}
    for specification in operations:
        item = {'operation': specification} if isinstance(specification, str) else dict(specification)
        if set(item) - {'operation', 'callable_path', 'kwargs'}:
            raise ValueError('unknown operation specification keys')
        name = item.get('operation')
        if (not isinstance(name, str) or not name or
                any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for char in name)):
            raise ValueError('operation requires a safe nonempty name')
        function = item.get('callable_path', callable_path)
        if not isinstance(function, str) or ':' not in function:
            raise ValueError('callable_path requires module:attribute')
        arguments = dict(kwargs or {})
        arguments.update(item.get('kwargs', {}))
        if reserved & set(arguments):
            raise ValueError('kwargs cannot override worker identity or budget')
        result.append({'operation': name, 'callable_path': function, 'kwargs': arguments})
    if len({row['operation'] for row in result}) != len(result):
        raise ValueError('operation names must be unique within a pipeline')
    # 在副作用前证明所有 callable 参数可序列化；不强行转换数组/路径对象。
    json.dumps(result, allow_nan=False)
    return result


def _audit_pending(private, before, subject, hemi, operation):
    for path in private.rglob('*'):
        if path.is_symlink():
            raise RuntimeError(f'{hemi} worker created a symlink: {path.relative_to(private)}')
    after = _snapshot(private)
    if set(before) - set(after):
        raise RuntimeError(f'{hemi} worker deleted input files')
    pending = []
    for relative, version in sorted(after.items()):
        if before.get(relative) == version:
            continue
        if _owned(relative, hemi):
            target = subject / relative
        elif relative in SHARED_LAST_RIGHT:
            if hemi != 'rh':
                continue
            target = subject / relative
        elif relative.startswith('scripts/'):
            path = Path(relative)
            if path.parts[1] in {'lh.topology.genetic', 'rh.topology.genetic'}:
                raise RuntimeError(f'undeclared genetic topology write by {hemi}: {relative}')
            # 保留相对层级，避免 scripts/a/report.json 与 b/report.json 覆盖。
            target = subject / 'scripts' / f'{operation}.{hemi}' / path.relative_to('scripts')
        else:
            raise RuntimeError(f'undeclared shared write by {hemi}: {relative}')
        pending.append((private / relative, target))
    return after, pending


def _publish(pending, subject, public_versions):
    prepared = []
    try:
        for source, target in pending:
            relative = str(target.relative_to(subject))
            current = (target.stat().st_size, target.stat().st_mtime_ns) if target.is_file() else None
            if current != public_versions.get(relative):
                raise RuntimeError('public output changed outside pipeline: ' + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.fnit-pipeline-pending')
            shutil.copyfile(source, temporary)
            prepared.append((temporary, target, relative))
        for temporary, target, relative in prepared:
            temporary.replace(target)
            public_versions[relative] = (target.stat().st_size, target.stat().st_mtime_ns)
    finally:
        for temporary, _, _ in prepared:
            temporary.unlink(missing_ok=True)
    return [relative for _, _, relative in prepared]


def _data_snapshot(subject):
    return {f'{folder}/{relative}': version
            for folder in ('mri', 'surf', 'label', 'stats')
            for relative, version in _snapshot(subject / folder).items()}


def _refresh_inputs(updates, private, subject):
    """只接受回调显式声明的自产公共文件，并独立复制；不共享可写 inode。"""
    if updates is None:
        return []
    if not isinstance(updates, dict):
        raise ValueError('stage_hook must return None or {relative_input: public_file_path}')
    refreshed = []
    for relative, source in sorted(updates.items()):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] not in ('mri', 'surf', 'label', 'stats'):
            raise ValueError('stage_hook refresh requires subject-relative data file')
        source = Path(source).resolve(strict=True)
        source.relative_to(subject)
        target = private / path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.fnit-refresh-pending')
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        refreshed.append(str(path))
    return refreshed


def run_hemisphere_pipeline(subject, operations, *, device, threads,
        callable_path='fnit.recon_all.native_free:_hemisphere_operation', kwargs=None,
        stage_hook=None, profile_stages=False, cuda_allocator_cache='inherit',
        startup_wait_seconds=90, trusted_host_pid_map=None):
    """持久双半球调度；全部参数、阶段回调及结果见 PERSISTENT_HEMISPHERE_PIPELINE.md。

    每侧一个完整独立copy和fresh exec；逐阶段暂停后审计私有写入、检测公共
    文件外部修改、原子发布，再在父进程调用stage_hook(operation,hemi,value,row)。
    surface发布/回调强制lh→rh，其余独立推进，终态metrics留给调用者串行。
    回调可返回显式公共自产输入刷新映射；不可把参考结果接入候选。失败保留
    已完成阶段回执，取消本调度子树，未通过阶段不发布，不标整链完成。
    不改变旧run_hemisphere_group；不fork CUDA、不用可写hardlink、不在线程
    修改精度。显存归属未知保留null；trusted_host_pid_map仅接收调用方可信映射。
    """
    from .profiling import ProcessTreeDeviceSampler
    from .thread_budget import native_thread_environment, thread_budget
    import torch
    validate_hemisphere_workers(2, threads)
    specifications = _operations(operations, callable_path, kwargs)
    if stage_hook is not None and not callable(stage_hook):
        raise ValueError('stage_hook must be callable or None')
    if (isinstance(startup_wait_seconds, bool) or not isinstance(startup_wait_seconds, (int, float))
            or not math.isfinite(startup_wait_seconds) or startup_wait_seconds <= 0):
        raise ValueError('startup_wait_seconds must be finite and > 0')
    _, cache = worker_cuda_cache_environment(os.environ, cuda_allocator_cache=cuda_allocator_cache)
    subject = Path(subject).resolve(strict=True)
    if not (subject / 'scripts').is_dir():
        raise FileNotFoundError('subject/scripts must already exist')
    device = resolve_worker_device(device)
    started = time.monotonic()
    # 先记录公开文件版本；工作区在scripts下，但不进入公开版本快照。
    public_versions = _snapshot(subject)
    root = Path(tempfile.mkdtemp(prefix='hemi-pipeline-', dir=subject / 'scripts'))
    report = {'operation': 'pipeline', 'operations': [s['operation'] for s in specifications],
              'status': 'running', 'total_thread_budget': threads, 'worker_threads': threads // 2,
              'workers_requested': 2, 'workers': {}, 'stages': {s['operation']: {} for s in specifications},
              'values': {s['operation']: {} for s in specifications}, 'published': [],
              'publication_events': [], 'worker_device': device, 'private_copy_count': 0,
              'worker_allocator_selection': cache, 'metrics_scope': 'caller serial after successful pipeline',
              'request_path_scope': 'historical private paths; workspace removed after pipeline'}
    processes, streams, records, snapshots = [], [], {}, {}
    failure = None
    sampler = None
    precision = {'matmul_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                 'cudnn_tf32': bool(torch.backends.cudnn.allow_tf32)}
    try:
        preparation_started = time.monotonic()
        for hemi in HEMISPHERES:
            private = root / hemi / subject.name
            private.mkdir(parents=True)
            for folder in ('mri', 'surf', 'label', 'stats'):
                source = subject / folder
                if source.exists():
                    shutil.copytree(source, private / folder)
                else:
                    (private / folder).mkdir()
            (private / 'scripts').mkdir()
            snapshots[hemi] = _snapshot(private)
            records[hemi] = {'private': private, 'next': 0}
            report['private_copy_count'] += 1
        report['private_copy_seconds'] = time.monotonic() - preparation_started
        report['parent_idle_cuda_cache'] = release_idle_parent_cuda_cache(device)
        sampler = ProcessTreeDeviceSampler(device=device, parent_pid=os.getpid(),
                                           trusted_host_pid_map=trusted_host_pid_map)
        deadline = time.monotonic() + startup_wait_seconds
        for hemi in HEMISPHERES:
            record = records[hemi]
            prefix = subject / 'scripts' / f'pipeline.{hemi}'
            ready, go = root / f'{hemi}.ready.json', root / f'{hemi}.go.json'
            request_path, worker_report = Path(str(prefix)+'.request.json'), Path(str(prefix)+'.report.json')
            child_environment, budget = native_thread_environment(threads=threads // 2)
            child_environment, selected_cache = worker_cuda_cache_environment(
                child_environment, cuda_allocator_cache=cuda_allocator_cache)
            source_root = str(Path(__file__).resolve().parents[2])
            child_environment['PYTHONPATH'] = source_root + os.pathsep + child_environment.get('PYTHONPATH', '')
            child_environment['FNIT_RECON_FORBID_ORIGINAL_STAGES'] = '1'
            request = {'callable':'fnit.recon_all.hemisphere_pipeline_worker:execute', 'operation':'pipeline',
                'device':device, 'threads':threads // 2, 'precision':precision,
                'profile_stages':profile_stages, 'allocator_policy':selected_cache['selected_policy'],
                'ready_path':str(ready), 'go_path':str(go), 'parent_pid':os.getpid(),
                'startup_deadline_monotonic':deadline,
                'kwargs':{'subject':str(record['private']), 'hemi':hemi, 'device':device,
                    'threads':threads // 2, 'operation':'pipeline', 'operations':specifications,
                    'control_directory':str(root), 'parent_pid':os.getpid(),
                    'precision':precision, 'profile_stages':profile_stages}}
            atomic_json(request_path, request)
            log = Path(str(prefix)+'.worker.log').open('w')
            streams.append(log)
            process = subprocess.Popen([sys.executable,'-m','fnit.recon_all.hemisphere_worker',
                str(request_path),str(worker_report)], env=child_environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            sampler.add_worker(process.pid)
            record.update(process=process, ready=ready, go=go, report_path=worker_report, environment=budget)
            report['workers'][hemi] = {'pid':process.pid, 'status':'starting'}
        admitted = set()
        while len(admitted) < 2:
            if time.monotonic() >= deadline:
                raise TimeoutError('persistent hemisphere startup deadline exceeded')
            for hemi in HEMISPHERES:
                record = records[hemi]
                process = record['process']
                if process.poll() is not None:
                    raise RuntimeError(f'{hemi} persistent worker failed before GO')
                if hemi not in admitted and record['ready'].is_file():
                    marker = json.loads(record['ready'].read_text())
                    if marker.get('pid') != process.pid or marker.get('operation_entered') is not False:
                        raise RuntimeError('invalid persistent worker READY marker')
                    atomic_json(record['go'], {'go_monotonic':time.monotonic()})
                    admitted.add(hemi)
            sampler.sample_if_due()
            time.sleep(.02)
        while any(record['next'] < len(specifications) for record in records.values()):
            for hemi in HEMISPHERES:
                record = records[hemi]
                sequence = record['next']
                if sequence == len(specifications):
                    continue
                process = record['process']
                receipt = root / f'{hemi}.{sequence:03d}.stage.json'
                if process.poll() not in (None,0):
                    raise RuntimeError(f'{hemi} persistent worker failed; sibling cancelled')
                if not receipt.is_file():
                    if process.poll() is not None:
                        raise RuntimeError(f'{hemi} worker exited without stage receipt')
                    continue
                row = json.loads(receipt.read_text())
                operation = specifications[sequence]['operation']
                if (row.get('pid') != process.pid or row.get('sequence') != sequence or
                        row.get('hemi') != hemi or row.get('operation') != operation):
                    raise RuntimeError('persistent stage receipt identity mismatch')
                report['stages'][operation][hemi] = row
                if row.get('status') != 'complete':
                    raise RuntimeError(f'{hemi} {operation} failed; stage not published')
                if operation == 'surface' and hemi == 'rh' and 'lh' not in report['values'][operation]:
                    continue
                audit_started = time.monotonic()
                after, pending = _audit_pending(record['private'], snapshots[hemi], subject, hemi, operation)
                row['audit_seconds'] = time.monotonic() - audit_started
                publish_started = time.monotonic()
                published = _publish(pending, subject, public_versions)
                row['publish_seconds'] = time.monotonic() - publish_started
                row['published'] = published
                paths = {str(source):str(target) for source,target in pending}
                value = _normalize_paths(row['value'],record['private'],subject,paths)
                report['values'][operation][hemi] = value
                row['value'] = value
                report['published'].extend(published)
                report['publication_events'].append({'operation':operation,'hemi':hemi,
                    'sequence':sequence,'published_monotonic':time.monotonic(),'files':published})
                if stage_hook:
                    hook_started = time.monotonic()
                    # 当前半球等待ACK，其两线程预算交给父回调，另一侧继续两线程。
                    with thread_budget(threads=threads // 2) as hook_budget:
                        updates = stage_hook(operation,hemi,value,row)
                    row['hook_thread_budget'] = hook_budget
                    row['hook_seconds'] = time.monotonic() - hook_started
                    row['refreshed_inputs'] = _refresh_inputs(updates,record['private'],subject)
                    after = _snapshot(record['private'])
                    # 只允许surface的串行defects和显式刷新源；不掩盖其他公共写入。
                    allowed = {'mri/surface.defects.mgz'} if operation == 'surface' else set()
                    allowed.update(str(Path(source).resolve().relative_to(subject))
                                   for source in (updates or {}).values())
                    current_data = _data_snapshot(subject)
                    expected_data = {key: version for key, version in public_versions.items()
                                     if Path(key).parts[0] in ('mri', 'surf', 'label', 'stats')}
                    for relative in set(current_data) | set(expected_data):
                        if current_data.get(relative) == expected_data.get(relative):
                            continue
                        if relative not in allowed or relative not in current_data:
                            raise RuntimeError('undeclared parent stage_hook data write: ' + relative)
                        public_versions[relative] = current_data[relative]
                snapshots[hemi] = after
                row['publication_status'] = 'complete'
                atomic_json(subject/'scripts'/f'pipeline.{operation}.{hemi}.stage.json', row)
                record['next'] += 1
                atomic_json(root/f'{hemi}.{sequence:03d}.ack.json', {'sequence':sequence,'pid':process.pid})
            sampler.sample_if_due()
            time.sleep(.02)
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None,0) for process in processes):
                raise RuntimeError('persistent worker finalization failed')
            sampler.sample_if_due()
            time.sleep(.02)
        for hemi, record in records.items():
            process = record['process']
            if process.returncode != 0 or not record['report_path'].is_file():
                raise RuntimeError(f'{hemi} persistent worker exited without successful report')
            worker = json.loads(record['report_path'].read_text())
            if worker.get('status') != 'complete':
                raise RuntimeError(f'{hemi} persistent worker final report failed')
            report['workers'][hemi] = _normalize_paths(worker,record['private'],subject)
            report['workers'][hemi]['environment'] = record['environment']
        for relative in set(report['published']):
            target = subject / relative
            current = (target.stat().st_size, target.stat().st_mtime_ns) if target.is_file() else None
            if current != public_versions.get(relative):
                raise RuntimeError('published output changed outside pipeline: ' + relative)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        failure = HemisphereGroupError(f'persistent hemisphere pipeline failed: {error}',report)
        raise failure from error
    finally:
        errors = []
        try:
            _cancel(processes)
        except BaseException as error:
            errors.append(('process_reaping',repr(error)))
        for stream in streams:
            try:
                stream.close()
            except BaseException as error:
                errors.append(('log_close',repr(error)))
        if report['status'] != 'complete':
            for hemi, record in records.items():
                path = record.get('report_path')
                if path and path.is_file():
                    try:
                        report['workers'][hemi] = json.loads(path.read_text())
                    except (ValueError,OSError) as error:
                        errors.append(('worker_failure_report',repr(error)))
                else:
                    report['workers'].setdefault(hemi,{}).update(status='report_unavailable')
                for sequence,specification in enumerate(specifications):
                    receipt = root/f'{hemi}.{sequence:03d}.stage.json'
                    if receipt.is_file() and hemi not in report['stages'][specification['operation']]:
                        report['stages'][specification['operation']][hemi] = json.loads(receipt.read_text())
        if sampler:
            try:
                sampler.sample_if_due(force=True)
                report['device_process_tree'] = sampler.report()
            except BaseException as error:
                errors.append(('device_sampling_finalization',repr(error)))
        cleanup_started = time.monotonic()
        try:
            shutil.rmtree(root)
        except BaseException as error:
            errors.append(('private_directory_cleanup',repr(error)))
            report['private_directory_retained'] = str(root)
        report['cleanup_seconds'] = time.monotonic() - cleanup_started
        report['group_wall_seconds'] = time.monotonic() - started
        if errors:
            report.update(status='failed',finalization_errors=errors)
        try:
            atomic_json(subject/'scripts/pipeline.hemisphere-group.json',report)
        except BaseException as error:
            errors.append(('group_report_publication',repr(error)))
            report.update(status='failed',finalization_errors=errors)
        if errors:
            if failure:
                failure.add_note('Pipeline finalization errors: '+repr(errors))
            else:
                raise HemisphereGroupError('persistent hemisphere finalization failed',report)
    return report
