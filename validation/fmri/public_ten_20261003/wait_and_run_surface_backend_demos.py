"""标准库 GPU0 等待器：正式队列全部正常保存后，持原物理锁串行验证四项完整 surface。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import shutil
import subprocess
import sys
import time
import traceback

REVISION = '1128bc52c7a0233266e5b8a8d7dc0b382994e676'
SUBJECTS = ('CON03', 'CON05', 'CON07', 'CON09', 'CON11')
FORBIDDEN_MODULES = ('torch', 'numpy', 'nibabel', 'fnit')
FASTPD_BINARY_SHA256 = '33c3f4c157897c42f9e5a4e26276248dbe2b3c23948078d92c254eab5896da56'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def process_snapshot(pid):
    path = Path('/proc') / str(pid)
    try:
        status = path.joinpath('status').read_text()
        uid = int(next(line for line in status.splitlines() if line.startswith('Uid:')).split()[1])
        stat = path.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        cmdline = path.joinpath('cmdline').read_bytes()
        return {'pid': int(pid), 'uid': uid, 'state': stat[0], 'ppid': int(stat[1]),
                'starttime': int(stat[19]), 'cmdline_sha256': hashlib.sha256(cmdline).hexdigest()}
    except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
        return None


def same_process(record):
    current = process_snapshot(record['pid'])
    return current is not None and all(current[key] == record[key] for key in ('uid', 'starttime', 'cmdline_sha256'))


def update_descendants(roots, tracked):
    rows = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row = process_snapshot(int(path.name))
            if row is not None and row['uid'] == os.getuid():
                rows[row['pid']] = row
    parents = {record['pid'] for record in roots + list(tracked.values()) if same_process(record)}
    while True:
        additions = [row for row in rows.values() if row['ppid'] in parents and row['pid'] not in parents]
        if not additions:
            break
        for row in additions:
            parents.add(row['pid'])
            tracked[str(row['pid']) + ':' + str(row['starttime'])] = row
    return [record for record in tracked.values() if same_process(record)]


def queue_gate(queue, reports, *, original_process_alive, tracked_descendants_alive):
    """纯 JSON/进程身份合同；空锁不能代替全部正式例成功。"""
    if queue.get('source_revision') != REVISION:
        return 'blocked', 'formal queue producer revision differs'
    if queue.get('status') in ('failed', 'stopped_on_failure'):
        return 'blocked', 'formal queue stopped on a preserved failure'
    cases = queue.get('cases', {})
    for subject in SUBJECTS:
        entry = cases.get(subject)
        report = reports.get(subject)
        if entry is not None and (entry.get('status') == 'failed' or entry.get('exit_code', 0) != 0):
            return 'blocked', subject + ' formal queue entry failed'
        if report is not None and report.get('status') == 'failed':
            return 'blocked', subject + ' saved formal report failed'
    if queue.get('status') != 'complete':
        if not original_process_alive:
            return 'blocked', 'original queue exited before final complete; running/missing reports retained'
        return 'waiting', 'original formal queue is still active'
    if set(cases) != set(SUBJECTS):
        return 'blocked', 'final queue case closure differs from the five authorized GPU0 subjects'
    for subject in SUBJECTS:
        entry, report = cases[subject], reports.get(subject)
        if entry.get('status') != 'complete' or entry.get('exit_code') != 0:
            return 'blocked', subject + ' final queue entry is not complete/exit0'
        if report is None or report.get('status') != 'complete' or report.get('source_revision') != REVISION:
            return 'blocked', subject + ' saved formal report is not complete under source1128'
    if original_process_alive:
        return 'waiting', 'all saved cases complete; original launcher/queue has not exited yet'
    if tracked_descendants_alive:
        return 'blocked', 'original formal descendant identity is still alive after the final queue state'
    return 'eligible', 'five formal reports and queue entries complete/exit0, final queue complete, original identities gone'


def read_gate(cohort, roots, tracked):
    queue_path = cohort / 'candidate_v4/queue_CON03.private.json'
    queue = json.loads(queue_path.read_text())
    reports = {}
    for subject in SUBJECTS:
        path = cohort / ('candidate_v4/' + subject + '/report/report.public.json')
        if path.is_file():
            reports[subject] = json.loads(path.read_text())
    living_descendants = update_descendants(roots, tracked)
    original_alive = any(same_process(record) for record in roots)
    decision, reason = queue_gate(queue, reports, original_process_alive=original_alive,
                                 tracked_descendants_alive=bool(living_descendants))
    snapshot = {'decision': decision, 'reason': reason, 'queue_status': queue.get('status'),
        'cases': {subject: {'queue_status': queue.get('cases',{}).get(subject,{}).get('status'),
                           'exit_code': queue.get('cases',{}).get(subject,{}).get('exit_code'),
                           'saved_report_status': reports.get(subject,{}).get('status')}
                  for subject in SUBJECTS}, 'original_queue_or_launcher_alive': original_alive,
        'original_tracked_descendant_alive_count': len(living_descendants)}
    return snapshot


def hardware_zero():
    text = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=index,uuid,name,pci.bus_id',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20)
    rows = [[item.strip() for item in line.split(',')] for line in text.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][0] != '0' or not rows[0][1].startswith('GPU-'):
        raise ValueError('physical GPU0 NVML identity cannot be verified')
    return dict(zip(('physical_index','uuid','name','pci_bus_id'), rows[0]))


def open_original_lock(path, identity):
    """只打开原有普通锁文件；路径替换、软链接和 inode 漂移均拒绝。"""
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    current = os.fstat(descriptor)
    if (not stat.S_ISREG(current.st_mode) or current.st_dev != identity['device']
            or current.st_ino != identity['inode']):
        os.close(descriptor)
        raise ValueError('the original physical lock inode changed while opening it')
    return descriptor


def surface_native_records(source, cases, search_path):
    """只读绑定所选 Workbench、自编 FastPD 与其五个源文件；不导入扩展。"""
    records = {}
    binaries = sorted((source/'src/fnit/msm').glob('_fastpd_native*.so'))
    if len(binaries)!=1 or sha256(binaries[0])!=FASTPD_BINARY_SHA256:
        raise ValueError('the actual frozen independently compiled FastPD binary differs')
    files={'fastpd_binary':binaries[0]}
    native_sources=source/'src/fnit/msm/_fastpd_src'
    required={'FastPD.h','block.h','fastpd_module.cpp','fnit_fastpd_model_stub.h','graph.h'}
    if {path.name for path in native_sources.iterdir() if path.is_file()}!=required:
        raise ValueError('the fixed FastPD native source closure differs')
    files.update({'fastpd_source/'+name:native_sources/name for name in sorted(required)})
    for name,path in cases.items():
        config=json.loads((path/'config.private.json').read_text())
        selected=shutil.which(str(config.get('wb_command','wb_command')),path=search_path)
        if selected is None:
            raise ValueError('the actual selected Workbench executable is missing')
        files['workbench/'+name]=Path(selected)
    for name,path in files.items():
        if not path.is_file():
            raise ValueError('a selected surface native dependency is not a regular readable file')
        records[name]={'requested_path':str(path),'resolved_path':str(path.resolve()),'sha256':sha256(path)}
    return records


def validate_surface_report(report, expected_uuid, expected_runner_sha):
    required = ('volume_ready','volume_reused','reconstruction_reused','readonly_input_guards_equal',
                'frozen_source_guards_equal','validation_helper_guard_equal','demo_runner_guard_equal',
                'configuration_guard_equal','actual_cuda_context_uuid_verified')
    if report.get('status') != 'complete' or any(report.get(key) is not True for key in required):
        raise ValueError('full surface API or required source/input/cache/UUID guards did not complete')
    if (report.get('source_revision') != REVISION or report.get('surface_device') != 'cuda:0'
            or report.get('cuda_visible_devices') != '0' or report.get('actual_cuda_context_gpu_uuid') != expected_uuid
            or report.get('demo_runner_sha256') != expected_runner_sha or report.get('frame_count') != 180
            or report.get('tr_seconds') != 2.1 or report.get('outputs',{}).get('dtseries',{}).get('shape') != [180,91282]):
        raise ValueError('completed output/device/frame/source identity differs from the locked request')
    for hemi in ('L','R'):
        if report.get('outputs',{}).get(hemi,{}).get('shape') != [180,32492]:
            raise ValueError('full 180-frame hemisphere output is missing')
    if report.get('under_20_gb') is not True or report.get('gpu_monitor_error_count') != 0:
        raise ValueError('the sampled GPU memory measurement did not verify the 20 decimal GB target')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fnit-root', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--formal-launcher-pid', type=int, required=True)
    parser.add_argument('--formal-queue-pid', type=int, required=True)
    parser.add_argument('--poll-seconds', type=float, default=10.)
    args = parser.parse_args()
    if any(name in sys.modules for name in FORBIDDEN_MODULES):
        raise RuntimeError('the waiting process must use only the standard library before obtaining the lock')
    root = args.fnit_root
    index = json.loads((root/'INDEX.json').read_text())
    entry = index['active_tasks']['fmri_surface_backends_20261003']
    workspace = root / entry['workspace']
    run_root = root / entry['runs']
    cohort = root / 'workspaces/fnit_surface_ten_public_20261003'
    source = cohort / 'source_1128bc52'
    if entry['frozen_source_revision'] != REVISION:
        raise ValueError('unified index no longer binds the frozen scientific source1128')
    queue_path = cohort/'candidate_v4/queue_CON03.private.json'
    queue = json.loads(queue_path.read_text())
    if queue.get('pid') != args.formal_queue_pid:
        raise ValueError('the original queue PID is not the currently saved GPU0 queue identity')
    roots = [process_snapshot(pid) for pid in (args.formal_launcher_pid,args.formal_queue_pid)]
    if any(record is None or record['uid'] != os.getuid() for record in roots):
        raise ValueError('capture the living authorized original formal processes before waiting')
    runner = workspace/'run_surface_backend_demo_v5.py'
    comparator = workspace/'compare_surface_backend_outputs_v1.py'
    dependencies = [Path(__file__),runner,comparator,source/'validation/fmri/public_ten_20261003/run_fnit_cohort.py']
    dependency_hashes = {str(path):sha256(path) for path in dependencies}
    cases = {
        'freesurfer':run_root/'freesurfer-source1128-prepared-v4',
        'provided_directory':run_root/'provided-source1128-prepared-v4',
        'provided_zip':run_root/'provided-zip-gpu-source1128-prepared-v2',
        'fnit_surface_only':run_root/'fnit-surface-only-source1128-prepared-v2'}
    config_hashes = {}
    for name,path in cases.items():
        config = json.loads((path/'config.private.json').read_text())
        if config.get('device') != 'cuda:0' or config.get('auto_volume') is not False or (path/'gpu_surface_attempt01').exists():
            raise ValueError('each locked demo requires its unchanged CUDA request and a new attempt')
        config_hashes[name] = sha256(path/'config.private.json')
    native_search_path=str(args.python.parent)+os.pathsep+os.environ.get('PATH','')
    native_records=surface_native_records(source,cases,native_search_path)
    target = run_root/'gpu_lane0_waiter_v2'
    target.mkdir(mode=0o700,exist_ok=False)
    lock_path = cohort/'.fnit_physical_gpu_0_benchmark.lock'
    lock_stat = lock_path.stat()
    lock_identity = {'device':lock_stat.st_dev,'inode':lock_stat.st_ino}
    plan = {'original_processes':roots,'dependency_hashes':dependency_hashes,'configuration_sha256':config_hashes,
            'lock_path':str(lock_path.resolve()),'lock_identity':lock_identity,'hostname':socket.gethostname(),
            'surface_native_records':native_records}
    save(target/'plan.private.json',plan)
    started = time.time()
    state = {'status':'waiting_formal_lane0','source_revision':REVISION,'waiter_pid':os.getpid(),
        'waiter_source_sha256':sha256(__file__),'surface_runner_sha256':sha256(runner),
        'comparison_source_sha256':sha256(comparator),'configuration_sha256':config_hashes,
        'surface_native_sha256_before':{name:value['sha256'] for name,value in native_records.items()},
        'models_imported_before_lock':[],'lock_deleted_or_replaced':False,
        'normal_queue_completion_evidence':'Immutable cohort code writes final complete only after all five subprocess exit0, then returns0; original launcher/queue identities must be absent. No direct waitpid OS exit code is claimed for a process that is not this waiter child.',
        'cases':{},'comparisons':{}}
    save(target/'report.public.json',state)
    tracked = {}
    last_gate = None
    descriptor = None
    try:
        while descriptor is None:
            gate = read_gate(cohort,roots,tracked)
            if gate != last_gate:
                state.update(status='blocked_formal_lane0' if gate['decision']=='blocked' else 'waiting_formal_lane0',gate=gate)
                save(target/'tracked_processes.private.json',list(tracked.values()))
                save(target/'report.public.json',state)
                print('FORMAL_GATE',json.dumps(gate),flush=True)
                last_gate = gate
            if gate['decision']=='eligible':
                current = lock_path.stat()
                if current.st_dev != lock_identity['device'] or current.st_ino != lock_identity['inode']:
                    raise ValueError('the original physical lock inode changed')
                candidate = open_original_lock(lock_path,lock_identity)
                try:
                    fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(candidate)
                else:
                    recheck = read_gate(cohort,roots,tracked)
                    if recheck['decision']!='eligible':
                        os.close(candidate)
                    else:
                        descriptor = candidate
                        state['gate_after_lock'] = recheck
                        break
            time.sleep(args.poll_seconds)
        if any(name in sys.modules for name in FORBIDDEN_MODULES):
            raise RuntimeError('the waiting parent imported scientific models before launching any locked child')
        for path,digest in dependency_hashes.items():
            if sha256(path)!=digest:
                raise ValueError('a frozen waiting/runner/comparison dependency changed')
        if surface_native_records(source,cases,native_search_path)!=native_records:
            raise ValueError('a selected surface native dependency changed while waiting')
        gpu = hardware_zero()
        state.update(status='running_locked_gpu_demos',lock_acquired=True,physical_gpu=gpu,
                     lock_wait_seconds=time.time()-started,models_imported_at_lock_acquisition=[])
        save(target/'report.public.json',state)
        for name,path in cases.items():
            if surface_native_records(source,cases,native_search_path)!=native_records:
                raise ValueError('a selected surface native dependency changed before a locked API')
            config_path = path/'config.private.json'
            if sha256(config_path)!=config_hashes[name]:
                raise ValueError('a staged configuration changed while waiting')
            output = path/'gpu_surface_attempt01'
            config = json.loads(config_path.read_text())
            environment = dict(os.environ,PYTHONPATH=str(source/'src'),CUDA_VISIBLE_DEVICES='0',
                FNIT_EXPECTED_PHYSICAL_GPU_UUID=gpu['uuid'],OMP_NUM_THREADS=str(config.get('cpu_threads',8)),
                OPENBLAS_NUM_THREADS=str(config.get('cpu_threads',8)),MKL_NUM_THREADS=str(config.get('cpu_threads',8)))
            environment.pop('PYTORCH_NO_CUDA_MEMORY_CACHING',None)
            environment['PATH'] = str(args.python.parent)+os.pathsep+environment.get('PATH','')
            argv = [str(args.python),'-u',str(runner),'--config',str(config_path),'--output',str(output),
                    '--source-root',str(source),'--source-revision',REVISION]
            log = root/'logs'/('fmri_surface_backends_20261003_'+name+'_gpu_attempt01.log')
            tick = time.perf_counter()
            with log.open('x') as stream:
                child = subprocess.Popen(argv,env=environment,stdin=subprocess.DEVNULL,stdout=stream,
                                         stderr=subprocess.STDOUT,pass_fds=(descriptor,))
                state['cases'][name] = {'status':'running','pid':child.pid,'CUDA_visible_devices':'0',
                    'expected_physical_gpu_uuid':gpu['uuid'],'configuration_sha256':config_hashes[name]}
                save(target/'report.public.json',state)
                code = child.wait()
            outcome = state['cases'][name]
            outcome.update(exit_code=code,process_wall_seconds=time.perf_counter()-tick)
            try:
                report_path = output/'report.public.json'
                report = json.loads(report_path.read_text())
                if code:
                    raise RuntimeError('the preserved full GPU surface attempt exited unsuccessfully')
                validate_surface_report(report,gpu['uuid'],dependency_hashes[str(runner)])
                if surface_native_records(source,cases,native_search_path)!=native_records:
                    raise ValueError('a selected surface native dependency changed during a locked API')
                outcome.update(status='complete',surface_report_sha256=sha256(report_path),
                    surface_native_dependencies_guard_equal=True,
                    actual_gpu_uuid=report['actual_cuda_context_gpu_uuid'],full_api_seconds=report['full_api_seconds'],
                    job_tree_peak_gpu_bytes=report['job_tree_peak_gpu_bytes'])
            except Exception as error:
                outcome.update(status='failed',error_type=type(error).__name__,detail=str(error))
            save(target/'report.public.json',state)
        pairs = [
            ('provided_dir_vs_zip_gpu',cases['provided_directory']/'gpu_surface_attempt01/files.private.json',cases['provided_zip']/'gpu_surface_attempt01/files.private.json'),
            ('provided_dir_cpu_vs_gpu',run_root/'provided-dir-cpu-source1128-prepared-v2/cpu_surface_attempt01/files.private.json',cases['provided_directory']/'gpu_surface_attempt01/files.private.json'),
            ('provided_zip_cpu_vs_gpu',run_root/'provided-zip-cpu-source1128-prepared-v2/cpu_surface_attempt01/files.private.json',cases['provided_zip']/'gpu_surface_attempt01/files.private.json'),
            ('fnit_formal_vs_surface_only_gpu',cohort/'candidate_v4/CON01/report/files.private.json',cases['fnit_surface_only']/'gpu_surface_attempt01/files.private.json')]
        cpu_environment = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        for name,reference,candidate in pairs:
            result = target/(name+'.public.json')
            log = target/(name+'.private.log')
            with log.open('x') as stream:
                code = subprocess.run([str(args.python),'-u',str(comparator),'--reference-files',str(reference),
                    '--candidate-files',str(candidate),'--output',str(result)],env=cpu_environment,stdout=stream,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,pass_fds=(descriptor,)).returncode
            state['comparisons'][name] = {'status':'complete' if code==0 else 'failed','exit_code':code,
                'comparison_report_sha256':sha256(result) if result.is_file() else None,
                'scope':'read-only full saved arrays and spheres, no fitting; measured after independent API calls'}
            save(target/'report.public.json',state)
        dependency_guards = {path:sha256(path)==digest for path,digest in dependency_hashes.items()}
        configuration_guards = {name:sha256(path/'config.private.json')==config_hashes[name]
                                for name,path in cases.items()}
        native_after=surface_native_records(source,cases,native_search_path)
        if not all(dependency_guards.values()) or not all(configuration_guards.values()):
            raise ValueError('a locked source/helper/configuration changed during demonstrations')
        if native_after!=native_records:
            raise ValueError('a selected surface native dependency changed during demonstrations')
        state.update(status='complete' if all(value['status']=='complete' for value in list(state['cases'].values())+list(state['comparisons'].values())) else 'completed_with_preserved_failures',
                     frozen_dependencies_guard_equal=all(dependency_guards.values()),
                     frozen_configurations_guard_equal=all(configuration_guards.values()),
                     surface_native_sha256_after={name:value['sha256'] for name,value in native_after.items()},
                     surface_native_dependencies_guard_equal=True,
                     physical_gpu_after=hardware_zero(),waiter_source_guard_equal=sha256(__file__)==dependency_hashes[str(Path(__file__))])
        if state['physical_gpu_after']!=gpu:
            raise ValueError('physical GPU identity changed during locked demonstrations')
    except BaseException as error:
        (target/'failure.private.txt').write_text(traceback.format_exc())
        state.update(status='failed_waiter',error_type=type(error).__name__,detail=str(error))
        raise
    finally:
        save(target/'report.public.json',state)
        # Child inherits this same open file description. Closing the parent fd
        # cannot release a still-running child's lock on interruption.
        if descriptor is not None:
            os.close(descriptor)


if __name__=='__main__':
    main()
