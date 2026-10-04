#!/usr/bin/env python3
"""CPU-only bounded metadata snapshot; never follow resource/source paths."""
import argparse
import datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import stat

FIXED = ('config.json', 'guard.config.json', 'launch.json', 'admission.json',
         'nominal/diagnostics/launch.json', 'attempt_01/retry_config.json',
         'attempt_01/monitor/monitor.json', 'attempt_01/monitor/gpu_samples.csv',
         'attempt_01/subject/fnit-native-free-run.json')
FOLDERS = {'guard': ('*.json', '*.log'),
           'attempt_01/diagnostics': ('*.json', '*.log'),
           'attempt_01/subject/scripts': ('*.hemisphere-group.json', '*.startup-*.report.json',
                                         '*.startup-*.worker.log', '*.startup-*.request.json')}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def open_directory(root_fd, relative):
    fd = os.dup(root_fd)
    try:
        for name in relative.split('/') if relative else ():
            next_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def declared_code_files(value, location=''):
    """Inspect copied JSON values only; do not open any referenced file."""
    found = []
    suffixes = ('.py', '.c', '.h', '.cpp', '.cu', '.so')
    if isinstance(value, dict):
        path = value.get('path')
        digest = value.get('sha256')
        if isinstance(path, str) and path.endswith(suffixes) and isinstance(digest, str):
            found.append({'path': path, 'sha256': digest, 'json_location': location,
                          'verification': 'declared metadata; referenced file not read'})
        for key, child in value.items():
            where = location + '/' + str(key)
            if isinstance(key, str) and key.endswith(suffixes):
                sha = child if isinstance(child, str) else child.get('sha256') if isinstance(child, dict) else None
                if isinstance(sha, str) and len(sha) == 64:
                    found.append({'path': key, 'sha256': sha, 'json_location': where,
                                  'verification': 'declared metadata; referenced file not read'})
            found.extend(declared_code_files(child, where))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(declared_code_files(child, location + '/' + str(index)))
    return found


def summarize(parsed):
    completions = [(name, data) for name, data in parsed.items() if name == 'attempt_01/diagnostics/completion.json']
    completion = completions[0][1] if completions else None
    pipeline = parsed.get('attempt_01/subject/fnit-native-free-run.json')
    mesh = pipeline.get('mesh_validation') if pipeline else None
    outputs = pipeline.get('output_validation') if pipeline else None
    completion_state = 'pending' if completion is None else completion.get('execution_status', 'unknown')
    admission = parsed.get('admission.json')
    admission_ready = bool(admission and admission.get('status') == 'complete'
                           and admission.get('exit_code') == 0 and admission.get('finished_utc')
                           and admission.get('cleanup_status') in
                           (None, 'all_owned_computation_exited', 'all_owned_exited', 'complete', 'completed'))
    # finished_utc is written after normal cleanup even if no TERM/KILL was needed.
    monitor = parsed.get('attempt_01/monitor/monitor.json')
    monitor_ready = monitor is None or (monitor.get('exit_code') == 0
                                        and monitor.get('monitor_thread_finished') is True)
    failure_records = [(name, data) for name, data in parsed.items()
                       if name.startswith('guard/') or name == 'admission.json']
    failures = [{'path': name, 'status': data.get('status'), 'exit_code': data.get('exit_code'),
                 'whole_case_exit_code': data.get('whole_case_exit_code'), 'error': data.get('error')}
                for name, data in failure_records if
                str(data.get('status', '')).startswith('failed')
                or str(data.get('status', '')).endswith('_failed')
                or data.get('status') in ('error', 'rejected', 'child_failed', 'interrupted')
                or any(isinstance(data.get(key), int) and data[key] != 0
                       for key in ('exit_code', 'whole_case_exit_code'))]
    output138_ready = bool(isinstance(outputs, dict) and outputs.get('status') == 'passed'
                           and outputs.get('expected') == 138 and outputs.get('present') == 138
                           and outputs.get('missing') == [])
    success = bool(completion and completion.get('execution_status') == 'complete' and completion.get('exit_code') == 0
                   and completion.get('pipeline_status') == 'complete' and pipeline and pipeline.get('status') == 'complete'
                   and isinstance(mesh, dict) and mesh.get('status') == 'passed'
                   and output138_ready and admission_ready and monitor_ready and not failures)
    if success:
        state = 'complete_integrity_passed'
    elif completion and (completion.get('execution_status') == 'failed' or completion.get('exit_code', 0) != 0):
        state = 'failed'
    elif failures:
        state = 'failed_preflight_or_guard'
    elif completion and completion.get('execution_status') == 'complete':
        state = 'completion_recorded_integrity_missing_or_failed'
    else:
        state = 'pending'
    groups = []
    standalone = []
    bindings = []
    for name, data in parsed.items():
        # Only declared source-code fields; never hash/read their external paths.
        selected = {k: v for k, v in data.items() if k in ('source', 'code_root', 'code_commit',
                     'source_archive_sha256', 'source_manifest', 'source_script_manifest', 'script_sha256',
                     'candidate_native_free_sha256', 'source_files', 'source_bindings', 'resource_sha256', 'inventory_sha256') or k.startswith('imported_fnit_sources')}
        code_files = declared_code_files(data)
        if selected or code_files:
            bindings.append({'snapshot_file': name, 'declared_source_binding': selected,
                             'declared_code_file_sha256': code_files, 'external_paths_read': False})
        if name.endswith('.hemisphere-group.json'):
            workers = {}
            attempts = data.get('startup_attempts', {})
            if not isinstance(attempts, dict):
                attempts = {}
            final_workers = data.get('workers', {})
            if not isinstance(final_workers, dict):
                final_workers = {}
            for hemi in ('lh', 'rh'):
                rows = attempts.get(hemi)
                if isinstance(rows, list):
                    reports = [row.get('worker_report', {}) if isinstance(row, dict) else {} for row in rows]
                    reports = [row if isinstance(row, dict) else {} for row in reports]
                    count = len(rows)
                    count_scope = 'reported startup_attempts list; not algorithm calls'
                else:
                    final = final_workers.get(hemi, {})
                    reports = [final if isinstance(final, dict) else {}]
                    count = None
                    count_scope = 'startup attempt count not recorded; no inference from final worker'
                flags = [row.get('operation_entered') for row in reports]
                workers[hemi] = {'startup_attempt_count': count, 'count_scope': count_scope,
                                 'operation_entered_each_attempt': flags,
                                 'known_operation_entries': sum(x is True for x in flags),
                                 'operation_entry_count_known': bool(flags) and all(isinstance(x, bool) for x in flags),
                                 'attempts': [{'pid': row.get('pid'), 'status': row.get('status'),
                                               'operation_entered': row.get('operation_entered'),
                                               'cuda_bootstrap_seconds': row.get('cuda_bootstrap_seconds')}
                                              for row in reports]}
            groups.append({'path': name, 'operation': data.get('operation'), 'status': data.get('status'),
                           'startup_wait_actual_seconds': data.get('startup_wait_actual_seconds', data.get('startup_wait_seconds_actual')),
                           'startup_wait_source_field': 'startup_wait_actual_seconds' if 'startup_wait_actual_seconds' in data else 'startup_wait_seconds_actual' if 'startup_wait_seconds_actual' in data else None, 'workers': workers})
        if '.startup-' in name and name.endswith('.report.json'):
            standalone.append({'path': name, 'pid': data.get('pid'), 'status': data.get('status'),
                               'operation_entered': data.get('operation_entered')})
    guard_states = [{'path': name, 'status': data.get('status'), 'exit_code': data.get('exit_code'), 'whole_case_exit_code': data.get('whole_case_exit_code')}
                    for name, data in parsed.items() if name.startswith('guard/')]
    running_records = [name for name, data in parsed.items() if data.get('status') in ('running', 'executing', 'waiting_gpu', 'waiting_lock')
                       or data.get('execution_status') == 'running']
    return {'run_state': state, 'passed': success, 'completion_state': completion_state,
            'observed_execution_state': 'completed_with_integrity' if success else 'running_or_waiting_reported' if running_records else 'completion_recorded' if completion else 'unknown_pending',
            'running_or_waiting_records': running_records, 'guard_status_records': guard_states,
            'completion': completion, 'pipeline_status': pipeline.get('status') if pipeline else 'pending',
            'mesh_validation': mesh if mesh is not None else {'status': 'missing'},
            'output_validation': outputs if outputs is not None else {'status': 'missing'},
            'guard_failure_records': failures, 'guard_exited_is_success_evidence': False,
            'admission': admission, 'monitor': monitor,
            'gates': {'output138_ready': output138_ready, 'admission_finished_successfully': admission_ready,
                      'existing_monitor_finished_successfully': monitor_ready},
            'groups': groups, 'standalone_startup_worker_reports': standalone, 'declared_source_bindings': bindings,
            'official_equivalence': 'not_assessed', 'integrity_scope': 'reported pipeline validation only; no mesh/MRI file reads'}


def collect(run_root, output):
    run_root, output = Path(run_root).absolute(), Path(output).absolute()
    resolved_root, resolved_out = run_root.resolve(), output.resolve()
    if resolved_out == resolved_root or resolved_out in resolved_root.parents or resolved_root in resolved_out.parents:
        raise ValueError('new output must be separate from input run')
    output.mkdir(parents=True, exist_ok=False)
    root_fd = os.open(run_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    observed = utc()
    files, parsed, rejected = [], {}, []
    try:
        names = set(FIXED)
        for folder, patterns in FOLDERS.items():
            try:
                fd = open_directory(root_fd, folder)
            except FileNotFoundError:
                continue
            except OSError as error:
                rejected.append({'path': folder, 'error': str(error)})
                continue
            try:
                for name in os.listdir(fd):
                    if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
                        names.add(folder + '/' + name)
            finally:
                os.close(fd)
        for relative in sorted(names):
            path = Path(relative)
            if 'license' in path.name.lower() or 'licence' in path.name.lower():
                rejected.append({'path': relative, 'error': 'license-named files excluded'})
                continue
            try:
                parent_fd = open_directory(root_fd, str(path.parent) if str(path.parent) != '.' else '')
                try:
                    fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
                finally:
                    os.close(parent_fd)
                try:
                    before = os.fstat(fd)
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError('not a regular metadata file')
                    with os.fdopen(fd, 'rb', closefd=False) as stream:
                        data = stream.read()  # Single byte snapshot; hash and parse this same data.
                    after = os.fstat(fd)
                finally:
                    os.close(fd)
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as error:
                rejected.append({'path': relative, 'error': str(error)})
                continue
            destination = output / 'snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            row = {'relative': relative, 'observed_utc': utc(), 'size_bytes': len(data), 'sha256': sha_bytes(data),
                   'source_changed_during_read': (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)}
            if relative.endswith('.json'):
                try:
                    value = json.loads(data)
                    if not isinstance(value, dict):
                        raise ValueError('JSON root must be object')
                    parsed[relative] = value
                    row['parse_status'] = 'valid'
                except (ValueError, UnicodeError, RecursionError) as error:
                    row.update(parse_status='invalid_or_incomplete', parse_error=str(error))
            else:
                row['parse_status'] = 'not_parsed'
            files.append(row)
    finally:
        os.close(root_fd)
    summary = summarize(parsed)
    invalid = [f['relative'] for f in files if f['parse_status'] == 'invalid_or_incomplete']
    changed = [f['relative'] for f in files if f['source_changed_during_read']]
    if invalid or changed:
        summary['passed'] = False
        summary['run_state'] = 'invalid_or_incomplete_snapshot'
    summary.update(schema=1, run_root=str(run_root), observed_utc=observed, finished_utc=utc(),
                   files=files, invalid_json_files=invalid, changed_during_read_files=changed,
                   rejected_paths=rejected, cpu_only=True,
                   scope='one-pass allowlisted JSON/CSV/log metadata; non-atomic across files; no resource/source payloads read',
                   collector={'path': str(Path(__file__).absolute()), 'sha256': sha_bytes(Path(__file__).read_bytes())})
    (output / 'startup.followup.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text('# Startup follow-up metadata快照\n\n'
        + f"observed UTC: {observed}\n\n状态：{summary['run_state']}；completion：{summary['completion_state']}；passed={summary['passed']}。\n\n"
        + '只读metadata；无completion为pending，guard exited不当通过。mesh完整性仅为pipeline报告，未读取影像或mesh。\n\n'
        + '详细startup尝试、operation_entered、原字节SHA、无效JSON及源码声明绑定见startup.followup.json。\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.run_root, args.output)
    print(json.dumps({'run_state': report['run_state'], 'completion_state': report['completion_state'],
                      'passed': report['passed'], 'output': str(args.output.absolute())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
