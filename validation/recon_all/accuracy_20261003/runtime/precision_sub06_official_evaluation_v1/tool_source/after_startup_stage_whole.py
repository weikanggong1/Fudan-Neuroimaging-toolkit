"""阶段数值 gates 通过后准备新候选计划，再启动独立原始T1整例。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
from resource_admission import digest, inventory, now, retry_config


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def prepare_launch(config_path, guard):
    """Create a bound preparation receipt, never fabricate a prior execution."""
    config_path = Path(config_path)
    whole = json.loads(config_path.read_text())
    nominal = Path(whole['diagnostic_root'])
    for path in (nominal, Path(whole['output']), Path(guard['retry_root']), Path(guard['admission_report'])):
        if path.exists():
            raise FileExistsError('refusing existing preparation/output/attempt/receipt: ' + str(path))
    retry_config(whole, Path(guard['retry_root']))  # Original isolation and GPU/invocation gates.
    if not Path(whole['input']).is_file():
        raise FileNotFoundError('input unavailable: ' + whole['input'])
    if digest(whole['input']) != whole['input_sha256']:
        raise ValueError('input SHA mismatch: ' + whole['input'])
    for key in ('python', 'fs_license'):
        if not Path(whole[key]).is_file():
            raise FileNotFoundError(key + ' unavailable: ' + whole[key])
    source = Path(whole['code_root'])
    archive_path = guard.get('source_archive', whole.get('source_archive'))
    if not archive_path:
        raise ValueError('source_archive path is required for a new candidate preparation')
    archive_path = Path(archive_path)
    if digest(archive_path) != whole['source_archive_sha256']:
        raise ValueError('source archive SHA mismatch: ' + str(archive_path))
    checked = 0
    archive_sources = {}
    with tarfile.open(archive_path, 'r:*') as archive:
        for member in archive:
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('unsafe source archive member: ' + member.name)
            if member.issym() or member.islnk():
                raise ValueError('source archive links are unsupported: ' + member.name)
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            with stream:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(chunk)
            expected = hasher.hexdigest()
            actual_path = source / name
            if not actual_path.is_file():
                raise FileNotFoundError('archive-bound source missing: ' + str(actual_path))
            if digest(actual_path) != expected:
                raise ValueError('archive/source SHA mismatch: ' + str(actual_path))
            archive_sources[str(name)] = expected
            checked += 1
    critical = ('src/fnit/recon_all/native_free.py', 'src/fnit/recon_all/hemisphere_worker.py',
                'src/fnit/recon_all/hemisphere_parallel.py')
    for name in critical:
        if name not in archive_sources:
            raise ValueError('critical source missing from archive binding: ' + name)
    actual_python = {str(p.relative_to(source)) for p in source.rglob('*.py')
                     if '.git' not in p.parts and '__pycache__' not in p.parts}
    archived_python = {name for name in archive_sources if name.endswith('.py')}
    if actual_python != archived_python:
        raise ValueError('source/archive Python file set differs: ' + repr(sorted(actual_python ^ archived_python)))
    source_sha = {name: digest(source / name) for name in critical}
    for name, value in source_sha.items():
        if value != archive_sources[name]:
            raise ValueError('critical source changed during preparation: ' + str(source / name))
    for name, expected in guard.get('expected_source_sha256', {}).items():
        if digest(source / name) != expected:
            raise ValueError('declared source SHA mismatch: ' + str(source / name))
    bindings = inventory(whole)
    for path, expected in guard.get('expected_resource_sha256', {}).items():
        if bindings.get(str(Path(path).resolve())) != expected:
            raise ValueError('declared resource SHA mismatch: ' + path)
    # Resource/admission replay remains unchanged: all original config keys must bind.
    launch = {**whole, 'status': 'prepared_not_executed', 'algorithm_entered': False,
              'prepared_utc': now(), 'preparation_scope': 'new candidate launch plan; no historical execution',
              'config_sha256': digest(config_path), 'candidate_native_free_sha256': source_sha[critical[0]],
              'source_sha256': source_sha, 'source_archive_path': str(archive_path.resolve()),
              'verified_source_archive_sha256': digest(archive_path), 'archive_checked_files': checked,
              'resource_sha256': bindings}
    nominal.mkdir(parents=True, exist_ok=False)
    path = nominal / 'launch.json'
    with path.open('x') as stream:
        stream.write(json.dumps(launch, ensure_ascii=False, indent=2) + '\n')
    return {'path': str(path), 'sha256': digest(path), 'status': 'prepared_not_executed',
            'algorithm_entered': False, 'config_sha256': launch['config_sha256'],
            'source_sha256': source_sha, 'archive_checked_files': checked}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    evaluated_role = cfg.get('evaluation_role', 'startup_only_candidate')
    if evaluated_role not in ('startup_only_candidate', 'precision_candidate'):
        raise ValueError('evaluation_role must identify startup_only_candidate or precision_candidate')
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'waiting_stage', 'started_utc': now(), 'pid': os.getpid(),
              'config_sha256': digest(args.config), 'algorithm_entered': False,
              'evaluated_role': evaluated_role,
              'scope': evaluated_role + '; same-input stages first then raw T1 empty-directory whole case',
              'stage_queue': cfg['stage_queue'], 'whole_case_config': cfg['whole_case_config'],
              'all_required_gates': False}
    path = out / 'guard.json'
    write(path, report)
    phase = 'stage_queue'
    try:
        deadline = time.monotonic() + cfg['stage_wait_seconds']
        while True:
            queue = json.loads(Path(cfg['stage_queue']).read_text())
            if queue['status'] != 'running':
                break
            if time.monotonic() > deadline:
                raise TimeoutError('stage queue wait expired: ' + cfg['stage_queue'])
            time.sleep(5)
        if not queue.get('all_jobs_succeeded'):
            report.update(status='stage_execution_failed', queue_sha256=digest(cfg['stage_queue']))
            return 1
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '4'
        phase = 'stage_comparisons'
        comparison_root = Path(cfg.get('comparison_output_root', out / 'comparisons'))
        report['comparison_output_root'] = str(comparison_root)
        comparisons = []
        for case in cfg['cases']:
            root = Path(cfg['stage_root'])
            destination = comparison_root / (case + '_comparison')
            if destination.exists():
                raise FileExistsError('comparison receipt directory already exists: ' + str(destination))
            command = [sys.executable, cfg['comparison_script'], '--baseline-stage',
                       str(root / (case + '_baseline') / 'outputs/annotation.stage.json'),
                       '--candidate-stage', str(root / (case + '_candidate') / 'outputs/annotation.stage.json'),
                       '--output', str(destination)]
            with (out / (case + '.comparison.log')).open('x') as stream:
                code = subprocess.call(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            receipt = destination / 'annotation.pair.json'
            pair = json.loads(receipt.read_text()) if receipt.exists() else {}
            comparisons.append({'case': case, 'command': command, 'exit_code': code,
                'receipt': str(receipt), 'sha256': digest(receipt) if receipt.exists() else None,
                'exact_regression_pass': pair.get('exact_regression_pass')})
            report['comparisons'] = comparisons
            write(path, report)
            if code or pair.get('exact_regression_pass') is not True:
                report['status'] = 'stage_numeric_regression_failed'
                return 1
        if not cfg['cases']:
            raise ValueError('at least one required stage comparison is needed')
        report.update(all_required_gates=True, queue_sha256=digest(cfg['stage_queue']))
        phase = 'prepare_candidate_launch'
        report['prepared_launch'] = prepare_launch(cfg['whole_case_config'], cfg)
        report['status'] = 'whole_case_admission'
        write(path, report)
        command = [sys.executable, cfg['resource_script'], '--config', cfg['whole_case_config'],
                   '--retry-root', cfg['retry_root'], '--report', cfg['admission_report'],
                   '--lock', '/tmp/fnit-shared-benchmark.lock', '--poll-seconds', '30',
                   '--query-timeout', '5', '--maximum-wait-seconds', '86400']
        report['whole_case_command'] = command
        write(path, report)
        phase = 'whole_case_admission'
        report['algorithm_entered'] = None  # Dispatch cannot prove the pipeline function entered.
        write(path, report)
        with (out / 'whole_case_admission.log').open('x') as stream:
            code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
        report.update(status='whole_case_exited', whole_case_exit_code=code,
                      meaning='Admission dispatch only; actual algorithm entry/completion require its own receipts.')
        return code
    except BaseException as error:
        report.update(status='guard_failed', failure_phase=phase, error=repr(error) + '; ' + str(error), failure_file=getattr(error, 'filename', None), traceback=traceback.format_exc())
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        report['finished_utc'] = now()
        write(path, report)


if __name__ == '__main__':
    raise SystemExit(main())
