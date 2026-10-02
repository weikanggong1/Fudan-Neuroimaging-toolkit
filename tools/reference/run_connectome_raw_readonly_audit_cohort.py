"""只读采集已完成官方 raw 参考，每例保存来源/文件审计及官方自身范围。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': sha256(path), 'size_bytes': path.stat().st_size}


def checked(item):
    current = record(item['path'])
    require(current['sha256'] == item['sha256'], 'actual immutable source/config/audit changed')
    return current


def atomic(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validated_audit(path, case, origin, config):
    result = json.loads(Path(path).read_text())
    require(result['case_id'] == case and result['contract_status'] == 'passed' and
            result['scientific_parity'] == 'not_assessed' and
            all(result['manifest'].get(key) == origin['reference_manifest'][key] for key in ('path', 'sha256')) and
            result['actual_worker']['sha256'] == config['source_files'][config['producer_worker']] and
            result['raw_case_binding']['raw_manifest_sha256'] == config['raw_manifest']['sha256'] and
            result['audit_script_sha256'] == config['source_files'][config['audit_worker']],
            'actual audit source/case/raw/reference identity differs')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--poll-seconds', type=float, default=30.)
    args = parser.parse_args(argv)
    require(args.poll_seconds >= 1, 'poll interval must be at least one second')
    require(sha256(args.config) == args.config_sha256, 'actual read-only audit configuration changed')
    config = json.loads(args.config.read_text())
    identities = {'configuration': record(args.config), 'raw_manifest': checked(config['raw_manifest']),
                  'case_origin_configuration': checked(config['case_origin_configuration'])}
    plan = json.loads(Path(identities['case_origin_configuration']['path']).read_text())
    require(Path(plan['raw_manifest']['path']).resolve() == Path(config['raw_manifest']['path']).resolve() and
            plan['raw_manifest']['sha256'] == config['raw_manifest']['sha256'],
            'read-only collector and official origin plan bind another raw source')
    cases = [case for group in plan['groups'].values() for case in group['case_ids']]
    require(len(set(cases)) == len(cases) == 10, 'exact canonical two-group cohort required')
    require(sha256(__file__) == config['source_files'][str(Path(__file__).resolve())],
            'actual read-only controller source differs')
    root = Path(config['output_root'])
    require(not root.exists(), 'fresh audit collector root required')
    root.mkdir(parents=True)
    status_path = root / 'cohort_status.json'
    state = {'scope': 'CPU read-only artifact/range collection; no official command or GPU execution',
        'state': 'waiting', 'execution_completed': False, 'scientific_parity': 'not_assessed',
        'controller_pid': os.getpid(), 'controller_sha256': sha256(__file__),
        'source_identity': identities, 'cases': {case: {'state': 'waiting'} for case in cases}}
    try:
        while True:
            require(sha256(args.config) == args.config_sha256, 'collector configuration changed while waiting')
            for path, digest in config['source_files'].items():
                checked({'path': path, 'sha256': digest})
            origin_path = Path(config['official_view']) / 'case_origin_binding.json'
            snapshot = origin_path.read_bytes()
            origin = json.loads(snapshot)
            require(origin['source_identity']['configuration']['sha256'] ==
                    config['case_origin_configuration']['sha256'] and set(origin['cases']) == set(cases),
                    'origin belongs to another actual two-group configuration')
            state['origin_snapshot'] = {'path': str(origin_path), 'sha256': hashlib.sha256(snapshot).hexdigest(),
                                       'policy': 'SHA of this exact dynamic atomic snapshot'}
            for case in cases:
                row, source = state['cases'][case], origin['cases'][case]
                if row['state'] in ('completed', 'failed') or source['state'] != 'completed':
                    continue
                manifest = checked(source['reference_manifest'])
                if case in config.get('completed_audits', {}):
                    output = checked(config['completed_audits'][case])
                    row['mode'] = 'reuse actual immutable previous read-only audit; no recalculation or copy'
                else:
                    path = root / (case + '.json')
                    command = [config['python'], config['audit_worker'], '--manifest', manifest['path'],
                        '--raw-manifest', identities['raw_manifest']['path'], '--raw-manifest-sha256',
                        identities['raw_manifest']['sha256'], '--case-id', case,
                        '--worker', config['producer_worker'], '--output', str(path)]
                    row.update(state='running', argv=command, manifest=manifest)
                    atomic(status_path, state)
                    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
                    with (root / (case + '.log')).open('x') as stream:
                        row['returncode'] = subprocess.run(command, env=env, stdout=stream,
                                                           stderr=subprocess.STDOUT).returncode
                    if row['returncode'] != 0:
                        row['state'] = 'failed'
                        continue
                    output = record(path)
                    row['mode'] = 'actual new read-only audit; no scientific output regenerated'
                result = validated_audit(output['path'], case, source, config)
                row.update(state='completed', audit=output, manifest=manifest,
                           cpu_readonly_audit_seconds=result['cpu_readonly_audit_seconds'])
            finished = all(row['state'] in ('completed', 'failed') for row in state['cases'].values())
            if finished or origin['state'] == 'failed':
                state['state'] = 'completed' if all(row['state'] == 'completed' for row in state['cases'].values()) else 'failed'
                state['execution_completed'] = state['state'] == 'completed'
            else:
                state['state'] = 'waiting_actual_reference_completion'
            state['utc'] = datetime.now(timezone.utc).isoformat()
            atomic(status_path, state)
            if finished or origin['state'] == 'failed':
                return
            time.sleep(args.poll_seconds)
    except Exception as error:
        state['state'] = 'failed'
        state['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        state['utc'] = datetime.now(timezone.utc).isoformat()
        atomic(status_path, state)


if __name__ == '__main__':
    main()
