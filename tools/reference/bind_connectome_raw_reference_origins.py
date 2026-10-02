"""绑定两组真实官方输出：仅完成病例建立受控软链接，不运行或复制科学计算。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_record(record):
    path = Path(record['path'])
    require(path.is_file() and sha256(path) == record['sha256'], 'bound actual source/config/contract changed')
    return {'path': str(path.resolve()), 'sha256': record['sha256'], 'size_bytes': path.stat().st_size}


def atomic(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def configuration(config_path, digest):
    config_record = checked_record({'path': str(config_path), 'sha256': digest})
    config = json.loads(Path(config_path).read_text())
    raw_record = checked_record(config['raw_manifest'])
    raw = json.loads(Path(raw_record['path']).read_text())
    canonical = [row['case_id'] for row in raw['cases']]
    require(len(canonical) == len(set(canonical)), 'canonical case IDs duplicated')
    groups = config['groups']
    flattened = [case for row in groups.values() for case in row['case_ids']]
    require(len(groups) == 2 and len(flattened) == len(set(flattened)) and set(flattened) == set(canonical),
            'two mutually exclusive groups must cover the exact canonical cohort')
    roots = [str(Path(row['output_root']).resolve()) for row in groups.values()]
    require(len(set(roots)) == len(roots), 'group outputs must use independent roots')
    require(config['seeds'] == [0, 1, 2, 3, 4] and config['n_seeds'] == 100000,
            'this frozen workload requires five sequential seeds and 100k attempted seeds')
    sources = {str(Path(path).resolve()): checked_record({'path': path, 'sha256': value})
               for path, value in config['source_files'].items()}
    names = {Path(path).name: item['sha256'] for path, item in sources.items()}
    require(len(names) == len(sources), 'source basenames must be unambiguous')
    for required in ('run_connectome_raw_official_cohort.py', 'benchmark_connectome_raw_official.py',
                     'benchmark_connectome_repeats_official.py', 'connectome_repeat_common.py'):
        require(required in names, 'actual frozen controller/worker/helpers required')
    reference = checked_record(config['verified_reference_manifest'])
    launches, common_launch = {}, None
    for group, row in groups.items():
        launches[group] = checked_record(row['launch_configuration'])
        launch = json.loads(Path(launches[group]['path']).read_text())
        shared = {key: value for key, value in launch.items() if key not in ('case_ids', 'output_root')}
        require(common_launch is None or shared == common_launch, 'both groups must use identical frozen source/input/binary/seed configuration')
        common_launch = shared
        require(launch['case_ids'] == row['case_ids'] and
                Path(launch['output_root']).resolve() == Path(row['output_root']).resolve() and
                launch['n_seeds'] == config['n_seeds'] and launch['seeds'] == config['seeds'] and
                launch['raw_manifest_sha256'] == raw_record['sha256'] and
                launch['verified_reference_manifest_sha256'] == reference['sha256'],
                'frozen launch configuration differs from case origin plan')
    return config, {'configuration': config_record, 'raw_manifest': raw_record,
                    'verified_reference_manifest': reference, 'source_files': sources,
                    'launch_configurations': launches}, names


def completed_case(case, row, group_root, config, source_hashes):
    require(row.get('returncode') == 0, 'completed controller case has nonzero worker returncode')
    actual_root = group_root / case
    report = actual_root / 'reference_manifest.json'
    identity = checked_record({'path': str(report), 'sha256': row['reference_manifest_sha256']})
    result = json.loads(report.read_text())
    require(result.get('execution_completed') is True and result.get('state') == 'completed' and
            result.get('case_id') == case, 'actual official case report incomplete or belongs to another case')
    require(result['seeds'] == config['seeds'] and result['parameters']['n_seed_attempts'] == config['n_seeds'] and
            result['parameters']['tracking_threads'] == 0 and result['downstream_threads'] == 8,
            'actual scientific seed/thread configuration differs')
    require(result['script_sha256'] == source_hashes['benchmark_connectome_raw_official.py'] and
            result['reference_command_helper_sha256'] == source_hashes['benchmark_connectome_repeats_official.py'] and
            result['matrix_helper_sha256'] == source_hashes['connectome_repeat_common.py'],
            'actual reference runtime source differs from frozen workload')
    require(result['raw_case_binding']['manifest_sha256'] == config['raw_manifest']['sha256'] and
            result['raw_case_binding']['case_id'] == case and
            result['raw_case_binding']['official_eddy_solver']['mode'] == config['expected_eddy_solver'],
            'actual canonical raw/official EDDY lineage differs')
    require(len(result['completed_commands']) == len(result['commands']) > 0 and
            all(command.get('returncode') == 0 for command in result['completed_commands']),
            'actual official commands incomplete or nonzero')
    contracts = {}
    for key, controller_key in (('anatomy_contract', 'anatomy_contract_sha256'),
                                ('official_dwi_contract', 'dwi_contract_sha256')):
        entry = checked_record(result['source'][key])
        require(entry['sha256'] == row[controller_key], 'worker consumed another producer contract')
        contracts[key] = entry
    return {'actual_case_root': str(actual_root.resolve()), 'reference_manifest': identity,
            'producer_contracts': contracts, 'official_eddy_solver': result['raw_case_binding']['official_eddy_solver'],
            'raw_b0_selection': result['raw_case_binding']['raw_b0_selection'],
            'actual_command_count': len(result['completed_commands'])}


def publish(config_path, digest, output_root):
    config, identities, source_hashes = configuration(config_path, digest)
    output_root = Path(output_root)
    require(str(output_root.resolve()) not in
            {str(Path(row['output_root']).resolve()) for row in config['groups'].values()},
            'controlled view must be separate from actual group outputs')
    output_root.mkdir(parents=True, exist_ok=True)
    cases, groups = {}, {}
    for name, group in config['groups'].items():
        group_root = Path(group['output_root'])
        status = group_root / 'cohort_status.json'
        group_record = {'actual_output_root': str(group_root.resolve()), 'case_ids': group['case_ids'],
                        'controller_status': None, 'state': 'not_launched'}
        state = None
        if status.is_file():
            status_bytes = status.read_bytes()
            state = json.loads(status_bytes)
            launch = json.loads(Path(group['launch_configuration']['path']).read_text())
            for key in ('official_dwi_root', 'official_anatomy_root'):
                require(Path(state[key]).resolve() == Path(launch[key]).resolve(), 'actual controller consumes another producer root')
            require(set(state['cases']) == set(group['case_ids']) and state['workers'] == 1 and
                    state['tracking_threads'] == 0 and state['downstream_threads'] == 8 and
                    state['raw_manifest_sha256'] == config['raw_manifest']['sha256'] and
                    state['verified_reference_manifest_sha256'] == config['verified_reference_manifest']['sha256'],
                    'actual group controller case/source configuration differs')
            for key, basename in (('controller_sha256', 'run_connectome_raw_official_cohort.py'),
                                  ('worker_sha256', 'benchmark_connectome_raw_official.py'),
                                  ('helper_sha256', 'benchmark_connectome_repeats_official.py'),
                                  ('matrix_helper_sha256', 'connectome_repeat_common.py')):
                require(state[key] == source_hashes[basename], 'actual group controller source differs')
            group_record.update(state=state['state'], controller_pid=state['controller_pid'],
                controller_status={'path': str(status.resolve()), 'sha256': hashlib.sha256(status_bytes).hexdigest(),
                                   'policy': 'SHA of this exact atomic status snapshot; producer status may later advance'})
        groups[name] = group_record
        for case in group['case_ids']:
            row = state['cases'][case] if state is not None else {'state': 'not_launched'}
            item = {'group': name, 'state': row['state'], 'actual_output_root': str(group_root.resolve()),
                    'planned_case_path': str(group_root / case), 'controlled_link': None}
            if row['state'] == 'completed':
                item.update(completed_case(case, row, group_root, config, source_hashes))
                link, target = output_root / case, Path(item['actual_case_root'])
                if link.is_symlink():
                    require(link.resolve() == target, 'preexisting controlled link points to another actual output')
                else:
                    require(not link.exists(), 'preexisting real directory cannot represent a controlled case link')
                    link.symlink_to(target, target_is_directory=True)
                item['controlled_link'] = {'path': str(link), 'actual_target': str(target),
                                           'policy': 'symlink only; no copied or newly generated scientific output'}
            cases[case] = item
    failed = any(row['state'] == 'failed' for row in groups.values())
    complete = all(row['state'] == 'completed' for row in cases.values()) and not failed
    result = {'schema_version': 1, 'scope': 'actual origin binding only; no scientific calculations or parity claim',
              'state': 'completed' if complete else 'failed' if failed else 'partial_or_waiting', 'execution_completed': complete,
              'scientific_parity': 'not_assessed', 'source_identity': identities,
              'groups': groups, 'cases': cases, 'utc': datetime.now(timezone.utc).isoformat()}
    atomic(output_root / 'case_origin_binding.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=15.)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        parser.error('poll interval must be at least one second')
    while True:
        result = publish(args.config, args.config_sha256, args.output_root)
        if not args.watch or result['execution_completed'] or result['state'] == 'failed' or all(
                row['state'] in ('completed', 'failed') for row in result['cases'].values()):
            print(json.dumps({'state': result['state'], 'completed_cases':
                              [case for case, row in result['cases'].items() if row['state'] == 'completed']}))
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
