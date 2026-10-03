"""Read-only completed CON11 audit; no image generation, tracking, or solver."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def record(path):
    path = Path(path)
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha(path)}


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--launch-sha256', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    verified = {}

    def bound(row):
        path = Path(row['path'])
        if str(path) not in verified:
            actual = record(path)
            require(actual['sha256'] == row['sha256'], 'actual bound file changed: ' + str(path))
            require('size_bytes' not in row or actual['size_bytes'] == row['size_bytes'], 'actual bound size changed')
            verified[str(path)] = actual
        else:
            require(verified[str(path)]['sha256'] == row['sha256'], 'conflicting declared digest')
        return path

    def read(row):
        return json.loads(bound(row).read_text())

    config = read({'path': str(args.config), 'sha256': args.config_sha256})
    launch = read({'path': str(args.launch), 'sha256': args.launch_sha256})
    require(config['GPU'] is False and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only actual audit required')
    require(launch['configuration']['sha256'] == args.config_sha256
            and launch['source']['sha256'] == config['orchestrator_sha256'], 'actual launch source/config differ')
    bound(launch['source']); bound(launch['launcher'])
    for row in list(config['frozen_sources'].values()) + list(config['known_inputs'].values()):
        bound(row)
    state_root = Path(config['state_root'])
    state = json.loads((state_root / 'status.json').read_text())
    require(state['state'] == 'completed' and state['anatomy_completed'] and state['reference_completed']
            and not state['prepare_rerun'] and not state['old_queue_modified'], 'actual whole follow-on not completed')
    handoff = json.loads((state_root / 'completed_handoff.json').read_text())
    require(handoff['state'] == 'completed' and handoff['case_id'] == config['case_id'], 'completed same-case handoff required')
    for row in [value for value in handoff.values() if isinstance(value, dict) and 'path' in value and 'sha256' in value]:
        bound(row)
    model = read(handoff['model_consumer'])
    anatomy = read(handoff['anatomy_consumer'])
    require(model['case_id'] == anatomy['case_id'] == 'sub-CON11'
            and model['execution_completed'] and model['state'] == anatomy['state'] == 'completed', 'same-case official producer required')
    require(anatomy['official_dwi_contract'] == handoff['model_consumer']
            and anatomy['prepared_report'] == config['known_inputs']['prepared_report'], 'actual original prepare/new model bindings differ')
    for row in model['files'].values():
        bound(row)
    prepared = read(anatomy['prepared_report'])
    for row in prepared['outputs'].values():
        bound(row)
    report = read(anatomy['official_anatomy_report'])
    require(len(prepared['outputs']) == 27 and len(report['outputs']) == 20
            and report['execution_completed'] and all(row['returncode'] == 0 for row in report['commands']), 'actual anatomy output/exit differs')
    for row in report['outputs'].values():
        bound(row)
    reference_config = read(handoff['reference_controller_configuration'])
    controller = read(handoff['reference_controller_status'])
    manifest = read(handoff['reference_manifest'])
    expected_roles = {'controller_sha256': 'reference_controller', 'worker_sha256': 'reference_worker',
                      'helper_sha256': 'reference_helper', 'matrix_helper_sha256': 'matrix_helper'}
    for field, role in expected_roles.items():
        require(controller[field] == config['frozen_sources'][role]['sha256'], 'original reference runtime differs: ' + role)
    require(controller['state'] == 'completed' and controller['execution_completed']
            and set(controller['cases']) == {'sub-CON11'} and controller['cases']['sub-CON11']['returncode'] == 0,
            'actual original reference controller not completed')
    require(reference_config['n_seeds'] == 100000 and reference_config['seeds'] == [0, 1, 2, 3, 4]
            and manifest['seeds'] == [0, 1, 2, 3, 4] and manifest['execution_completed'], 'original five-repeat workload differs')
    require(manifest['raw_case_binding']['rawprep_canonical_dwi_coverage_verified'] is True
            and len(manifest['input_readbacks']) == 13 and len(manifest['source']['profiles']) == 8,
            'actual raw/input/atlas lineage incomplete')
    require(len(manifest['completed_commands']) == len(manifest['commands'])
            and all(row['returncode'] == 0 for row in manifest['completed_commands']), 'official reference command failed')
    for row in manifest['programs'].values():
        bound(row)
    common = config['frozen_sources']['matrix_helper']
    path = bound(common)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('frozen_CON11_matrix_reader', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    reference_root = Path(config['reference_output'])
    repeats = {}
    for seed in config['reference']['seeds']:
        directory = reference_root / f'seed-{seed}'
        saved = manifest['outputs'][str(seed)]
        bound({'path': str(directory / 'tracks.tck'), 'sha256': saved['tracks_sha256']})
        for name, row in saved['scalars'].items():
            bound({'path': str(directory / name), 'sha256': row['sha256']})
        actual = helper.load_profiles(directory)
        require(set(actual) == set(anatomy['atlases']), 'actual complete eight atlas matrices required')
        for name, (_, meta) in actual.items():
            require(meta == saved['profiles'][name] and meta['nodes'] == anatomy['atlases'][name]['n_nodes'],
                    'matrix SHA, node semantics, or shape changed: ' + name)
        repeats[str(seed)] = {'accepted_tracks': saved['accepted_tracks'], 'attempted_seeds': saved['attempted_seeds'],
                             'scalar_nonfinite_counts': {name: row['nonfinite_count'] for name, row in saved['scalars'].items()},
                             'atlases': {name: {'nodes': meta['nodes'], 'matrix_sha256': meta['matrix_sha256']} for name, (_, meta) in actual.items()}}
    for phase in ['anatomy', 'reference']:
        exit_report = json.loads((state_root / (phase + '_exit.json')).read_text())
        require(exit_report['returncode'] == 0 and exit_report['GPU'] is False, 'actual CPU phase failed')
        bound(exit_report['configuration']); bound(exit_report['orchestrator'])
    print(json.dumps({'schema_version': 1, 'case_id': 'sub-CON11', 'state': 'actual_completed_followon_readonly_verified',
        'observed_UTC': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(), 'GPU_used': False,
        'configuration': record(args.config), 'launch': record(args.launch), 'handoff': record(state_root / 'completed_handoff.json'),
        'actual_verified_file_count': len(verified), 'actual_verified_bytes': sum(row['size_bytes'] for row in verified.values()),
        'prepared_outputs_reused': 27, 'anatomy_complete_outputs': 20,
        'official_reference_commands_exit_zero': len(manifest['completed_commands']),
        'reference_total_wall_seconds': manifest['total_wall_seconds'], 'repeats': repeats,
        'scientific_runtime_unchanged': expected_roles, 'old_namespace_modified': False,
        'timing_scope': state['timing_scope'], 'audit_wall_seconds': time.perf_counter() - started,
        'scientific_parity': 'not_assessed_by_metadata_audit', 'verified_files': verified}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
