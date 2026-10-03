#!/usr/bin/env python3
"""Independent stdlib, read-only metadata review; no scientific statistics run."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tarfile

ACTUAL = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_final_raw_task04_20261003_v7')
TOOL_SHA = '9be317df1bc38fdaf993ccb16b29e52f902c681ecbf80a3d85c08521ef4ee235'
RAW_SHA = '120a3e1a431ad23177bc9a4a7c17291e52e374085e0fe7314115b656de8db676'
TASK4_COMMIT = '177db9d0db52d0663f50d37fa5509cca57a3cbbf'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    path = Path(path)
    content = path.read_bytes()
    return {'path': str(path), 'size_bytes': len(content), 'sha256': digest(content)}


def load(path):
    return json.loads(Path(path).read_bytes())


def actual_review(root):
    config = load(root / 'configuration.json')
    ready = load(root / 'observation_000000.json')
    snapshot = load(root / 'immutable_input_snapshot.json')
    final = load(root / 'final_report.json')
    mixed = load(config['mixed_configuration']['path'])
    canonical = load(config['raw_manifest']['path'])
    tool = next(Path(item['path']) for item in config['source_files']
                if item['path'].endswith('/final_raw_matrix_tools_v7/benchmark_connectome_final_raw_envelope.py'))
    require(identity(tool)['sha256'] == TOOL_SHA, 'actual frozen metadata tool changed')
    raw = Path(config['raw_tool'])
    require(identity(raw)['sha256'] == RAW_SHA and snapshot[str(raw)] == RAW_SHA,
            'original numerical comparison bytes changed or omitted')
    required = ready['required_file_sha256']
    require(len(required) == 1480 and all(snapshot.get(path) == sha for path, sha in required.items()),
            'required output ledger is missing terminal SHA bindings')
    require(len(snapshot) == 9099 and len(ready['chains']) == 20, 'unexpected actual qualification scope')
    source_files = {}
    for chain in ready['chains']:
        require(len(chain['required_file_sha256']) == 74, 'unexpected required per-chain output scope')
        source = chain['qualified_origin']['actual_source']
        for relative, sha in source['source_sha256'].items():
            path = str((Path(source['directory']) / relative).resolve())
            require(path not in source_files or source_files[path] == sha, 'conflicting source identity')
            source_files[path] = sha
    require(len(source_files) == 2388, 'unexpected two-tree source scope')
    for path, sha in source_files.items():
        require(snapshot.get(path) == sha and identity(path)['sha256'] == sha, 'scientific source bytes changed')
    raw_files = {str(Path(item['path']).resolve()): item['sha256']
                 for case in canonical['cases'] for item in case['input_files']}
    require(len(raw_files) == 101 and all(snapshot.get(path) == sha for path, sha in raw_files.items()),
            'canonical raw SHA ledger omitted from terminal snapshot')
    static = {item['path']: item['sha256'] for item in mixed['static_JSON_bindings']}
    history = config['historical_observation_JSONs']
    require(len(history) == 7, 'unexpected historical exemption set')
    historical_proofs = []
    for item in history:
        require(set(item) == {'path', 'sha256', 'role'} and
                item['role'] == 'historical_prior_comparison_observation', 'invalid historical role')
        path, sha = item['path'], item['sha256']
        require(static.get(path) == snapshot.get(path) == identity(path)['sha256'] == sha,
                'historical file bytes are not exactly frozen')
        historical_proofs.append(item)
    skipped = []
    def scan(value, context):
        if isinstance(value, dict):
            if all(key in value for key in ('content', 'policy', 'path')):
                skipped.append({'context': context, 'path': value['path'], 'policy': value['policy']})
                return
            for key, child in value.items():
                scan(child, context + '/' + str(key))
        elif isinstance(value, list):
            for key, child in enumerate(value):
                scan(child, context + '/' + str(key))
    scan(ready, 'ready')
    audit = ready['official_origin_audit']
    observations = audit['current_controller_observations']
    controller_paths = {item['path'] for item in observations}
    require(len(skipped) == 6 and {item['path'] for item in skipped} == controller_paths,
            'mutable observation exception unexpectedly covers other data')
    owners = {case: item['controller_binding'] for case, item in audit['explicit_origin_map']['cases'].items()}
    fixed = ('scope', 'raw_manifest_sha256', 'dataset', 'snapshot', 'official_dwi_root',
             'official_anatomy_root', 'workers', 'downstream_threads', 'tracking_threads',
             'controller_sha256', 'worker_sha256', 'helper_sha256', 'matrix_helper_sha256',
             'verified_reference_manifest_sha256')
    controller_proofs = []
    for observation in observations:
        path = Path(observation['path'])
        before, current = observation['content'], load(path)
        require(all(current.get(key) == before.get(key) for key in fixed) and
                set(current['cases']) == set(before['cases']), 'official controller workload changed')
        selected = []
        other = {}
        for case, row in current['cases'].items():
            if owners.get(case) == observation['controller_binding']:
                require(row == before['cases'][case] and row['state'] == 'completed', 'selected row changed')
                selected.append(case)
            else:
                require(row['state'] not in ('running', 'completed'), 'reassigned case was dispatched')
                other[case] = row['state']
        end = next(item for item in final['official_controller_end_observations'] if item['path'] == str(path))
        require(end['selected_completed_rows_equal'] and end['reassigned_case_not_dispatched'],
                'actual terminal controller verification missing')
        controller_proofs.append({'path': str(path), 'controller_binding': observation['controller_binding'],
                                 'selected_completed_rows_equal': True, 'selected_cases': selected,
                                 'unselected_states': other, 'ready_sha256': observation['sha256'],
                                 'recorded_terminal_sha256': end['after_sha256'],
                                 'review_sha256': identity(path)['sha256']})
    require(len(final['runs']) == 20 and sum(item['accepted'] for item in final['runs']) == 2782 and
            sum(item['total'] for item in final['runs']) == 4800 and
            all(item['matrix_envelope_status'] == 'failed' for item in final['runs']) and
            final['full_ten_scientific_match'] is False and final['FNIT_self'] == 'not_assessed' and
            final['population'] == 'not_assessed' and final['MRI_or_GPU_started'] is False,
            'final scientific-scope conclusion changed')
    return {'review_kind': 'independent_readonly_metadata_and_source_byte_review',
            'host': socket.gethostname(), 'review_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'MRI_or_GPU_started': False, 'scientific_statistics_recomputed': False,
            'records': {name: identity(root / name) for name in
                        ('configuration.json', 'observation_000000.json', 'immutable_input_snapshot.json', 'final_report.json')},
            'frozen_reader': identity(tool), 'original_numerical_reader': identity(raw),
            'coverage': {'snapshot_files': len(snapshot), 'required_output_path_SHA_equal': len(required),
                         'canonical_unique_raw_path_SHA_equal': len(raw_files),
                         'scientific_source_files_independently_rehashed': len(source_files),
                         'source_JSON_files_byte_bound': sum(path.endswith('.json') for path in source_files)},
            'coverage_note': 'raw/output ledger coverage compared; large MRI output contents were not rehashed again by this review. The actual reader completed both terminal full-snapshot SHA checks before final_report.',
            'historical_JSON_exact_byte_proofs': historical_proofs,
            'mutable_exception_actual_records': skipped, 'controller_independent_recheck': controller_proofs,
            'namespace_guards': {'initial': 'canonical raw resolved parents + selected actual roots + official actual producers',
                                 'ready': 'all snapshot files + qualified job/root/source + resolved official producer/contracts',
                                 'separate_actual_output_root': str(root)},
            'scientific_result_observed': {'completed': 20, 'failed': 20, 'accepted': 2782, 'total': 4800,
                                           'full_ten_scientific_match': False, 'FNIT_self': 'not_assessed', 'population': 'not_assessed'}}


def delivery_review(repository):
    repository = Path(repository)
    commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == TASK4_COMMIT, 'Task4 final delivery commit changed')
    tool = subprocess.check_output(['git', '-C', str(repository), 'show',
                                   commit + ':tools/reference/benchmark_connectome_final_raw_envelope.py'])
    require(digest(tool) == TOOL_SHA, 'committed reader differs from actual frozen reader')
    folder = repository / 'validation/connectome/tenraw_20261002/task_04_final_raw_matrix_results_v7'
    index = load(folder / 'evidence_index.json')
    require(len(index['files']) == 122, 'unexpected archive payload count')
    with tarfile.open(folder / 'evidence.tar.gz') as archive:
        for item in index['files']:
            member = archive.getmember(item['archive_path'])
            content = archive.extractfile(member).read()
            require(member.isfile() and len(content) == item['size_bytes'] and digest(content) == item['sha256'],
                    'archive member differs from exact index: ' + item['archive_path'])
        extra = sorted({item.name for item in archive.getmembers() if item.isfile()} -
                       {item['archive_path'] for item in index['files']})
        require(extra == ['evidence_index.json'], 'unindexed payload in evidence archive')
        require(archive.extractfile('evidence_index.json').read() == (folder / 'evidence_index.json').read_bytes(),
                'internal archive index differs')
    return {'review_kind': 'independent_exact_commit_and_archive_bytes_review',
            'review_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'task4_commit': commit,
            'committed_reader_sha256': digest(tool), 'matches_actual_frozen_reader': True,
            'archive': identity(folder / 'evidence.tar.gz'), 'archive_index': identity(folder / 'evidence_index.json'),
            'archive_payload_members_verified': len(index['files']), 'additional_member': extra,
            'delivered_final_report': identity(folder / 'final_report.json'),
            'MRI_or_GPU_started': False, 'scientific_statistics_recomputed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-root', type=Path, default=ACTUAL)
    parser.add_argument('--delivery-repository', type=Path)
    args = parser.parse_args()
    result = delivery_review(args.delivery_repository) if args.delivery_repository else actual_review(args.actual_root)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
