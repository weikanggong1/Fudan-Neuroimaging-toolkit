"""读取已有GCA收据和有序LTA，不执行注册或MRI算法。"""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(left, right):
    from fnit.recon_all.ca_normalize_python import read_voxel_lta
    a, b = read_voxel_lta(left), read_voxel_lta(right)
    delta = np.abs(a - b)
    return {'different_elements': int(np.count_nonzero(delta)),
            'max_abs_error': float(delta.max()), 'p99_abs_error': float(np.percentile(delta, 99)),
            'left_matrix': a.tolist(), 'right_matrix': b.tolist(),
            'left_sha256': sha(left), 'right_sha256': sha(right),
            'complete_file_bytes_equal': Path(left).read_bytes() == Path(right).read_bytes(),
            'units': 'type-0 source voxel to atlas voxel; linear entries dimensionless, translation in target voxels; not millimetres'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    base = args.root / 'runs/diagnose_python_gca_isolation_20261010_v1'
    producer = args.root / 'runs/recon_python_profile_20261010_v3'
    output = {'scope': 'same-input complete self-owned GCA stages; shared load, not ABBA, not raw-T1 end-to-end speed or official equivalence',
              'overall_metric_equivalence': 'not_assessed', 'subjects': {},
              'sampler_memory_note': 'namespace process ownership null stays null; total card peaks include other jobs; parent allocator counters disabled, not zero memory',
              'comparison_script_sha256': sha(__file__),
              'provenance_note': 'Stage-start live benchmark SHA is preserved, but historical full bytes were not captured before periodic updates. Immutable source manifest, executed modules and input bytes are independently verified. A changed live benchmark snapshot is not treated as an algorithm failure or a matching historical SHA.'}
    for subject in ['sub06', 'sub07']:
        old_benchmark = producer / (subject + '-candidate/benchmark.json')
        # Preserve still-live producer metadata as an immutable diagnostic snapshot.
        metadata_copy = base / (subject + '-producer-benchmark.private.json')
        if not metadata_copy.exists():
            metadata_copy.write_bytes(old_benchmark.read_bytes())
        rows = {}
        for config in ['cpu64', 'torch1024']:
            for number in [1, 2]:
                folder = base / f'{subject}-{config}-r{number}'
                if (folder / 'diagnosis.private.json').is_file():
                    receipt = json.loads((folder / 'diagnosis.private.json').read_text())
                    manifest_file = args.root / 'workspaces/recon_python_profile_20261010_v3/SOURCE_MANIFEST.json'
                    manifest = json.loads(manifest_file.read_text())
                    snapshot = json.loads(metadata_copy.read_text())
                    current_checks = {}
                    for name, info in receipt['inputs'].items():
                        copied = folder / 'subject/mri' / name
                        produced = Path(info['producer_path'])
                        current_checks[name] = {
                            'recorded_sha256': info['sha256'],
                            'copied_sha256': sha(copied), 'producer_current_sha256': sha(produced),
                            'copied_matches_recorded': sha(copied) == info['sha256'],
                            'producer_current_matches_recorded': sha(produced) == info['sha256']}
                    module_checks = {}
                    for module_name, recorded_sha in receipt['source_sha256'].items():
                        relative = 'src/' + module_name.replace('.', '/') + '.py'
                        module_checks[module_name] = {
                            'recorded_sha256': recorded_sha,
                            'manifest_sha256': manifest['files'][relative]['sha256'],
                            'actual_current_sha256': sha(args.root / 'workspaces/recon_python_profile_20261010_v3' / relative)}
                    rows[f'{config}-r{number}'] = {'status': receipt['status'], 'receipt_sha256': sha(folder / 'diagnosis.private.json'),
                        'producer_benchmark_sha256_recorded_at_stage_start': receipt['producer_benchmark_sha256'],
                        'producer_benchmark_poststage_snapshot_sha256': sha(metadata_copy),
                        'producer_benchmark_snapshot_sha_matches': receipt['producer_benchmark_sha256'] == sha(metadata_copy),
                        'producer_benchmark_historical_full_bytes_available': False,
                        'producer_benchmark_raw_sha_status': 'historical_raw_snapshot_unavailable_live_periodic_updates',
                        'producer_fixed_configuration_matches_snapshot':
                            receipt['producer_code_version'] == snapshot['code_version'] and
                            receipt['producer_gca_configuration'] == {key: snapshot[key] for key in receipt['producer_gca_configuration']},
                        'producer_source_manifest_current_sha_matches': receipt['producer_source_manifest_sha256'] == sha(manifest_file),
                        'producer_source_manifest_semantics_match': receipt['producer_source_version'] == manifest,
                        'executed_module_checks': module_checks,
                        'executed_module_sha_verified': all(row['recorded_sha256'] == row['manifest_sha256'] == row['actual_current_sha256'] for row in module_checks.values()),
                        'input_hash_checks': current_checks,
                        'input_hashes_verified': all(row['copied_matches_recorded'] and row['producer_current_matches_recorded'] for row in current_checks.values()),
                        'wall_seconds': receipt.get('full_call_seconds'),
                        'worker_timing': receipt.get('worker', {}).get('timing'),
                        'scoring_profile': receipt.get('worker', {}).get('scoring_profile'),
                        'source_sha256': receipt.get('source_sha256'), 'inputs': receipt['inputs'],
                        'source_manifest_sha256': receipt['producer_source_manifest_sha256'],
                        'parent_state_unchanged': receipt.get('parent_state_unchanged'),
                        'parent_live_tensor_unchanged': receipt.get('parent_live_tensor_unchanged'),
                        'child_allocated_peak_bytes': receipt.get('worker', {}).get('allocated_peak_bytes'),
                        'child_reserved_peak_bytes': receipt.get('worker', {}).get('reserved_peak_bytes'),
                        'memory_status': receipt.get('memory', {}).get('status'),
                        'peak_tree_total_bytes': receipt.get('memory', {}).get('peak_tree_total_bytes'),
                        'peak_target_device_used_bytes': receipt.get('memory', {}).get('peak_target_device_used_bytes')}
        comparisons = {}
        baseline = base / f'{subject}-cpu64-r1/subject/mri/transforms/talairach.lta'
        if baseline.exists():
            for config in ['cpu64', 'torch1024']:
                for number in [1, 2]:
                    name = f'{config}-r{number}'
                    other = base / f'{subject}-{name}/subject/mri/transforms/talairach.lta'
                    if other.is_file() and rows.get(name, {}).get('status') == 'complete':
                        comparisons[name + '-vs-cpu64-r1'] = compare(baseline, other)
            control = producer / f'{subject}-candidate/subject/mri/transforms/talairach.lta'
            if control.exists():
                try:
                    comparisons['isolated-vs-v3-inprocess'] = compare(control, baseline)
                except (ValueError, IndexError) as error:
                    comparisons['isolated-vs-v3-inprocess'] = {'status': 'pending_partial_lta', 'error': repr(error)}
            else:
                comparisons['isolated-vs-v3-inprocess'] = {'status': 'pending_v3_lta'}
        medians = {}
        for config in ['cpu64', 'torch1024']:
            seconds = [row['wall_seconds'] for name, row in rows.items() if name.startswith(config) and row['status'] == 'complete']
            medians[config] = {'completed': len(seconds), 'wall_seconds': seconds,
                               'median_seconds': statistics.median(seconds) if seconds else None}
        output['subjects'][subject] = {'stages': rows, 'comparisons': comparisons, 'observed_times': medians}
    count = sum(row['status'] == 'complete' for sub in output['subjects'].values() for row in sub['stages'].values())
    output['complete_stage_count'] = count
    output['status'] = 'complete_stages' if count == 8 else 'partial_stages'
    (base / 'summary.provenance_v2.private.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({'status': output['status'], 'complete_stage_count': count,
                      'times': {key: value['observed_times'] for key, value in output['subjects'].items()},
                      'provenance': {key: {name: {field: row[field] for field in ['producer_benchmark_snapshot_sha_matches', 'producer_fixed_configuration_matches_snapshot', 'producer_source_manifest_current_sha_matches', 'producer_source_manifest_semantics_match', 'executed_module_sha_verified', 'input_hashes_verified']} for name,row in value['stages'].items()} for key,value in output['subjects'].items()},
                      'matrix_differences': {key: {k: v.get('different_elements', v.get('status')) for k, v in value['comparisons'].items()} for key, value in output['subjects'].items()}}, indent=2))


if __name__ == '__main__':
    main()
