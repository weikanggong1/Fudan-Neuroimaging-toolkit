"""Publish saved one-call object API and preprocessing controls anonymously."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(value):
    keys = ('status', 'returncode', 'wall_seconds', 'cpu_affinity', 'max_cpu_threads',
            'started_utc', 'finished_utc', 'load_before', 'load_after',
            'maximum_sampled_tree_rss_bytes', 'maximum_sampled_tree_threads')
    return {k: value[k] for k in keys if k in value}


def anonymize(value):
    if isinstance(value, dict):
        return {key: anonymize(item) for key, item in value.items()
                if key not in ('job', 'argv', 'environment', 'source_root', 'outputs') or
                key == 'outputs' and isinstance(item, dict)}
    if isinstance(value, list):
        return [anonymize(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        return Path(value).name
    return value


def image_gate(row):
    result = {}
    for name in ('whole_grid', 'official_synthstrip_brain', 'upper_coordinate_boundary_band'):
        metrics = row[name]
        nrmse = metrics['nrmse_reference_p99_minus_p1']
        result[name] = (metrics['max_abs'] == 0 if nrmse is None else nrmse <= 1e-3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'posthoc', 'control', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    results, posthoc, control = [json.loads(path.read_text()) for path in (args.results, args.posthoc, args.control)]
    for row in [*results['receipts'].values(), posthoc['receipt'], control['receipt']]:
        if row['status'] != 'complete' or row['returncode'] != 0:
            raise ValueError('unfinished verification or posthoc stage')
    if results['api']['status'] != 'complete':
        raise ValueError('unfinished object call')
    gates = {
        'official_world_transform_max_vector_mm_1e-3': {
            name: row['corner_world_error_mm']['max'] <= 1e-3
            for name, row in results['official']['transforms'].items()},
        'official_images_nrmse_p99_minus_p1_1e-3_or_exact_zero_range': {
            name: image_gate(row) for name, row in results['official']['images'].items()},
        'same_candidate_exact_images': {
            name: row['whole_grid']['exact_equal'] for name, row in results['same_candidate']['images'].items()},
        'same_candidate_exact_matrices': {
            name: row['matrix']['exact_equal'] for name, row in results['same_candidate']['transforms'].items()},
        'direct_float32_preprocessing_control_only': {
            name: {'tensor_equal': row['path_direct_tensor_equal'],
                   'normalized_network_input_equal': row['path_direct_normalized_equal']}
            for name, row in control['report']['inputs'].items()},
        'direct_float32_complete_network_output_assessed': False,
    }
    report = {
        'schema_version': 1,
        'scope': 'one real CPU affine256 full bidirectional materialized SpatialImage API call, then read-only controls',
        'coverage': {'model': 'affine', 'extent': 256, 'compute_inverse': True,
                     'other_modes_or_object_parameter_branches': 'not_run',
                     'original_reference_or_path_network_rerun': False, 'GPU_used': False},
        'canonical_index_sha256': results['canonical_index_sha256'],
        'api': anonymize(results['api']),
        'process_receipts': {key: receipt(row) for key, row in results['receipts'].items()},
        'existing_reference_receipt_sha256': results['official_existing_receipt_sha256'],
        'existing_path_candidate_receipt_sha256': results['candidate_existing_receipt_sha256'],
        'same_candidate': anonymize(results['same_candidate']),
        'official': anonymize(results['official']),
        'posthoc_saved_headers_and_preprocessing': {'report': posthoc['report'], 'receipt': receipt(posthoc['receipt']),
                                                  'private_report_sha256': posthoc['report_sha256']},
        'direct_float32_preprocessing_control': {'report': control['report'], 'receipt': receipt(control['receipt']),
                                               'private_report_sha256': control['report_sha256']},
        'gates': gates,
        'claims': {'same_path_candidate_bitwise_output_equal': False,
                   'all_official_fixed_gates_passed': False, 'cold_CLI_speedup': None,
                   'direct_float32_preprocessing_exact': True,
                   'direct_float32_end_to_end_output_equivalence': 'not_assessed'},
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'report.public.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    manifest = {
        'schema_version': 1,
        'private_inputs_sha256': {path.name: digest(path) for path in (args.results, args.posthoc, args.control)},
        'public_files_sha256': {path.name: digest(path) for path in sorted(output.iterdir())
                                if path.is_file() and path.name != 'manifest.public.json' and
                                path.suffix in ('.py', '.json', '.md')},
    }
    (output / 'manifest.public.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'published', 'official_all_gates_passed': False,
                      'path_bitwise_outputs_equal': False, 'no_CNN_direct_control': True}))


if __name__ == '__main__':
    main()
