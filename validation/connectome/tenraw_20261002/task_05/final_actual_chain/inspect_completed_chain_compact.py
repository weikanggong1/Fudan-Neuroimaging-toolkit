"""Read-only compact audit of final metadata, not another numerical comparison."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import struct
import zlib

root = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), str(path)
    return json.loads(path.read_bytes())

def identity(path):
    return {'path': str(path), 'sha256': sha(path)}

def bound(item):
    value = read(item['path']); assert sha(item['path']) == item['sha256'], item['path']
    return value

config_path = root / 'formal_actual_mixed_waiter_tools_v1/configuration.json'
config = read(config_path)
for item in config['static_JSON_bindings'] + config['helper_files']:
    assert sha(item['path']) == item['sha256'], item['path']
launch = root / 'root_actual_mixed_final_CPU_chain_v1.launch.json'
chain_path = root / 'root_actual_mixed_final_CPU_chain_v1/status.json'
chain = read(chain_path)
result = {'scope': 'read-only final metadata audit; scientific comparisons are immutable original reader reports, not recomputed here',
          'chain_status': identity(chain_path), 'actual_status': chain['status'], 'launch': identity(launch),
          'configuration': identity(config_path), 'all_frozen_metadata_and_reader_bytes_match': True,
          'failed_v3_driver': chain['failed_v3_driver'], 'completed_subset_receipt': chain['completed_case_subset_receipt'],
          'GPU_or_MRI_started_by_audit': False}
if chain['status'] != 'completed_actual_ten_unique_comparison_summary_export':
    result.update(finalized=False, error=chain.get('error'), stage_status='pending_or_failed_actual_chain')
    print(json.dumps(result, indent=2)); raise SystemExit(0)

state_path = Path(config['comparison_report_dir']) / 'status.json'
summary_path = Path(config['summary_report_dir']) / 'summary.json'
export_path = root / 'root_actual_cohort_export_v3/status.json'
state, summary, export = read(state_path), read(summary_path), read(export_path)
assert state['status'] == 'completed_actual_ten_case_comparison' and state['completed_cases'] == state['anatomy_compared_cases'] == 10 and state['failed_cases'] == 0
assert summary['status'] == 'complete_actual_ten_case_tables' and summary['ready_for_ten_case_render'] is True and summary['completed_pairs'] == 10
assert export['status'] == 'completed_actual_ten_case_export' and export['completed_pairs'] == 10
assert summary['comparison_status_sha256'] == sha(state_path)
assert len(summary['matrix_rows']) == len({(x['case_id'], x['atlas'], x['kind']) for x in summary['matrix_rows']}) == 320
assert len(summary['anatomy_rows']) == len({(x['case_id'], x['file']) for x in summary['anatomy_rows']}) == 130
assert len(summary['source_rows']) == len({(x['case_id'], x['arm']) for x in summary['source_rows']}) == 20
expected = {'baseline': 'deefeb6908c3c14a9aa7b4cf154c8df941a56abffd4a4e893045a1c4d1ddfd4a',
            'candidate': '9fd44cbc49c9cdfc16c9a8cff2971fec3239b450dce41222e6ef059861eb0dc7'}
assert state['actual_source_fingerprints'] == expected
receipt = bound(chain['completed_case_subset_receipt'])
failed = bound(chain['failed_v3_driver'])
assert failed['status'] == 'selected_recovery_failed_or_ineligible' and receipt['failed_case_keys'] == ['candidate/sub-CON08']
assert len(receipt['completed_case_keys']) == 6 and len(receipt['not_dispatched_case_keys']) == 3
matrix_stats = {kind: [] for kind in ('count', 'sift2_fbc', 'mean_length', 'mean_fa')}
case_results = []; process_peaks = []; image_results = []
for case_id, record in state['cases'].items():
    assert record['status'] == 'completed_comparison'
    comparison = bound(record['connectome']); anatomy = bound(record['anatomy'])
    assert comparison['status'] == 'completed' and anatomy['status'] == 'completed' and len(anatomy['files']) == 13
    assert len(comparison['atlases']) == 8
    scopes = {}
    for arm in ('baseline', 'candidate'):
        evidence = comparison[arm]; budget = evidence['memory_budget']
        assert budget['status'] == 'observed_below_budget' and not budget['monitor_issues']
        assert all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x < 20_000_000_000 for x in budget['measurements'].values())
        process_peaks.append(budget['measurements']['process_tree'])
        assert evidence['actual_source']['source_fingerprint'] == expected[arm]
        scopes[arm] = {'actual_GPU_root': evidence['actual_GPU_root'], 'GPU_report': evidence['gpu_report'],
                       'wall_report': evidence['wall_report'], 'memory_budget': budget,
                       'raw_dwi_cli_seconds': evidence['raw_dwi_cli_total_runtime_seconds'],
                       'timing_scope': evidence['timing_scope']}
    for atlas, values in comparison['atlases'].items():
        assert set(values['matrices']) == set(matrix_stats)
        for kind, metrics in values['matrices'].items(): matrix_stats[kind].append(metrics)
        image_results.append(values['atlas_labels'])
    image_results.extend(comparison['images'].values())
    case_results.append({'case_id': case_id, 'connectome': record['connectome'], 'anatomy': record['anatomy'],
                         'FS_scientific_equal': anatomy['all_requested_scientific_data_equal'],
                         'count_exact_all_atlases': comparison['count_exact_all_atlases'],
                         'matrix_numeric_exact_all_atlases': comparison['matrix_numeric_exact_all_atlases'],
                         'node_counts': {k: v['node_count'] for k, v in comparison['atlases'].items()}, 'actual_evidence': scopes})
metrics = {}
for kind, values in matrix_stats.items():
    assert len(values) == 80
    metrics[kind] = {'checks': 80, 'exact_scientific_array_equal_count': sum(v['exact_scientific_array_equal'] for v in values),
                     'max_abs_error': max(v['max_abs_error'] for v in values), 'max_rmse': max(v['rmse'] for v in values),
                     'matrix_numeric_neq_total': sum(v['numeric_neq'] for v in values),
                     'raw_scalar_bits_neq_total': sum(v['raw_scalar_bits_neq'] for v in values)}
rows = summary['case_rows']; timing = {}
for arm in ('baseline', 'candidate'):
    values = [row[arm + '_raw_dwi_cli_total_runtime_seconds'] for row in rows]
    assert len(values) == 10 and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values)
    timing[arm] = {'raw_DWI_CLI_seconds_per_case': {row['case_id']: row[arm + '_raw_dwi_cli_total_runtime_seconds'] for row in rows},
                   'median_raw_DWI_CLI_seconds': statistics.median(values),
                   'scope': 'actual independent complete raw-DWI stage using same-round official FS; not continuous cold pipeline total; queue/worker/FS and prior attempts retained separately'}
export_files = []
for path in sorted(export_path.parent.rglob('*')):
    if path.is_file() and not path.is_symlink(): export_files.append({**identity(path), 'size_bytes': path.stat().st_size})
result.update(finalized=True, comparison_status=identity(state_path), summary=identity(summary_path), export_status=identity(export_path),
              unique_case_count=10, matrix_checks=320, FS_case_comparisons=10, FS_array_checks=130, source_rows=20,
              source_fingerprints=expected, metrics=metrics, cases=case_results, timing=timing,
              strict_count_equal_cases=sum(c['count_exact_all_atlases'] for c in case_results),
              matrix_numeric_exact_cases=sum(c['matrix_numeric_exact_all_atlases'] for c in case_results),
              all_independent_FS_scientific_equal=all(c['FS_scientific_equal'] for c in case_results),
              image_checks=len(image_results), strict_scientific_equal_image_checks=sum(x['strict_scientific_equal'] for x in image_results),
              max_process_tree_bytes=max(process_peaks), export_files=export_files,
              acceptance_scope='completed actual comparison coverage; MRtrix repeat-envelope acceptance remains separate; no pipeline speedup or cold full-wall inference')
origin_path = root / 'formal_actual_mixed_GPU_origins_v1/origins.json'
origins = read(origin_path)['bindings']
assert len(origins) == len({(x['case_id'], x['arm']) for x in origins}) == 10
origin_keys = {(x['case_id'], x['arm']) for x in origins}
source_ledger = []
for row in summary['source_rows']:
    case = next(x for x in case_results if x['case_id'] == row['case_id'])
    evidence = case['actual_evidence'][row['arm']]
    assert row['source_fingerprint'] == expected[row['arm']]
    assert row['actual_GPU_root'] == evidence['actual_GPU_root']
    for key, alias in [('GPU_report', 'GPU_report'), ('wall_report', 'wall_report')]:
        bound(evidence[alias])
        assert row[key + '_sha256'] == evidence[alias]['sha256']
    assert ((row['case_id'], row['arm']) in origin_keys) == (row['GPU_origin_binding'] is not None)
    source_ledger.append({k: row[k] for k in ('case_id', 'arm', 'source_fingerprint', 'actual_GPU_root', 'actual_GPU_driver', 'GPU_report_sha256', 'wall_report_sha256')})

def check_origin_bindings(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            assert sha(value['path']) == value['sha256'], value['path']
        for child in value.values(): check_origin_bindings(child)
    elif isinstance(value, list):
        for child in value: check_origin_bindings(child)
check_origin_bindings(origins)
png_metadata = []
for item in export_files:
    if not item['path'].endswith('.png'): continue
    data = Path(item['path']).read_bytes(); assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos = 8; chunks = []; header = None
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos+4])[0]; kind = data[pos+4:pos+8]
        payload = data[pos+8:pos+8+length]; crc = struct.unpack('>I',data[pos+8+length:pos+12+length])[0]
        assert zlib.crc32(kind + payload) & 0xffffffff == crc
        chunks.append(kind.decode('ascii'))
        if kind == b'IHDR': header = struct.unpack('>IIBBBBB', payload)
        pos += 12 + length
    assert pos == len(data) and chunks[-1] == 'IEND' and header and header[0] > 0 and header[1] > 0
    png_metadata.append({**item, 'width': header[0], 'height': header[1], 'bit_depth': header[2], 'color_type': header[3], 'all_chunk_CRCs_valid': True})
assert len(png_metadata) == 5
result.update(actual_mixed_origins=identity(origin_path), recovered_origin_count=10,
              actual_source_report_ledger=source_ledger, all_20_GPU_wall_report_SHA_bindings_match=True,
              all_10_origin_recursive_SHA_bindings_match=True, plotting_PNG_metadata=png_metadata)
print(json.dumps(result, indent=2, allow_nan=False))
