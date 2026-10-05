"""Posthoc complete180/MNI/91k comparison with independently bound late provenance."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
from continuous_closure import Audit, BINDINGS_SHA, ORDER, checked_case, fresh_output, load, require, sha256

METRIC_SHA = 'cb04445f6acf4252bf067500e70d53ec2981ee3ce9a24d9f47365c6ec12737d1'
ENDPOINT_SHA = 'dedd26ec627d709d447bfb5522c86fbabea6ca7f0d49ddd06ccd2671d31a661c'
CLOSURE_SHA = '8217e7281b9774bda964167a2f6c4081e9cd04d1d8815463ac998b3c6c9d069e'


def entry(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'sha256': sha256(p)}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def queue_check(queue):
    require(queue['order'] == ORDER and queue['physical_gpu'] == 0, 'wrong paired queue identity')
    require(queue['cold_reconstruction_excluded'] is True and queue['MSM_estimation_excluded'] is True,
            'wrong queue processing scope')
    require(queue['free_memory_admission_mib'] == 20480, 'wrong memory admission policy')


def save_endpoint_result(result, endpoint_target, record, failure_path):
    """Preserve a rejected preflight even when the frozen endpoint creates no output."""
    record.update(status=result['status'], input_guards_equal=result['input_guards_equal'])
    comparison_path = endpoint_target / 'comparison.public.json'
    if comparison_path.is_file():
        record['comparison'] = entry(comparison_path)
    else:
        require(result['status'] == 'validation_failed', 'completed endpoint did not save its report')
        write(failure_path, result)
        record['preflight_failure'] = entry(failure_path)
        record['failure'] = result.get('failure')
        record['endpoint_output_created'] = False
    return record


def completion_code(records):
    if any(x['status'] not in ('pending', 'comparison_complete', 'provenance_preflight_passed') for x in records):
        return 2
    return 0 if all(x['status'] in ('comparison_complete', 'provenance_preflight_passed') for x in records) else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--metric-helper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true', help='Validate saved metadata and hashes; do not calculate endpoint metrics.')
    args = parser.parse_args()
    started = time.perf_counter()
    audit = Audit()
    bindings = load(audit.check({'path': str(args.bindings), 'sha256': BINDINGS_SHA}, 'original_bindings'))
    audit.check({'path': str(args.metric_helper), 'sha256': METRIC_SHA}, 'metric_helper')
    endpoint = Path(__file__).with_name('endpoint_compare.py')
    audit.check({'path': str(endpoint), 'sha256': ENDPOINT_SHA}, 'endpoint_producer')
    closure = Path(__file__).with_name('continuous_closure.py')
    audit.check({'path': str(closure), 'sha256': CLOSURE_SHA}, 'late_closure_producer')
    audit.snapshot(__file__, 'collector_producer')
    runs = args.runs.resolve(strict=True)
    protected = [runs, Path(__file__).parent, Path(args.bindings).parent, Path(args.metric_helper).parent,
                 bindings['raw_bids_root'], bindings['fnit_root'], bindings['reference_root'], bindings['fnit_source_root']]
    # The frozen four manifests exist before launch. Read protection roots before mkdir.
    for case, variant in ORDER:
        manifest = load(runs / 'manifests' / (case + '.' + variant + '.private.json'))
        protected += [manifest['source_root'], Path(manifest['validation_helper']['path']).parent,
                      manifest['configuration']['hcp_assets_dir'], manifest['configuration']['recon_all']]
        protected += [Path(item['path']).parent for item in manifest['input_files'].values()]
    target = fresh_output(args.output, protected)
    queue_path = runs / 'queue.private.json'
    queue_bytes = queue_path.read_bytes()
    queue = json.loads(queue_bytes)
    queue_check(queue)
    queue_sha_before = hashlib.sha256(queue_bytes).hexdigest()
    verified, records = {}, []
    for case in ['CON01', 'CON06']:
        for variant in ['middle', 'robust']:
            item = checked_case(case, variant, runs, bindings, queue, audit)
            if item is None:
                records.append({'case_id': case, 'variant': variant, 'status': 'pending'})
            elif item['status'] != 'ready':
                records.append({'case_id': case, 'variant': variant, 'status': item['status'], 'report': entry(item['report_path'])})
            else:
                verified[(case, variant)] = item
    audit.finish()
    target.mkdir(parents=True, exist_ok=False)
    write(target / 'queue.snapshot.private.json', queue)
    (target / 'manifests').mkdir()
    (target / 'endpoints').mkdir()
    if not args.preflight_only:
        from endpoint_compare import compare
    for (case, variant), item in verified.items():
        report = item['report']
        candidate = {'label': 'FNIT_' + variant, 'report': entry(item['report_path']),
                     'report_fields': {'case_id': 'case_id', 'frames': 'frames', 'tr_seconds': 'tr_seconds', 'raw_hashes': 'raw_hashes'},
                     'outputs': {key: item['files'][key] for key in ('preproc_mni', 'dtseries')},
                     'performance': {'timings': [
                         {'name': 'api_seconds', 'seconds': report['api_seconds'], 'boundary': 'API_entry_to_return_cuda_sync',
                          'scope': 'raw_volume_and_surface_reusing_own_reconstruction_and_MSM', 'report_field': 'api_seconds'},
                         {'name': 'api_plus_validation_seconds', 'seconds': report['api_plus_validation_seconds'],
                          'boundary': 'API_entry_through_external_output_validation_and_original_guard_checks',
                          'scope': 'excludes_monitor_join_and_final_wrapper_report_save', 'report_field': 'api_plus_validation_seconds'}]}}
        ref = bindings['cases'][case]['reference']
        reference = {'label': 'fMRIPrep25.2.4', 'report': ref['report'],
                     'report_fields': {'case_id': 'subject', 'frames': 'frames', 'tr_seconds': 'repetition_time', 'raw_hashes': 'input_sha256'},
                     'outputs': {key: ref[key] for key in ('preproc_mni', 'dtseries')}, 'performance': {'timings': []}}
        manifest = {'case_id': case, 'cohort_id': 'fnit_reference_alignment_20261004_' + variant,
                    'frames': 180, 'tr_seconds': bindings['cases'][case]['tr_seconds'], 'raw': bindings['cases'][case]['raw'],
                    'brain_mask': bindings['brain_mask'], 'cifti_axis_assets': bindings['cifti_axis_assets'],
                    'metric_helper': entry(args.metric_helper), 'candidate': candidate, 'reference': reference,
                    'protected_roots': [str(p) for p in protected]}
        manifest_path = target / 'manifests' / (case + '.' + variant + '.private.json')
        write(manifest_path, manifest)
        record = {'case_id': case, 'variant': variant, 'status': 'provenance_preflight_passed',
                  'original_report': candidate['report'], 'original_manifest': entry(item['manifest_path']),
                  'process_seconds': item['execution']['process_seconds'],
                  'process_boundary': 'parent_perf_counter_before_log_open_and_Popen_through_child_exit',
                  'process_scope': 'imports_preflight_API_external_validation_monitor_join_and_wrapper_report_save_excludes_queue_wait',
                  'original_runner_guard_subset_files': item['original_subset_count'],
                  'late_verified_runtime_reconstruction_files': item['late_runtime_reconstruction_files'],
                  'late_verified_resource_files': item['late_resource_files'],
                  'late_closure_boundary': 'independent_posthoc_verification_no_original_MRI_clock_extension',
                  'late_closure_limitation': 'Runtime16closure hashes recorded by the original API now match original formal reconstruction and current files;11consumed resources now match original fixed resource manifest. This does not add missing original runner before_after guards or prove all resources unchanged throughout MRI.'}
        if not args.preflight_only:
            manifest['_manifest_entry'] = entry(manifest_path)
            endpoint_target = target / 'endpoints' / case / variant
            result = compare(manifest, endpoint_target)
            save_endpoint_result(result, endpoint_target, record,
                                 target / (case + '.' + variant + '.preflight_failure.public.json'))
        records.append(record)
    before, after = audit.finish()
    current_queue = load(queue_path)
    queue_check(current_queue)
    # This is a live queue: allow new completed cases, preserve each used exit/timing record.
    for item in verified.values():
        require(item['execution'] in current_queue['completed'], 'used queue execution record changed')
    public_records = []
    for record in records:
        public = {}
        for key, value in record.items():
            if isinstance(value, dict) and 'path' in value and 'sha256' in value:
                public[key + '_sha256'] = value['sha256']
            else:
                public[key] = value
        public_records.append(public)
    summary = {'schema': 'fnit.continuous_late_provenance.v2', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'mode': 'provenance_only' if args.preflight_only else 'endpoint_comparison', 'records': public_records,
               'source_guards_equal': True, 'late_input_guards_equal': before == after,
               'input_sha256_before': before, 'input_sha256_after': after,
               'queue_snapshot_sha256': queue_sha_before, 'queue_current_sha256': sha256(queue_path),
               'queue_mutability_scope': 'allow_new_completed_cases_require_used_records_unchanged',
               'collector_sha256': sha256(__file__), 'closure_helper_sha256': sha256(closure),
               'posthoc_seconds': time.perf_counter() - started,
               'posthoc_timing_boundary': 'collector_start_through_final_guard_checks_excludes_MRI_and_queue_wait',
               'scientific_equivalence': 'not_assessed'}
    write(target / 'provenance.public.json', summary)
    write(target / 'collection.private.json', records)
    return completion_code(records)


if __name__ == '__main__':
    raise SystemExit(main())
