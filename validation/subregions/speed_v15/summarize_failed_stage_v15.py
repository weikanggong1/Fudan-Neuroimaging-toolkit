import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path('validation/subregions').resolve()))
from compare_optimization import gpu_load as gpu_load_summary

root = Path('validation/subregions/speed_v15')
label = 'full_stage_fast_v15_20261001'
status = json.loads((root / (label + '_status.json')).read_text())
launch = json.loads((root / (label + '_launch.json')).read_text())
verification = json.loads((root / 'stage_v15_failed_remote_verification.json').read_text())
log = (root / (label + '.log')).read_text()
lines = log.splitlines()
completed = {}
for name in ('brainstem', 'thalamus', 'hippo-amygdala-left'):
    marker = 'Finished ' + name + ': '
    line = next((line for line in lines if marker in line), None)
    if line is not None:
        completed[name] = ast.literal_eval(line.split(marker, 1)[1])
error = next(line for line in reversed(lines) if line.startswith('torch.OutOfMemoryError:'))
summary = {
    'scope': 'failed real official_stage_inputs run; incomplete output is not a speed or accuracy benchmark',
    'label': label, 'state': status['state'], 'exit_code': status['exit_code'],
    'driver_elapsed_until_failure_seconds': status['finished_unix'] - status['started_unix'],
    'compute_wall_seconds': None, 'api_total_seconds': None, 'output_save_seconds': None,
    'final_result_available': False, 'run_output_directory_was_empty_after_failure': True,
    'unavailable_final_metrics': ['six_family_native_comparison', '110_soft_volumes',
                                  'all_official_thresholds', 'four_final_min_jacobians', 'highres_outputs'],
    'failure': {'exception': 'torch.OutOfMemoryError', 'message': error,
                'structure': 'hippo-amygdala-right', 'phase': 'fine intensity stage 1/3 posterior construction after outer 7/7',
                'allocation_requested_mib': 776.0, 'global_free_mib_at_error': 443.06,
                'own_including_non_pytorch_gib_at_error': 6.59,
                'pytorch_allocated_gib_at_error': 2.99, 'pytorch_reserved_unallocated_mib_at_error': 100.14,
                'pytorch_allocator_limit_gib_from_error': 11.08,
                'frame': 'gems/gaussian.py:101 log_joint = log_prior + class_log_likelihood[label_classes.long()]',
                'interpretation': 'allocation exceeded the reported free memory on the shared GPU; no complete candidate result was saved'},
    'status': status, 'launch': launch, 'gpu_load': gpu_load_summary(root / (label + '_gpu_load.jsonl')),
    'sampled_own_process_peak_mib': status['max_own_process_memory_mib'],
    'pytorch_peak_gpu_gib': None,
    'source_commit_claim': launch['source_commit'],
    'executed_stack_source_files': verification['executed_stack_source_files'],
    'stack_source_hashes_match_manifest': verification['stack_source_hashes_match_manifest'],
    'completed_structure_log_metadata': completed,
    'replacement_run_label': 'full_stage_fast_v15_gpu1_20261001',
    'replacement_source_is_unchanged': True,
}
output = root / 'stage_v15_failed_summary.json'
with output.open('x') as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)
    stream.write('\n')
files = []
for item in verification['files']:
    blob = (root / item['archive_path']).read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    if sha != item['sha256'] or len(blob) != item['bytes']:
        raise RuntimeError(item['archive_path'])
    files.append({**item, 'local_sha256': sha, 'remote_match': True})
shutil.copyfile('/tmp/fnit_v15_failed_summarize.py', root / 'summarize_failed_stage_v15.py')
for name in ('stage_v15_failed_summary.json', 'summarize_failed_stage_v15.py', 'stage_v15_failed_remote_verification.json'):
    blob = (root / name).read_bytes()
    files.append({'archive_path': name, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest(),
                  'scope': 'local derived failure validation evidence'})
with (root / 'stage_v15_failed_small_file_archive_manifest.json').open('x') as stream:
    json.dump({'scope': 'failed stage v15 run; original label retained, no MRI data copied',
               'all_remote_files_match': True, 'files': files}, stream, indent=2)
    stream.write('\n')
print(json.dumps({'summary': str(output), 'state': status['state'], 'exit_code': status['exit_code'],
                  'elapsed_seconds': summary['driver_elapsed_until_failure_seconds'],
                  'completed_structure_log_metadata': list(completed),
                  'source_hashes_match': summary['stack_source_hashes_match_manifest'],
                  'remote_files_match': True, 'load_samples': summary['gpu_load']['samples']}))
