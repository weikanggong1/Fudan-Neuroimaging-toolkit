"""Build an anonymous v4 FastVBM report from completed, collected JSON records.

Only the Python standard library is required. This reads JSON, never loads MRI,
executes commands, invokes inference, downloads resources, or contacts a server.
The previous public report supplies the existing official reference receipts;
all current array/geometry conclusions are derived from the four 13-map scores.
"""

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path, PureWindowsPath


SOURCE_LABEL = 'task5_candidate_cpu_v4'
HEAD_PREFIX = '6f1e2b38'
ARCHIVE_SHA256 = 'ffda47a74376fbaec07c3e8aedaacdc30f2a60398b919c0d5feae38e3beba0d9'
FILES = (
    'T1_brain.nii.gz', 'brain_mask.nii.gz', 'T1_brain_pve_0.nii.gz',
    'T1_brain_pve_1.nii.gz', 'T1_brain_pve_2.nii.gz', 'T1_brain_seg.nii.gz',
    'T1_brain_pveseg.nii.gz', 'T1_brain_mixeltype.nii.gz', 'T1_brain_bias.nii.gz',
    'T1_brain_restore.nii.gz', 'T1_GM_to_template_GM.nii.gz',
    'T1_GM_JAC_nl.nii.gz', 'T1_GM_to_template_GM_mod.nii.gz',
)
COMPARISONS = ('fnirt_vs_v2', 'synthmorph_vs_v2',
               'fnirt_vs_official', 'synthmorph_vs_official')
QUEUE_FIELDS = (
    'job_sha256', 'hostname', 'max_cpu_threads', 'cpu_affinity', 'status',
    'started_utc', 'finished_utc', 'wall_seconds', 'returncode',
    'maximum_sampled_tree_rss_bytes', 'maximum_sampled_tree_threads',
    'load_before', 'load_after',
)
THREAD_ENVIRONMENT = {
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS',
    'CUDA_VISIBLE_DEVICES', 'OMP_PROC_BIND', 'OMP_PLACES',
    'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS',
    'TF_ENABLE_ONEDNN_OPTS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',
    'MALLOC_ARENA_MAX',
}
PRIVATE_KEYS = {
    'argv', 'command', 'cmd', 'commands', 'command_line', 'commandline',
    'job_command_basenames_only', 'stdout', 'stderr', 'cwd', 'workdir',
    'environment', 'PYTHONPATH', 'FS_LICENSE', 'license_file',
    'password', 'token', 'api_key', 'credentials',
}
# Also redact embedded absolute locations in otherwise descriptive strings.
ABSOLUTE_PATH = re.compile(r'(?<![\w/])(?:[A-Za-z]:[\\/]|/)[^\s\"\'<>;,\)\]\}]+')
SHA256 = re.compile(r'^[0-9a-fA-F]{64}$')


def redact(value):
    """Preserve parameter names/basenames and remove locations/command payloads."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PRIVATE_KEYS:
                continue
            public_key = redact(key)
            if public_key in result:
                raise ValueError('basename redaction would merge distinct metadata keys')
            result[public_key] = redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        if value.startswith('/'):
            return Path(value).name or '[redacted location]'
        if re.match(r'^[A-Za-z]:[\\/]', value):
            return PureWindowsPath(value).name or '[redacted location]'
        option, separator, argument = value.partition('=')
        if separator and argument.startswith('/'):
            return option + separator + Path(argument).name
        if value.startswith(('http://', 'https://')):
            return value
        return ABSOLUTE_PATH.sub(
            lambda match: (PureWindowsPath(match[0]).name if ':' in match[0][:2]
                           else Path(match[0]).name) or '[redacted location]', value)
    return value


def queue_summary(record):
    """Retain existing queue timing/resource semantics; omit all argv payloads."""
    if not isinstance(record, dict):
        raise ValueError('queue record must be an object')
    result = {key: record[key] for key in QUEUE_FIELDS if key in record}
    job = record.get('job', {})
    if isinstance(job, dict) and 'id' in job:
        result['job_id'] = job['id']
    elif 'job_id' in record:
        result['job_id'] = record['job_id']
    samples = record.get('resource_samples', record.get('resource_samples_count', 0))
    result['resource_samples_count'] = len(samples) if isinstance(samples, list) else samples
    environment = record.get('environment', record.get('thread_environment', {}))
    if isinstance(environment, dict):
        result['thread_environment'] = {
            key: item for key, item in environment.items() if key in THREAD_ENVIRONMENT
        }
    return redact(result)


def seconds(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(field + ' must be a measured number of seconds')
    if not math.isfinite(value) or value < 0:
        raise ValueError(field + ' must be finite and nonnegative')
    return float(value)


def completed(record, field):
    if record.get('status') != 'complete' or record.get('returncode') != 0:
        raise ValueError(field + ' is not a completed successful queue run')
    return seconds(record.get('wall_seconds'), field + '.wall_seconds')


def observed_all(values):
    """False if contradicted, null if unmeasured, true only if every item is true."""
    values = list(values)
    if any(value is False for value in values):
        return False
    if not values or any(value is not True for value in values):
        return None
    return True


def boolean(row, key):
    value = row.get(key)
    return value if isinstance(value, bool) else None


def form_equal(row, form):
    value = boolean(row, form + '_equal')
    if value is not None:
        return value
    error = row.get(form + '_maximum_absolute_error')
    return error == 0 if isinstance(error, (int, float)) and not isinstance(error, bool) else None


def comparison_summary(comparison):
    rows = comparison.get('outputs')
    if not isinstance(rows, dict) or set(rows) != set(FILES):
        raise ValueError('each comparison must contain exactly the thirteen expected map names')
    if comparison.get('output_count') != len(FILES):
        raise ValueError('comparison output_count does not match the thirteen map records')
    summaries = {}
    for name in FILES:
        row = rows[name]
        if not isinstance(row, dict):
            raise ValueError('each map comparison must be an object')
        complete = row.get('comparison_status') == 'complete'
        header = row.get('header_equal', {})
        if not isinstance(header, dict):
            raise ValueError('header_equal must be an object')
        shape = boolean(row, 'shape_equal')
        dtype = boolean(row, 'dtype_equal')
        if dtype is None and 'candidate_dtype' in row and 'reference_dtype' in row:
            dtype = row['candidate_dtype'] == row['reference_dtype']
        qform, sform = form_equal(row, 'qform'), form_equal(row, 'sform')
        reported_header = observed_all(header.values())
        geometry = observed_all((shape, dtype, boolean(row, 'affine_equal'),
                                 qform, sform, reported_header))
        # Do not trust a global comparator conclusion instead of individual evidence.
        array = boolean(row, 'values_exact_equal') if complete else None
        bits = row.get('different_bit_patterns')
        bitwise = (bits == 0 if isinstance(bits, int) and not isinstance(bits, bool)
                   and dtype is True and complete else None)
        summaries[name] = {
            'comparison_complete': complete,
            'finite': boolean(row, 'finite'),
            'arrays_exact_equal': array,
            'array_bit_patterns_equal': bitwise,
            'shape_equal': shape, 'dtype_equal': dtype,
            'affine_equal': boolean(row, 'affine_equal'),
            'qform_equal': qform, 'sform_equal': sform,
            'all_reported_header_fields_equal': reported_header,
            'reported_geometry_equal': geometry,
            'extensions_equal': boolean(row, 'extensions_equal'),
        }
    fields = (
        'comparison_complete', 'finite', 'arrays_exact_equal', 'array_bit_patterns_equal',
        'shape_equal', 'dtype_equal', 'affine_equal', 'qform_equal', 'sform_equal',
        'all_reported_header_fields_equal', 'reported_geometry_equal', 'extensions_equal',
    )
    return {
        'output_count': len(rows),
        **{'all_thirteen_' + field: observed_all(row[field] for row in summaries.values())
           for field in fields},
        'reported_header_fields': sorted(set().union(
            *(set(row.get('header_equal', {})) for row in rows.values()))),
        'header_fields_checked_for_every_map': sorted(set.intersection(
            *(set(row.get('header_equal', {})) for row in rows.values()))),
        'complete_nifti_header_bitwise_equality_measured': False,
        'geometry_definition': 'shape, dtype, affine, qform, sform and the listed header fields; extensions reported separately',
        'per_map': summaries,
    }


def current_candidate(report, branch):
    allowed = ('status', 'method', 'settings', 'timing_sec', 'timing_definition',
               'outputs', 'fast', 'registration', 'fnirt_equivalent',
               'fsl_flirt_equivalent', 'fsl_fnirt_numerically_equivalent')
    # FNIT labels scientific parity as 'experimental'; successful completion is
    # established by the queue receipt, not by changing that scientific label.
    if report.get('status') not in ('ok', 'complete', 'success', 'experimental'):
        raise ValueError(branch + ' candidate report has an unsupported status')
    if not isinstance(report.get('outputs'), dict) or len(report['outputs']) != len(FILES):
        raise ValueError(branch + ' candidate report must enumerate thirteen outputs')
    return redact({key: report[key] for key in allowed if key in report})


def stage_timings(reference):
    allowed = ('name', 'wall_seconds', 'status', 'returncode', 'started_utc', 'finished_utc')
    return [redact({key: row[key] for key in allowed if key in row})
            for row in reference.get('stages', [])]


def source_summary(metadata):
    required = ('source_label', 'head', 'archive_sha256', 'source_manifest_sha256',
                'verified_files_count', 'all_manifest_file_hashes_verified_before_queue',
                'includes_reviewed_uncommitted_changes', 'source_modules_sha256',
                'canonical_index_updated_utc', 'resource_identity')
    if any(key not in metadata for key in required):
        raise ValueError('metadata is missing required source/resource provenance fields')
    if metadata['source_label'] != SOURCE_LABEL or not metadata['head'].startswith(HEAD_PREFIX):
        raise ValueError('collected source is not the requested final v4 source')
    if metadata['archive_sha256'] != ARCHIVE_SHA256:
        raise ValueError('collected archive differs from the reviewed v4 archive')
    if not SHA256.fullmatch(str(metadata['source_manifest_sha256'])):
        raise ValueError('source manifest SHA-256 is missing or invalid')
    if metadata['verified_files_count'] != 1257 or (
            metadata['all_manifest_file_hashes_verified_before_queue'] is not True):
        raise ValueError('collected metadata does not confirm the 1257-file verification')
    if metadata['includes_reviewed_uncommitted_changes'] is not True:
        raise ValueError('metadata does not identify the reviewed uncommitted v4 source')
    modules = metadata['source_modules_sha256']
    if not isinstance(modules, dict) or not modules or any(
            not SHA256.fullmatch(str(value)) for value in modules.values()):
        raise ValueError('source_modules_sha256 must contain verified module digests')
    result = {key: metadata[key] for key in required}
    for key in ('plan_sha256', 'scoring_sources_sha256',
                'post_run_python_files_verified', 'post_run_python_source_unchanged'):
        if key in metadata:
            result[key] = metadata[key]
    result['1257_verified'] = True
    result['reviewed_uncommitted'] = metadata['includes_reviewed_uncommitted_changes']
    result['head_is_not_a_clean_commit_receipt_for_entire_candidate'] = True
    result['source_verification_scope'] = 'frozen v4 archive and its manifest, including reviewed uncommitted changes'
    return redact(result)


def build_report(collected, prior, prior_sha256):
    provenance = source_summary(collected['metadata'])
    queues, candidates, timing = {}, {}, {}
    for branch in ('fnirt', 'synthmorph'):
        record = collected['candidate_' + branch + '_record']
        current_wall = completed(record, 'candidate_' + branch)
        queues[branch] = queue_summary(record)
        candidate = current_candidate(collected['candidate_' + branch + '_report'], branch)
        candidates[branch] = candidate
        internal = candidate.get('timing_sec', {})
        if not isinstance(internal, dict):
            raise ValueError('candidate timing_sec must be an object')
        internal = {key: seconds(value, branch + '.timing_sec.' + key)
                    for key, value in internal.items()}
        previous = prior['queue_measurements']['candidate_' + branch + '_complete_cli']
        previous_wall = completed(previous, 'previous_v2_' + branch)
        timing[branch] = {
            'current_complete_cli_wall_seconds': current_wall,
            'current_internal_report_seconds': internal,
            'wall_minus_internal_report_total_seconds': (
                current_wall - internal['total'] if 'total' in internal else None),
            'previous_frozen_v2_complete_cli_wall_seconds': previous_wall,
            'observed_current_minus_previous_v2_seconds': current_wall - previous_wall,
            'scope': 'one new process, raw T1 through thirteen saved maps, including launch, reads and saves; no cache flush; v2 receipt reused; not an adjacent pair or ABBA repeat',
        }
    comparison_payloads, summaries = {}, {}
    for name in COMPARISONS:
        raw = collected[name]
        summaries[name] = comparison_summary(raw)
        keys = ('output_count', 'outputs', 'regions', 'linear_registration',
                'affine_pull_world', 'registration_qc_field_equal')
        comparison_payloads[name] = redact({key: raw[key] for key in keys if key in raw})
        comparison_payloads[name]['scope'] = (
            'current v4 versus saved prior v2 maps' if name.endswith('_v2')
            else 'current v4 versus existing original-software reference maps')
    official_fnirt = prior['queue_measurements']['original_fnirt_complete_chain']
    fnirt_wall = completed(official_fnirt, 'previous_original_fnirt')
    morph_timing = prior['synthmorph_reference_timing']
    morph_stage_sum = seconds(morph_timing['reconstructed_complete_stage_sum_seconds'],
                              'previous_original_synthmorph_stage_sum')
    if abs(fnirt_wall - 769.365) > .002 or abs(morph_stage_sum - 580.141) > .002:
        raise ValueError('prior public report does not contain the specified official receipts')
    reference_steps = prior.get('original_reference_steps', {})
    previous_queues = prior['queue_measurements']
    references = {
        'remeasured_for_v4': False,
        'prior_public_report_sha256': prior_sha256,
        'fnirt_complete_chain_wall_seconds': fnirt_wall,
        'fnirt_complete_chain_queue_receipt': queue_summary(official_fnirt),
        'fnirt_stage_timings': stage_timings(reference_steps.get('fnirt', {})),
        'synthmorph_complete_stage_sum_seconds': morph_stage_sum,
        'synthmorph_stage_sum_components': redact(morph_timing),
        'synthmorph_stage_receipts': {
            name: queue_summary(previous_queues[name]) for name in (
                'original_synthmorph_branch_upstream_reused',
                'original_synthmorph_postprocessing_replay_only') if name in previous_queues
        },
        'synthmorph_native_stage_timings': stage_timings(reference_steps.get(
            'synthmorph_native_run_with_superseded_rounded_geometry_postprocessing', {})),
        'synthmorph_corrected_postprocessing_stage_timings': stage_timings(reference_steps.get(
            'synthmorph_final_postprocessing_replay_actual_nifti_geometry', {})),
        'synthmorph_fresh_complete_cli_wall_seconds': None,
        'timing_scope': 'FNIRT is a previous complete-chain wall clock; SynthMorph is the sum of separately measured upstream, native model and corrected postprocessing stages. Existing data/resources and filesystem caches were used. Neither reference was rerun for v4.',
    }
    report = {
        'schema_version': 2, 'date': '2026-10-04',
        'scope': 'one real raw T1; two final v4 CPU branches; thirteen saved maps per branch; original reference receipts reused',
        'source_provenance': provenance,
        'input_reference_metadata': redact(prior['input']),
        'input_reference_metadata_origin': 'previous anonymous report; current resource identities are listed in source_provenance',
        'protocol': {
            'sample_count': 1, 'exclusive_node': False,
            'final_frozen_v4_complete_raw_to_thirteen_maps_rerun': True,
            'native_calls_only_in_reference_workers': True,
            'original_reference_rerun_for_v4': False,
            'cache_flush_performed': False,
            'adjacent_reference_candidate_pair': False,
            'complete_pipeline_ab_ba_repeats': False,
            'stable_speedup_established': False,
            'formal_end_to_end_speed_ratio_established': False,
            'queue_settings_are_reported_from_current_records': True,
            'reference_versions_from_previous_report': redact(prior.get('protocol', {}).get('reference_versions', {})),
            'reference_hardware_from_previous_report': {
                key: prior.get('protocol', {}).get(key) for key in ('host', 'cpu')
            },
            'numba_compilation_scope': 'OS and Numba caches were not cleared; complete wall clocks include compilation costs actually incurred during startup; first-installation JIT was not separately measured',
            'assets_downloaded_or_redistributed': False,
            'raw_mri_files_or_private_commands_published': False,
            'derived_brain_figures_published': bool(collected.get('figure_scope')),
        },
        'queue_measurements': queues,
        'candidate_reports': candidates,
        'current_candidate_timing': timing,
        'existing_official_reference_receipts': references,
        'comparisons': comparison_payloads,
        'comparison_summaries': summaries,
        'figure_scope': redact(collected.get('figure_scope')),
        'conclusions': {
            'v4_vs_v2_all_thirteen_arrays_equal_by_branch': {
                branch: summaries[branch + '_vs_v2']['all_thirteen_arrays_exact_equal']
                for branch in ('fnirt', 'synthmorph')},
            'v4_vs_v2_all_thirteen_array_bit_patterns_equal_by_branch': {
                branch: summaries[branch + '_vs_v2']['all_thirteen_array_bit_patterns_equal']
                for branch in ('fnirt', 'synthmorph')},
            'v4_vs_v2_all_thirteen_reported_geometry_equal_by_branch': {
                branch: summaries[branch + '_vs_v2']['all_thirteen_reported_geometry_equal']
                for branch in ('fnirt', 'synthmorph')},
            'v4_vs_official_all_thirteen_arrays_equal_by_branch': {
                branch: summaries[branch + '_vs_official']['all_thirteen_arrays_exact_equal']
                for branch in ('fnirt', 'synthmorph')},
            'array_and_geometry_findings_computed_from_current_thirteen_map_rows': True,
            'stable_speedup_established': False,
            'remaining_work': ['adjacent balanced complete-chain repeats', 'wider real-subject coverage'],
        },
    }
    if 'scoring_records' in collected:
        records = collected['scoring_records']
        if isinstance(records, dict):
            report['scoring_queue_measurements'] = {
                redact(key): queue_summary(value) for key, value in records.items()}
        elif isinstance(records, list):
            report['scoring_queue_measurements'] = [queue_summary(value) for value in records]
        else:
            raise ValueError('scoring_records must be an object or list of queue records')
    return redact(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collected', required=True, type=Path, help='private collected JSON records')
    parser.add_argument('--prior-public', required=True, type=Path, help='existing independent anonymous v2 report')
    parser.add_argument('--output', required=True, type=Path, help='new anonymous v4 report JSON')
    args = parser.parse_args()
    if args.output.resolve() in (args.collected.resolve(), args.prior_public.resolve()):
        raise ValueError('output must not overwrite either input JSON')
    collected = json.loads(args.collected.read_text(encoding='utf-8'))
    prior_bytes = args.prior_public.read_bytes()
    report = build_report(collected, json.loads(prior_bytes), hashlib.sha256(prior_bytes).hexdigest())
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    forbidden = ('/cwStorage/', '/public/', '/home/', '/mnt/', '/tmp/', 'FS_LICENSE',
                 'gongwk@', 'gwk_44019', 'BEGIN PRIVATE KEY', 'PYTHONPATH',
                 'job_command_basenames_only')
    if any(token in serialized for token in forbidden):
        raise ValueError('private location, environment or command escaped redaction')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=args.output.parent,
                                     prefix=args.output.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(serialized)
    try:
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({'comparisons': len(COMPARISONS), 'maps_per_comparison': len(FILES),
                      'bytes': len(serialized.encode('utf-8')),
                      'conclusions': report['conclusions']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
