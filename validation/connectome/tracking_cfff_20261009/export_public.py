"""Whitelist FNIT real tracking benchmark reports for public release.

No benchmark is executed. Missing measurements stay missing; null stays null.
The export drops input names/grids, server paths, UUIDs/PIDs, traces, and the
per-streamline digest list. It preserves all recorded timing samples, strict
comparisons, source identities and profiler kernel totals.

Usage:
  python public_export_tracking_cfff.py --input report.private.json \
      --output report.public.json
  python public_export_tracking_cfff.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

SHA = re.compile(r'[0-9a-f]{64}\Z')
UTC = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z')
IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
FORBIDDEN = (
    re.compile(r'(?i)\bGPU-[0-9a-f-]+'),
    re.compile(r'(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'),
    re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    re.compile(r'\b[A-Za-z]:[\\/]'),
    re.compile(r'\\\\[A-Za-z0-9._-]+[\\/]'),
    re.compile(r'(?<![A-Za-z0-9_])/(?:[A-Za-z0-9_.~-]+/)*[A-Za-z0-9_.~-]+'),
)
INPUT_ROLES = ('fod', 'five_tissue', 'gmwmi', 'fa')
SOURCE_ROLES = ('baseline_tracking', 'candidate_tracking', 'baseline_fod',
                'candidate_fod', 'fixed_fod', 'benchmark')
VARIANTS = ('baseline', 'candidate')
SH_FUNCTIONS = ('tracking_sh_precomputed', 'real_sh')
OUTPUT_FIELDS = ('point_counts', 'packed_points', 'accepted_seeds', 'lengths_mm',
                 'endpoints', 'mean_fa')
STRICT_FIELDS = OUTPUT_FIELDS + ('paths', 'seeds_attempted')
OPTIONS = ('n_seeds', 'lmax', 'five_tissue_spacing_mm', 'seed', 'batch_size',
           'arc_proposals', 'max_length_mm', 'min_length_mm', 'step_mm',
           'max_angle_degrees', 'cutoff', 'power', 'compile_arc')
MEMORY_FIELDS = ('peak_allocated_bytes', 'peak_reserved_bytes',
                 'peak_allocated_gb', 'peak_reserved_gb')
HOT_FIELDS = ('tracking_seconds', 'sample_count', 'median_seconds', 'mean_seconds',
              'min_seconds', 'max_seconds')
NOTE_FIELDS = ('input_kind', 'memory_note', 'timing_note', 'cold_note')
NUMERIC_METADATA = ('schema_version', 'cpu_threads', 'cpu_interop_threads',
                    'memory_budget_bytes', 'torch_allocator_budget_bytes',
                    'warmup_seeds', 'profile_seeds', 'strict_comparison_count',
                    'candidate_speedup_from_hot_medians')
BOOL_METADATA = ('tf32_matmul', 'tf32_cudnn', 'autocast', 'all_strict_equal',
                 'fixed_fod_function_identity_equal')
STRING_METADATA = ('status', 'baseline_commit', 'candidate_commit', 'fod_mode',
                   'torch_version', 'cuda_version', 'python_version',
                   'input_dtype', 'affine_dtype', 'point_dtype')


def safe_text(value):
    if value is None:
        return value
    if not isinstance(value, str):
        raise TypeError('Expected string metadata')
    if any(pattern.search(value) for pattern in FORBIDDEN):
        raise ValueError('An allowed text field contains private path/address/identity data')
    return value


def number(value):
    if value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError('Expected finite numeric metadata or null')
    return value


def numeric_vector(value):
    if value is None:
        return value
    if not isinstance(value, list):
        raise TypeError('Expected numeric list')
    return [number(item) for item in value]


def boolean(value):
    if value is not None and not isinstance(value, bool):
        raise TypeError('Expected Boolean metadata or null')
    return value


def digest(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise TypeError('Expected a lowercase SHA-256 digest')
    return value


def select(obj, fields, converter):
    if not isinstance(obj, dict):
        raise TypeError('Expected metadata dictionary')
    return {key: converter(obj[key]) for key in fields if key in obj}


def timing_fields(obj):
    # Numeric *_seconds keys support additional measured phases without
    # exporting arbitrary strings or recursively copying untrusted metadata.
    return {key: number(value) for key, value in obj.items()
            if IDENTIFIER.fullmatch(key) and key.endswith('_seconds')}


def gpu_telemetry(value):
    if value is None:
        return value
    if not isinstance(value, dict):
        raise TypeError('Expected selected GPU telemetry dictionary')
    result = {}
    if 'utc' in value:
        if not isinstance(value['utc'], str) or not UTC.fullmatch(value['utc']):
            raise TypeError('Invalid recorded UTC telemetry timestamp')
        result['utc'] = value['utc']
    if 'unavailable' in value:
        result['unavailable'] = safe_text(value['unavailable'])
        return result
    numeric_fields = ('gpu_utilization_percent', 'memory_used_mib', 'power_watts', 'sm_clock_mhz')
    if 'query_columns' in value and 'row' in value:
        columns = value['query_columns']
        rows = list(csv.reader(value['row'].splitlines()))
        if len(rows) != 1 or len(rows[0]) != len(columns):
            raise ValueError('Selected GPU telemetry must contain one consistent row')
        raw = dict(zip(columns, rows[0]))
        for key in numeric_fields:
            # Earlier benchmark source used this wrong label for the same
            # nvidia-smi memory.used query; only the public label is corrected.
            input_key = 'total_memory_mib' if key == 'memory_used_mib' and key not in raw else key
            if input_key in raw:
                try:
                    item = float(raw[input_key].strip())
                except ValueError:
                    # N/A is retained explicitly and is never converted to zero.
                    if raw[input_key].strip() in ('N/A', '[N/A]', 'Not Supported'):
                        result[key + '_availability'] = raw[input_key].strip()
                        continue
                    raise
                number(item)
                result[key] = int(item) if item.is_integer() else item
    else:
        result.update(select(value, numeric_fields, number))
    return result


def strict_comparison(value):
    result = select(value, ('assessed', 'all_equal', 'path_count_equal'), boolean)
    if 'reason' in value:
        result['reason'] = safe_text(value['reason'])
    for key in ('torch_equal', 'dtype_shape_bytes_equal'):
        if key in value:
            result[key] = select(value[key], STRICT_FIELDS, boolean)
    for key in ('first_mismatched_path_indices', 'first_mismatched_raw_path_indices'):
        if key in value:
            result[key] = numeric_vector(value[key])
    # Forward compatible recorded counts, never inferred from truncated lists.
    for key, item in value.items():
        if IDENTIFIER.fullmatch(key) and key.endswith('_count') and key not in result:
            result[key] = number(item)
    return result


def output_summary(value):
    result = select(value, ('accepted_streamlines', 'total_path_points', 'seeds_attempted'), number)
    if 'field_sha256' in value:
        result['field_sha256'] = select(value['field_sha256'], OUTPUT_FIELDS, digest)
    if 'output_sha256' in value:
        result['output_sha256'] = digest(value['output_sha256'])
    if 'digest_format' in value:
        result['digest_format'] = safe_text(value['digest_format'])
    if 'point_dtype' in value:
        result['point_dtype'] = safe_text(value['point_dtype'])
    # Deliberately do not export value['streamlines'] or point coordinates.
    return result


def profile_events(value):
    if not isinstance(value, list):
        raise TypeError('Expected recorded profiler event list')
    result = []
    for event in value:
        public = timing_fields(event)
        public.update(select(event, ('cpu_seconds_inclusive', 'cpu_seconds_self',
                                     'device_seconds_inclusive', 'device_seconds_self'), number))
        public.update(select(event, ('count',), number))
        public.update(select(event, ('name', 'device_type'), safe_text))
        public.update(select(event, ('is_user_annotation',), boolean))
        result.append(public)
    return result


def profile_summary(value):
    result = timing_fields(value)
    result.update(select(value, ('kernel_count',), number))
    result.update(select(value, ('aggregation',), safe_text))
    return result


def public_run(value):
    result = timing_fields(value)
    result.update(select(value, ('variant', 'phase', 'point_dtype'), safe_text))
    result.update(select(value, ('ordinal', 'n_seeds'), number))
    for key in ('system_load_before', 'system_load_after'):
        if key in value:
            result[key] = numeric_vector(value[key])
    for key in ('gpu_before', 'gpu_after'):
        if key in value:
            result[key] = gpu_telemetry(value[key])
    for key in ('tracking_memory', 'tracking_and_d2h_memory'):
        if key in value:
            result[key] = select(value[key], MEMORY_FIELDS, number)
    if 'output' in value:
        result['output'] = output_summary(value['output'])
    if 'strict_comparison' in value:
        result['strict_comparison'] = strict_comparison(value['strict_comparison'])
    if 'tck_sha256' in value:
        result['tck_sha256'] = digest(value['tck_sha256'])
    if 'profile_summary' in value:
        result['profile_summary'] = profile_summary(value['profile_summary'])
    if 'profile_top_events' in value:
        result['profile_top_events'] = profile_events(value['profile_top_events'])
    return result


def assert_public(value):
    """Fail closed on prohibited data even if a future whitelist field changes."""
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if any(pattern.search(rendered) for pattern in FORBIDDEN):
        raise ValueError('Public JSON contains a private path/address/UUID')
    prohibited_keys = ('path', 'filename', 'uuid', 'pid', 'streamlines', 'profile_trace',
                       'tck_path', 'input_affines', 'input_shapes')
    def visit(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if key.lower() in prohibited_keys or re.search(r'(?:^|_)(?:uuid|pid)(?:_|$)', key.lower()):
                    raise ValueError('Public JSON contains a prohibited key')
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)
    visit(value)


def export_report(value):
    if not isinstance(value, dict):
        raise TypeError('Benchmark report must be a dictionary')
    result = {'public_export_schema_version': 1}
    result.update(select(value, NUMERIC_METADATA, number))
    result.update(select(value, BOOL_METADATA, boolean))
    result.update(select(value, STRING_METADATA + NOTE_FIELDS, safe_text))
    if 'input_files' in value:
        result['input_files'] = {
            role: {**select(value['input_files'][role], ('size_bytes',), number),
                   **select(value['input_files'][role], ('sha256',), digest)}
            for role in INPUT_ROLES if role in value['input_files']}
    if 'source_sha256' in value:
        result['source_sha256'] = select(value['source_sha256'], SOURCE_ROLES, digest)
    if 'cpu_affinity' in value:
        result['cpu_affinity'] = numeric_vector(value['cpu_affinity'])
    if 'device' in value:
        result['device'] = {**select(value['device'], ('device', 'name'), safe_text),
                            **select(value['device'], ('total_memory_bytes',), number)}
    if 'options' in value:
        result['options'] = {}
        for key in OPTIONS:
            if key not in value['options']:
                continue
            item = value['options'][key]
            result['options'][key] = (boolean(item) if key == 'compile_arc' else
                                      numeric_vector(item) if key == 'five_tissue_spacing_mm' else number(item))
    if 'fod_function_identity' in value:
        result['fod_function_identity'] = {
            role: select(value['fod_function_identity'][role], SH_FUNCTIONS, boolean)
            for role in VARIANTS if role in value['fod_function_identity']}
    if 'cross_variant_fod_function_identity' in value:
        result['cross_variant_fod_function_identity'] = select(
            value['cross_variant_fod_function_identity'], SH_FUNCTIONS, boolean)
    if 'shared_input_timing' in value:
        result['shared_input_timing'] = timing_fields(value['shared_input_timing'])
    if 'runs' in value:
        if not isinstance(value['runs'], list):
            raise TypeError('runs must be a list')
        result['runs'] = [public_run(run) for run in value['runs']]
    if 'hot_order' in value:
        result['hot_order'] = [safe_text(item) for item in value['hot_order']]
    if 'hot_summary' in value:
        result['hot_summary'] = {}
        for variant in VARIANTS:
            if variant not in value['hot_summary']:
                continue
            hot = value['hot_summary'][variant]
            result['hot_summary'][variant] = {
                key: numeric_vector(hot[key]) if key == 'tracking_seconds' else number(hot[key])
                for key in HOT_FIELDS if key in hot}
    if 'profile_summary' in value:
        result['profile_summary'] = profile_summary(value['profile_summary'])
    if 'profile_top_events' in value:
        result['profile_top_events'] = profile_events(value['profile_top_events'])
    if 'error' in value:
        # Error messages can contain paths, addresses and child process PIDs.
        result['error'] = select(value['error'], ('type',), safe_text)
    assert_public(result)
    return result


def self_test():
    h = 'a' * 64
    uuid = 'GPU-12345678-1234-1234-1234-123456789abc'
    private = '/cpfs/private/secret/report.json'
    fields = {key: True for key in STRICT_FIELDS}
    base_run = {
        'variant': 'baseline', 'phase': 'first_full_call', 'ordinal': 0,
        'n_seeds': 10000, 'tracking_seconds': 100.125,
        'tracking_process_cpu_seconds': 3.5, 'path_pack_seconds': .25,
        'd2h_seconds': .125, 'comparison_seconds': .5, 'summary_seconds': 1.,
        'future_measured_seconds': .75, 'system_load_before': [1., 2., 3.],
        'gpu_before': {'utc': '2026-10-09T10:00:00Z',
                       'query_columns': ['uuid', 'gpu_utilization_percent', 'memory_used_mib', 'power_watts', 'sm_clock_mhz'],
                       'row': uuid + ', 97, 4100, 270.5, 1200'},
        'gpu_after': None,
        'tracking_memory': {'peak_allocated_bytes': 1234, 'peak_reserved_bytes': 5678,
                            'peak_allocated_gb': .000001234, 'peak_reserved_gb': .000005678},
        'output': {'accepted_streamlines': 2718, 'total_path_points': 10000,
                   'seeds_attempted': 10000, 'field_sha256': {key: h for key in OUTPUT_FIELDS},
                   'output_sha256': h, 'digest_format': 'v1: named dtype/shape headers plus little-endian tensor bytes',
                   'streamlines': [{'point_count': 42, 'sha256': 'b' * 64}], 'path': private},
        'strict_comparison': {'assessed': False, 'all_equal': None,
                              'reason': 'no same-scale reference output'},
        'tck_path': private, 'profile_trace': private, 'pid': 999999,
        'tck_sha256': h, 'tck_write_seconds': .99,
        'profile_summary': {'kernel_count': 42, 'kernel_device_seconds': .25,
                            'aggregation': 'Each raw CUDA kernel event is counted once.'},
        'profile_top_events': [{'name': 'aten::index_copy_', 'device_type': 'DeviceType.CPU',
                               'count': 17, 'cpu_seconds_inclusive': .1,
                               'cpu_seconds_self': .05, 'device_seconds_inclusive': .09,
                               'device_seconds_self': .08, 'is_user_annotation': False,
                               'path': private}],
    }
    assessed_run = dict(base_run, variant='candidate', phase='hot', tracking_seconds=90.25,
                        strict_comparison={'assessed': True, 'all_equal': True,
                                           'torch_equal': fields, 'dtype_shape_bytes_equal': fields,
                                           'path_count_equal': True,
                                           'first_mismatched_path_indices': [],
                                           'first_mismatched_raw_path_indices': []})
    source = {
        'schema_version': 1, 'status': 'passed', 'fod_mode': 'paired',
        'input_files': {'fod': {'name': private, 'size_bytes': 12345, 'sha256': h}},
        'input_shapes': {'fod': [104, 104, 72, 45]},
        'input_affines': {'fod': [[1, 0, 0, 0]]},
        'source_sha256': {key: h for key in SOURCE_ROLES},
        'device': {'device': 'cuda:0', 'name': 'NVIDIA A100', 'total_memory_bytes': 80_000_000_000,
                   'uuid': uuid, 'pid': 123456},
        'input_dtype': 'float32', 'affine_dtype': 'float64', 'point_dtype': 'float32',
        'baseline_commit': '4f56cc9d5937a766c432743809ab3b5baf11ca78',
        'options': {'n_seeds': 10000, 'seed': 0, 'batch_size': 8192,
                    'compile_arc': False, 'five_tissue_spacing_mm': [1., 1., 1.],
                    'step_mm': None, 'path': private},
        'fod_function_identity': {variant: {key: True for key in SH_FUNCTIONS} for variant in VARIANTS},
        'cross_variant_fod_function_identity': {key: False for key in SH_FUNCTIONS},
        'fixed_fod_function_identity_equal': False,
        'shared_input_timing': {'io_seconds': 4., 'h2d_seconds': .1, 'path': private},
        'runs': [base_run, assessed_run], 'strict_comparison_count': 1,
        'all_strict_equal': True, 'hot_order': ['candidate'],
        'hot_summary': {'baseline': {'tracking_seconds': [], 'sample_count': 0, 'median_seconds': None},
                        'candidate': {'tracking_seconds': [90.25], 'sample_count': 1, 'median_seconds': 90.25}},
        'candidate_speedup_from_hot_medians': None,
        'error': {'type': 'RuntimeError', 'message': private + ' ' + uuid},
        'server_path': private, 'gpu_uuid': uuid, 'host': '203.0.113.42',
    }
    public = export_report(source)
    assert public['input_files']['fod'] == {'size_bytes': 12345, 'sha256': h}
    assert public['runs'][0]['future_measured_seconds'] == .75
    assert public['runs'][0]['gpu_before']['memory_used_mib'] == 4100
    assert public['runs'][0]['gpu_after'] is None
    assert public['runs'][0]['profile_summary'] == base_run['profile_summary']
    assert public['runs'][0]['profile_top_events'][0]['count'] == 17
    for key in ('cpu_seconds_inclusive', 'cpu_seconds_self', 'device_seconds_inclusive', 'device_seconds_self'):
        assert public['runs'][0]['profile_top_events'][0][key] == base_run['profile_top_events'][0][key]
    assert public['strict_comparison_count'] == 1 and public['all_strict_equal'] is True
    assert public['fod_mode'] == 'paired' and public['cross_variant_fod_function_identity']['real_sh'] is False
    assert public['runs'][0]['strict_comparison']['all_equal'] is None
    assert public['hot_summary']['baseline']['median_seconds'] is None
    assert 'streamlines' not in public['runs'][0]['output']
    assert public['error'] == {'type': 'RuntimeError'}
    # All private strings were in explicitly dropped fields.
    assert private not in json.dumps(public) and uuid not in json.dumps(public)
    shared = dict(source, fod_mode='shared', fixed_fod_function_identity_equal=True)
    del shared['fod_function_identity']
    del shared['cross_variant_fod_function_identity']
    result = export_report(shared)
    assert result['fod_mode'] == 'shared' and result['fixed_fod_function_identity_equal'] is True
    assert 'fod_function_identity' not in result  # never invent absent identity data
    empty = export_report({'schema_version': 1, 'status': 'running', 'runs': []})
    assert empty == {'public_export_schema_version': 1, 'schema_version': 1, 'status': 'running', 'runs': []}
    for forbidden in [private, uuid, '203.0.113.42', r'C:\private\thing', r'\\private\share\thing']:
        for where in ['timing_note', 'point_dtype']:
            unsafe = dict(source, **{where: forbidden})
            try:
                export_report(unsafe)
            except ValueError:
                pass
            else:
                raise AssertionError('Unsafe whitelisted string escaped the final privacy check')
    legacy = dict(base_run['gpu_before'])
    legacy['query_columns'] = ['uuid', 'gpu_utilization_percent', 'total_memory_mib', 'power_watts', 'sm_clock_mhz']
    assert gpu_telemetry(legacy)['memory_used_mib'] == 4100
    for bad_key in ('pid_seconds', 'gpu_uuid_seconds'):
        unsafe = dict(base_run, **{bad_key: 123456})
        try:
            assert_public(public_run(unsafe))
        except ValueError:
            pass
        else:
            raise AssertionError('Identity key escaped numeric timing validation')
    unavailable = {'unavailable': 'TimeoutExpired'}
    assert gpu_telemetry(unavailable) == unavailable
    print('self-test passed: paired/shared, missing/null, all timing samples, strict flags, profiler, prohibited fields')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.input is None and args.output is None:
            return
    if args.input is None or args.output is None:
        parser.error('--input and --output must be supplied together')
    if args.output.exists():
        parser.error('Output already exists; choose a fresh path to preserve prior reports')
    value = json.loads(args.input.read_text())
    public = export_report(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'exported', 'private_report_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                      'public_report_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'run_count': len(public.get('runs', []))}))


if __name__ == '__main__':
    main()
