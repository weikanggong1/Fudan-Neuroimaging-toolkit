"""Build a bounded public report from locally collected private JSON records."""
import argparse
import json
from pathlib import Path


RUN_FIELDS = ('status', 'wall_seconds', 'returncode', 'job_sha256', 'hostname',
              'max_cpu_threads', 'cpu_affinity', 'started_utc', 'finished_utc',
              'load_before', 'load_after', 'maximum_sampled_tree_rss_bytes',
              'maximum_sampled_tree_threads', 'resource_samples')


def gate(report, model):
    if report is None:
        return {'status': 'pending'}
    field = {}
    for name, row in report['transforms'].items():
        if model in ('affine', 'rigid'):
            field[name] = row['corner_world_error_mm']['max'] <= 1e-3
        else:
            field[name] = (row['whole_grid']['max_abs'] <= 1e-3 and
                           row['whole_grid']['rmse'] <= 1e-4)
    images = {}
    for name, row in report['images'].items():
        images[name] = {}
        for region in ('whole_grid', 'official_synthstrip_brain',
                       'upper_coordinate_boundary_band'):
            if region not in row:
                images[name][region] = 'not measured'
                continue
            values = row[region]
            nrmse = values['nrmse_reference_p99_minus_p1']
            images[name][region] = ('pass' if nrmse <= 1e-3 else 'fail') if nrmse is not None else (
                'exact zero-error, zero dynamic range' if values['rmse'] == 0 else
                'undefined NRMSE: zero dynamic range, nonzero error')
    return {'field': field, 'image_regions': images,
            'field_all_pass': all(field.values()),
            'all_region_gates_pass': all(field.values()) and all(
                status in ('pass', 'exact zero-error, zero dynamic range')
                for row in images.values() for status in row.values()),
            'full_FOV_and_brain_all_pass': all(
                row.get(region) == 'pass'
                for row in images.values()
                for region in ('whole_grid', 'official_synthstrip_brain'))}


def bounded(value):
    """Remove private absolute paths while retaining exact metrics and hashes."""
    if isinstance(value, dict):
        return {key: bounded(item) for key, item in value.items()
                if key not in ('source_root', 'source_sha256', 'source_files')}
    if isinstance(value, list):
        return [bounded(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        return Path(value).name
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    def get(path):
        return data.get(path)
    def run(path):
        row = get(path)
        return {key: row[key] for key in RUN_FIELDS if key in row} if row else {'status': 'pending'}
    default_reports = {
        'rigid': 'task3_accuracy_numa2_v1/rigid_candidate_official_compare/report.private.json',
        'affine': 'task3_diagnostics_numa2_v1/affine_v1_compare/report.private.json',
        'deform': 'task3_readers_v2/deform_default_compare/report.private.json',
        'joint': 'task3_final_readers/joint_default_compare/report.private.json',
    }
    modes = {}
    for model, path in default_reports.items():
        report = get(path)
        rows = {name: run(f'task3_candidate_numa2_v1/{model}_{name}/record.json')
                for name in ('0_baseline', '1_reference', '2_candidate', '3_candidate', '4_reference')}
        latest = get(f'task3_cpu_v3/{model}_256_v3_compare/report.private.json') if model in ('rigid', 'affine') else report
        modes[model] = {
            'formal_registration_candidate': 'v1',
            'formal_R_C_C_R': rows,
            'comparison_v1': bounded(report), 'gate_v1': gate(report, model),
            'latest_candidate': 'v3',
            'latest_full_CLI': run(f'task3_cpu_v3/{model}_256_v3/record.json') if model in ('rigid', 'affine') else rows['2_candidate'],
            'latest_comparison': bounded(latest), 'latest_gate': gate(latest, model),
            'latest_scope': ('separate complete v3 linear CLI observation; do not relabel v1 R-C-C-R as a v3 run'
                             if model in ('rigid', 'affine') else
                             'v3 changes only final affine coordinate evaluation and CPU nearest ties; this dense registration path retains all v1 operations'),
        }
    variants = {}
    for model, path in {
        'rigid': 'task3_diagnostics_numa2_v1/rigid_header_192_compare/report.private.json',
        'affine': 'task3_diagnostics_numa2_v1/affine_mid_192_patched_compare/report.private.json',
        'deform': 'task3_diagnostics_numa2_v1/deform_hyper_steps_init_192_compare/report.private.json',
        'joint': 'task3_readers_v2/joint_variant_compare/report.private.json',
    }.items():
        q = get(path)
        variants[model] = {
            'reference': run(f'task3_features_numa2_v1/{model}_192_reference/record.json'),
            'candidate': run(f'task3_features_numa2_v1/{model}_192_candidate/record.json'),
            'comparison': bounded(q), 'gate': gate(q, model),
        }
    variants['affine']['reference_scope'] = 'unmodified reference failed with mixed TensorFlow MatMul dtypes; comparison uses explicitly patched diagnostic, not native acceptance'
    variants['affine']['patched_reference'] = run('task3_diagnostics_numa2_v1/affine_init_mid_patched_reference/record.json')
    variants['affine']['patch'] = bounded(get('task3_diagnostics_numa2_v1/affine_init_mid_patched_reference/patch.private.json'))
    variants['affine']['latest_v3_comparison'] = bounded(get('task3_cpu_v3/affine_mid_192_v3_compare/report.private.json'))
    variants['affine']['latest_v3_gate'] = gate(get('task3_cpu_v3/affine_mid_192_v3_compare/report.private.json'), 'affine')
    selected = {}
    namespaces = ('task3_apply_affine_numa2_v1', 'task3_apply_extended_numa2_v1',
                  'task3_apply_v2', 'task3_cpu_v3/affine_apply_v3',
                  'task3_conversion_v1', 'task3_conversion_v2',
                  'task3_final_postreaders', 'task3_final_plot',
                  'task3_final_readers/rigid_debug', 'task3_final_readers/deform_debug',
                  'task3_sqrt_diagnostic')
    for key, value in data.items():
        if any(key.startswith(ns) for ns in namespaces) and key.endswith('.json'):
            if key.endswith('/record.json'):
                selected[key] = run(key)
            elif key.endswith('report.private.json') or 'comparison' in key:
                selected[key] = bounded(value)
    for key in ('task3_cpu_v3/affine_apply_comparison.private.json',
                'task3_features_numa2_v1/affine_api_functional/artifacts/report.private.json',
                'task3_features_numa2_v1/joint_api_functional/artifacts/report.private.json',
                'task3_diagnostics_numa2_v1/joint_cpu_phase_observer/artifacts/report.private.json',
                'task3_features_numa2_v1/joint_reference_observer/observer.private.json',
                'task3_layout_numa2_v1/layout_comparison.private.json'):
        if key in data:
            selected[key] = bounded(data[key])
    gpu = {}
    for version in ('v1', 'v2', 'v3'):
        summary = get(f'task3_gpu_{version}/regression_summary.private.json')
        gpu[version] = bounded(summary)
    sources = {}
    for version, path in {
        'v1': 'task3_gpu_v1/affine_1_candidate/artifacts/report.private.json',
        'v2': 'task3_gpu_v2/affine_candidate_v2/artifacts/report.private.json',
        'v3': 'task3_gpu_v3_resume/affine_candidate_v3_resume/artifacts/report.private.json',
    }.items():
        report = get(path)
        if report:
            sources[version] = report['source_sha256']
    sources['v4'] = get('source_manifest_v4')['source_sha256']
    report = {
        'schema': 'fnit.synthmorph.cpu.benchmark.20261004.v1',
        'data': {'dataset': 'OpenNeuro ds003138 v1.0.1', 'license': 'CC0',
                 'registration_grid': [224, 288, 288],
                 'real_DWI_grid': [120, 120, 68, 2],
                 'DWI_scope': 'two acquired AP frames (b0 and diffusion); no whole-time-series throughput claim',
                 'input_sha256': get('task3_gpu_v1/affine_1_candidate/artifacts/report.private.json')['input_sha256']},
        'reference': {'FreeSurfer': '8.2.0-1', 'TensorFlow': '2.13.1', 'Surfa': '0.6.3', 'FSL': '6.0.7.4',
                      'scope': 'independent original reference environments only; no production dependency'},
        'reference_source_sha256': {
            path.split('site-packages/')[-1].split('python/packages/')[-1]: value
            for path, value in get('task3_features_numa2_v1/joint_reference_observer/observer.private.json')['source_sha256'].items()},
        'baseline_commit': '1d31e7baaebbb644ab199471f7fe6282721455fd',
        'CPU_v1_commit': 'ff99745565c2ed1c144bc209eafa6715a5e1da83',
        'CPU_v2_commit': 'bf791091980667b569993eb05f67b6f257d8554d',
        'CPU_v3_commit': 'd95123ac',
        'CPU_v4_debug_commit': '6dd3b044',
        'common_nib_commit': 'dc2fc052',
        'fixed_gates': json.loads(Path(__file__).with_name('acceptance.json').read_text()),
        'source_sha256': sources,
        'protocol': {'configured_threads': 8, 'physical_cpu_affinity': [2,6,10,14,18,22,26,30],
                     'os_threads_scope': 'helper and idle pools can exceed 8 OS threads; sampled maxima retained; usable physical core budget is 8',
                     'shared_node_nonexclusive': True,
                     'cache': 'new process for each full CLI; OS page cache not flushed; loaded-model API and observers separated',
                     'GPU_scope': 'same-device old/new regression, full input grid, allocator reserved; no NVML total-process memory claim'},
        'default_modes': modes,
        'parameter_variants': variants,
        'functional_and_apply': selected,
        'GPU_regression': gpu,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
