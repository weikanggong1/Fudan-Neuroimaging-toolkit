"""Export source-bound joint precision continuation, retaining failed candidates."""
import argparse
import functools
import hashlib
import json
from pathlib import Path

RUN_FIELDS = ('status', 'wall_seconds', 'returncode', 'job_sha256', 'hostname',
              'max_cpu_threads', 'cpu_affinity', 'started_utc', 'finished_utc',
              'load_before', 'load_after', 'maximum_sampled_tree_rss_bytes',
              'maximum_sampled_tree_threads', 'resource_samples')


def gate(comparison, model):
    fields = {name: row['whole_grid']['max_abs'] <= 1e-3 and row['whole_grid']['rmse'] <= 1e-4
              for name, row in comparison['transforms'].items()}
    images = {}
    for name, row in comparison['images'].items():
        images[name] = {}
        for region in ('whole_grid', 'official_synthstrip_brain', 'upper_coordinate_boundary_band'):
            metric = row[region]
            normalized = metric['nrmse_reference_p99_minus_p1']
            images[name][region] = normalized <= 1e-3 if normalized is not None else metric['rmse'] == 0
    return {'fields': fields, 'images': images,
            'all_gates_passed': all(fields.values()) and all(all(row.values()) for row in images.values())}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate-version', type=int, default=17)
    parser.add_argument('--final-validation', action='store_true',
                        help='require completed current CPU timings, unchanged-mode and GPU/API regressions')
    args = parser.parse_args()
    base = Path(args.root)
    work = base / 'workspaces/smri_cpu_20261004/remaining_20261004/morph'
    runs = base / 'runs/smri_cpu_20261004/remaining_20261004/morph'

    def run(path):
        value = read(path)
        result = {key: value[key] for key in RUN_FIELDS if key in value}
        timing = Path(path).parent / 'time.txt'
        if timing.exists():
            # Retain GNU time's resource figures, excluding the private argv.
            result['GNU_time_resources'] = {
                line.strip().rsplit(': ', 1)[0]: line.strip().rsplit(': ', 1)[1]
                for line in timing.read_text().splitlines()
                if ': ' in line and not line.lstrip().startswith('Command being timed:')}
            result['GNU_time_file_sha256'] = digest(timing)
        return result

    def source(path):
        record = read(path)
        return verified_source(record['job']['source_manifest'])

    @functools.lru_cache(maxsize=None)
    def verified_source(filename):
        manifest = Path(filename)
        value = read(manifest)
        for relative, expected in value['files'].items():
            if digest(manifest.parent / relative) != expected:
                raise RuntimeError('frozen source changed: ' + relative)
        return {'manifest_sha256': digest(manifest), 'verified_files': len(value['files']),
                'base_commit': value.get('base_commit'),
                **{key: value[key] for key in ('full_CLI_mathematical_oracle', 'runtime_delta_from_v29') if key in value},
                'files': {key: sha for key, sha in value['files'].items()
                          if key.startswith('src/fnit/synthmorph/') or
                          key in ('src/fnit/_nib.py', 'src/fnit/_world_resampling.py',
                                  'src/fnit/_transforms.py')}}

    original = read(work / 'decode_v1/validation/synthmorph/cpu_20261004/report.public.json')
    report = {'schema': 'fnit.synthmorph.joint.cpu.precision.20261004.v1',
              'data': original['data'], 'fixed_gates': read(work / 'decode_v1/validation/synthmorph/cpu_20261004/acceptance.json'),
              'fixed_gates_sha256': digest(work / 'decode_v1/validation/synthmorph/cpu_20261004/acceptance.json'),
              'accepted_main': 'eea929d4', 'CPU_host': 'nodecw7',
              'physical_cores': [2, 6, 10, 14, 18, 22, 26, 30], 'threads': 8,
              'shared_node': True, 'OS_page_cache_flushed': False,
              'timing_scope': 'new-process CLI including all loading, computation and bidirectional saving; cached own Eigen binary; initial build separately measured',
              'failed_candidates': {}, 'candidate': {}, 'worker_sha256': digest(__file__)}
    versions = {10, 12, 15, 17, args.candidate_version}
    if args.candidate_version >= 29:
        versions.add(23)
    for version in sorted(versions):
        rows = {}
        for name in ('joint', 'joint_192'):
            path = runs / ('nodecw7_final_v' + str(version)) / name
            comparison = read(path / 'nodecw7_reference_comparison.private.json')
            rows[name] = {'extent': 256 if name == 'joint' else 192,
                          'hyper': .5 if name == 'joint' else .75,
                          'steps': 7 if name == 'joint' else 5,
                          'run': run(path / 'record.json'), 'source': source(path / 'record.json'),
                          'comparison': comparison, 'gate': gate(comparison, 'joint')}
        report['candidate' if version == args.candidate_version else 'failed_candidates'][str(version)] = rows
    report['candidate_all_CPU_gates_passed'] = all(row['gate']['all_gates_passed']
                                                  for row in report['candidate'][str(args.candidate_version)].values())
    report['same_feature_stages'] = {name: read(runs / 'nodecw7_fit_v14' / name / 'artifacts/report.private.json')
                                     for name in ('extent192', 'extent256')}
    report['same_feature_stages']['scope'] = 'Identical saved real features; this isolates affine arithmetic and is not a complete registration gate.'
    report['center_composition'] = {
        'original_same_halves': read(runs / 'nodecw7_center_v16/reference/artifacts/report.private.json'),
        'torch_controls': read(runs / 'nodecw7_center_v16/torch/artifacts/report.private.json'),
        'production': read(runs / 'nodecw7_center_v16/production_v17.private.json')}
    report['initial_build'] = run(runs / 'nodecw7_fit_v14/extent192/record.json')
    report['cached_stage'] = run(runs / 'nodecw7_fit_v14/extent256/record.json')
    builds = list((runs / 'eigen_build_cache_v14').glob('*/build.json'))
    if len(builds) != 1:
        raise RuntimeError('ambiguous Eigen build identity')
    build = read(builds[0]); identity = build['identity']
    report['build'] = {'eigen_version': '3.4.0', 'compiler': Path(identity['compiler_argv'][0]).name,
                       'compiler_sha256': identity['compiler_sha256'],
                       'compiler_version': identity['compiler_version'], 'flags': identity['flags'],
                       'source_sha256': identity['source_sha256'], 'binary_sha256': build['binary_sha256'],
                       'manifest_sha256': digest(builds[0]),
                       'headers_sha256': hashlib.sha256(json.dumps(identity['eigen_headers_sha256'], sort_keys=True).encode()).hexdigest(),
                       'dependency_scope': 'independent conda-forge Eigen3.4.0/GCC11; no FreeSurfer or TensorFlow headers/runtime'}
    if args.candidate_version >= 23:
        report['live_original'] = {}
        for extent in (192, 256):
            prefix = runs / 'nodecw7_live_reference_v18'
            report['live_original'][str(extent)] = {
                'diagnostic_run': run(prefix / ('joint_' + str(extent)) / 'record.json'),
                'tensor_observer': read(prefix / ('joint_' + str(extent)) / 'artifacts/report.private.json'),
                'formal_CLI_complete_byte_equality': read(prefix / ('observer_output_equality' + str(extent) + '.private.json')),
                'live_affine_arithmetic': read(runs / 'nodecw7_stage_v19' / ('affine_' + str(extent)) / 'artifacts/report.private.json'),
                'affine_detector_identical_inputs': read(runs / 'nodecw7_features_v21' / ('features' + str(extent)) / 'artifacts/report.private.json'),
                'production_network_inputs': read(runs / 'nodecw7_inputs_v22' / ('input' + str(extent)) / 'artifacts/report.private.json')}
        report['finite_first_layer'] = read(runs / 'nodecw7_stage_v19/first_layer_torch/artifacts/report.private.json')
        report['preprocessing_first_difference'] = read(runs / 'nodecw7_stage_v19/network_input/artifacts/report.private.json')
        for key, directory in (('finite_second_layer', 'nodecw7_second_layer_v24'),
                               ('finite_tail', 'nodecw7_tail_v25')):
            path = runs / directory
            report[key] = {name: {'run': run(path / name / 'record.json'),
                                 'operators': read(path / name / 'artifacts/report.private.json')}
                           for name in ('reference', 'torch')}
            report[key]['timing_scope'] = 'finite saved-real-input operator diagnostic, including interpreter/import/I/O; not a full registration benchmark'
        report['finite_tail_original_vs_live'] = read(runs / 'nodecw7_tail_v25/final_original_vs_live.private.json')
    if args.candidate_version >= 29:
        report['small_convolution_primitive'] = {
            'run': run(runs / 'nodecw7_small_backend_v26/torch/record.json'),
            'operators': read(runs / 'nodecw7_small_backend_v26/torch/artifacts/report.private.json')}
        report['explicit_primitive_features'] = {
            str(extent): {'run': run(runs / 'nodecw7_features_v27' / ('features' + str(extent)) / 'record.json'),
                          'detector': read(runs / 'nodecw7_features_v27' / ('features' + str(extent)) / 'artifacts/report.private.json')}
            for extent in (192, 256)}
        report['integrated_affine'] = {
            str(extent): {'run': run(runs / 'nodecw7_affine_input_v29' / ('extent' + str(extent)) / 'record.json'),
                          'source': source(runs / 'nodecw7_affine_input_v29' / ('extent' + str(extent)) / 'record.json'),
                          'stages': read(runs / 'nodecw7_affine_input_v29' / ('extent' + str(extent)) / 'artifacts/report.private.json')}
            for extent in (192, 256)}
        report['source_compatibility'] = read(runs / 'public_v29/source_compatibility.public.json')
        report['unit_tests'] = {'passed': 187,
                                'command': 'PYTHONPATH=src python -m pytest tests/synthmorph tests/applywarp/test_world_transform.py -q'}
    if args.final_validation:
        gpu = runs / 'gpu_joint_v29'
        report['GPU_joint'] = {
            'comparison': read(gpu / 'comparison.private.json'),
            'arms': {role: {'run': run(gpu / role / 'record.json'),
                            'source': source(gpu / role / 'record.json'),
                            'initialization': read(gpu / role / 'artifacts/initialization.private.json')}
                     for role in ('baseline', 'candidate')},
            'timing_scope': 'same current physical GPU1, shared external activity; early CUDA initialization is benchmark-only; no stable speed claim'}
        api_path = runs / 'nodecw7_accept_v29/joint_object_api'
        api = read(api_path / 'artifacts/report.private.json')
        api.pop('source_root', None)
        report['materialized_API'] = {'run': run(api_path / 'record.json'),
                                       'source': source(api_path / 'record.json'),
                                       'observer': api,
                                       'contract': read(api_path / 'artifacts/materialized.private.json')}
        paired = runs / 'nodecw7_accept_v29'
        report['current_adjacent_CPU_timing'] = {
            'order': ['A1_v7', 'C1_v29', 'R_reference', 'C2_v29', 'A2_v7'],
            'scope': 'same current eight physical cores and lock, fresh full CLI per arm; original reference central; page cache not flushed; shared memory/CPU/I/O contention recorded',
            'arms': {role: {'run': run(paired / role / 'record.json'),
                            **({'source': source(paired / role / 'record.json')} if role != 'R_reference' else {})}
                     for role in ('A1_v7', 'C1_v29', 'R_reference', 'C2_v29', 'A2_v7')}}
        report['current_adjacent_CPU_timing']['output_preservation'] = read(paired / 'timing_outputs.private.json')
        for row in report['current_adjacent_CPU_timing']['arms'].values():
            if row['run']['status'] != 'complete' or row['run']['returncode'] != 0:
                raise RuntimeError('current adjacent CPU timing is incomplete')
        report['unchanged_CPU_modes'] = {
            'comparison': read(paired / 'unchanged_modes.private.json'),
            'arms': {model: {'run': run(paired / model / 'record.json'),
                             'source': source(paired / model / 'record.json')}
                     for model in ('rigid', 'affine', 'deform')}}
        report['final_saved_affine_replay'] = read(paired / 'saved_affine_replay.private.json')
        report['brain_figure'] = read(runs / 'public_v29/cpu_joint_v29_brains.json')
        report['packaging'] = read(runs / 'public_v29/packaging.public.json')
        final = runs / 'nodecw7_guard_v30'
        report['final_inference_policy'] = {
            'scope': 'guard-only final source: eval/no-grad/FP32/standard Conv3d/no CPU autocast/no leaf or global hooks; other CPU callers retain accepted v7 arithmetic; CUDA unchanged',
            'source_compatibility': read(work / 'validation_v30/source_compatibility_v30.public.json'),
            'stages': {str(extent): {'run': run(final / ('extent' + str(extent)) / 'record.json'),
                                   'source': source(final / ('extent' + str(extent)) / 'record.json'),
                                   'stage': read(final / ('extent' + str(extent)) / 'artifacts/report.private.json')}
                       for extent in (192, 256)}}
        for extent, row in report['final_inference_policy']['stages'].items():
            preceding = report['integrated_affine'][extent]['stages']
            row['exact_v29_arrays_preserved'] = all(
                row['stage']['rows'][key]['actual_array_sha256'] == preceding['rows'][key]['actual_array_sha256']
                for key in preceding['rows'])
        report['finite_phase_cost'] = {
            str(extent): {'run': run(final / ('phase' + str(extent)) / 'record.json'),
                          'source': source(final / ('phase' + str(extent)) / 'record.json'),
                          'cost': read(final / ('phase' + str(extent)) / 'cost.private.json')}
            for extent in (192, 256)}
        report['all_final_gates_passed'] = (
            report['candidate_all_CPU_gates_passed'] and
            report['GPU_joint']['comparison']['all_gates_passed'] and
            report['materialized_API']['contract']['all_gates_passed'] and
            report['unchanged_CPU_modes']['comparison']['all_gates_passed'] and
            report['final_saved_affine_replay']['all_gates_passed'] and
            report['source_compatibility']['all_gates_passed'] and
            report['current_adjacent_CPU_timing']['output_preservation']['all_gates_passed'] and
            report['final_inference_policy']['source_compatibility']['all_gates_passed'] and
            all(row['exact_v29_arrays_preserved'] and row['stage']['CPU_inference_policy_enabled']
                for row in report['final_inference_policy']['stages'].values()))
        if not report['all_final_gates_passed']:
            raise RuntimeError('final CPU/GPU/API preservation gate failed')
        report['validation_only_plot_failure'] = {
            'run': run(final / 'figure/record.json'),
            'reason': 'inference environment has no matplotlib; reused registered independent plotting environment; no production change'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'candidate_all_CPU_gates_passed': report['candidate_all_CPU_gates_passed']}))


if __name__ == '__main__':
    main()
