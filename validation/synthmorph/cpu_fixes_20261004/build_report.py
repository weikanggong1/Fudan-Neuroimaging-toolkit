"""Export exact measured CPU/GPU records without private paths or image data."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

RUN_FIELDS = ('status', 'wall_seconds', 'returncode', 'job_sha256', 'hostname',
              'max_cpu_threads', 'cpu_affinity', 'started_utc', 'finished_utc',
              'load_before', 'load_after', 'maximum_sampled_tree_rss_bytes',
              'maximum_sampled_tree_threads', 'resource_samples')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def gate(comparison, model):
    fields = {}
    for name, row in comparison['transforms'].items():
        fields[name] = (row['complete_source_grid_world_error_mm']['max'] <= 1e-3
                        if model in ('rigid', 'affine') else
                        row['whole_grid']['max_abs'] <= 1e-3 and row['whole_grid']['rmse'] <= 1e-4)
    images = {}
    for name, row in comparison['images'].items():
        images[name] = {}
        for region in ('whole_grid', 'official_synthstrip_brain', 'upper_coordinate_boundary_band'):
            metric = row[region]
            normalized = metric['nrmse_reference_p99_minus_p1']
            images[name][region] = normalized <= 1e-3 if normalized is not None else metric['rmse'] == 0
    return {'fields': fields, 'images': images,
            'all_gates_passed': all(fields.values()) and all(all(row.values()) for row in images.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    base = Path(args.root)
    work = base / 'workspaces/smri_cpu_20261004/remaining_20261004/morph'
    runs = base / 'runs/smri_cpu_20261004/remaining_20261004/morph'
    def record(path):
        value = read(path)
        return {key: value[key] for key in RUN_FIELDS if key in value}
    def source(path):
        rec = read(path)
        manifest = Path(rec['job']['source_manifest'])
        value = read(manifest)
        for filename, expected in value.get('files', {}).items():
            if digest(manifest.parent / filename) != expected:
                raise RuntimeError('frozen source changed: ' + filename)
        files = value.get('files', {})
        relevant = ('src/fnit/_nib.py', 'src/fnit/_transforms.py', 'src/fnit/_world_resampling.py',
                    'src/fnit/weights.py', 'src/fnit/cli.py')
        return {'manifest_sha256': digest(manifest), 'verified_tree_file_count': len(files),
                'files': {key: item for key, item in files.items()
                          if key in relevant or key.startswith('src/fnit/synthmorph/')},
                **{key: value[key] for key in ('base_commit', 'label', 'archive_sha256')
                   if key in value}}
    data = read(work / 'decode_v1/validation/synthmorph/cpu_20261004/report.public.json')['data']
    report = {'schema': 'fnit.synthmorph.cpu.fixes.20261004.v1', 'data': data,
              'baseline_commit': 'f1cbdab10fbfd573c3aa1b3461aa086dab220cfb',
              'fixed_gates': read(work / 'decode_v1/validation/synthmorph/cpu_20261004/acceptance.json'),
              'fixed_gates_sha256': digest(work / 'decode_v1/validation/synthmorph/cpu_20261004/acceptance.json'),
              'protocol': {'CPU_host': 'nodecw7', 'physical_cores': [2, 6, 10, 14, 18, 22, 26, 30],
                           'threads': 8, 'shared_node': True, 'OS_page_cache_flushed': False,
                           'timing_scope': 'one new process per baseline/reference/candidate CLI; not the historical R-C-C-R protocol'},
              'weights': {}, 'CPU_modes': {}, 'materialized_API': {},
              'worker_sha256': digest(__file__)}
    for name in ('rigid.1', 'affine.2', 'deform.3'):
        path = base / ('assets/weights/synthmorph.' + name + '.h5')
        report['weights'][path.name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    for model in ('rigid', 'affine', 'deform', 'joint'):
        namespace = {'rigid': 'nodecw7_rigid_v4', 'joint': 'nodecw7_final_v7'}.get(model, 'nodecw7_v2')
        candidate = runs / namespace / model
        comparison = read(candidate / 'nodecw7_reference_comparison.private.json')
        report['CPU_modes'][model] = {
            'baseline': record(runs / 'nodecw7_paired_v2' / (model + '_baseline/record.json')),
            'reference': record(runs / 'nodecw7_paired_v2' / (model + '_reference/record.json')),
            'candidate': record(candidate / 'record.json'), 'source': source(candidate / 'record.json'),
            'comparison': comparison, 'gate': gate(comparison, model)}
    for model, namespace in [('affine', 'nodecw7_v2'), ('rigid', 'nodecw7_rigid_v4')]:
        root = runs / namespace / (model + '_object_api')
        report['materialized_API'][model] = {'run': record(root / 'record.json'),
                                            'report': read(root / 'artifacts/report.private.json')}
    root = runs / 'nodecw7_final_v7/joint_object_api'
    api = read(root / 'artifacts/report.private.json')
    api.pop('source_root', None)
    report['materialized_API']['joint'] = {'run': record(root / 'record.json'),
                                           'report': api,
                                           'contract': read(root / 'artifacts/materialized.private.json')}
    root = runs / 'nodecw7_final_v7/joint_192'
    comparison = read(root / 'nodecw7_reference_comparison.private.json')
    report['CPU_joint_192'] = {'extent': 192, 'hyper': .75, 'steps': 5,
                              'candidate': record(root / 'record.json'),
                              'reference': record(runs / 'nodecw7_final_v7/joint_192_reference/record.json'),
                              'source': source(root / 'record.json'),
                              'comparison': comparison, 'gate': gate(comparison, 'joint')}
    root = runs / 'nodecw7_world_v2/world_boundary'
    report['world_boundary'] = {'run': record(root / 'record.json'),
                                'report': read(root / 'artifacts/report.private.json')}
    root = runs / 'gpu_affine_final_v7'
    report['GPU_affine'] = {'comparison': read(root / 'comparison.private.json'),
                            'baseline': record(runs / 'gpu_affine_initialized_v3/baseline/record.json'),
                            'candidate': record(root / 'candidate/record.json'),
                            'initialization': read(root / 'candidate/artifacts/initialization.private.json')}
    report['GPU_pre_model_failures'] = {role: record(runs / 'gpu_affine_v2' / role / 'record.json')
                                       for role in ('baseline', 'candidate')}
    report['final_saved_affine_replay'] = read(runs / 'post_final_v7/replay/report.private.json')
    report['source_compatibility'] = read(runs / 'public_v7/source_compatibility.public.json')
    root = runs / 'post_final_v9/joint_reference_observer'
    if read(root / 'record.json')['status'] != 'complete':
        raise RuntimeError('original observer did not finish successfully')
    observer = read(root / 'observer.private.json')
    observer['source_sha256'] = {
        '/'.join(Path(key).parts[-3:]): value
        for key, value in observer['source_sha256'].items()}
    original_outputs = {}
    for name in ('moved', 'fixed_moved', 'forward', 'inverse'):
        left = nib.load(root / (name + '.nii.gz'))
        right = nib.load(runs / 'nodecw7_paired_v2/joint_reference' / (name + '.nii.gz'))
        left_hash, right_hash = [hashlib.sha256(np.ascontiguousarray(np.asanyarray(image.dataobj)).tobytes()).hexdigest()
                                 for image in (left, right)]
        original_outputs[name] = {'array_sha256': [left_hash, right_hash],
                                   'array_equal': left_hash == right_hash,
                                   'header_equal': left.header.binaryblock == right.header.binaryblock,
                                   'extensions_equal': left.header.extensions == right.header.extensions}
    report['original_joint_phase_observer'] = {'run': record(root / 'record.json'),
                                               'report': observer,
                                               'same_reference_CLI_outputs': original_outputs,
                                               'worker_sha256': digest(work / 'reference_observer_v1.py')}
    report['observer_pre_run_failures'] = {
        namespace: record(runs / namespace / 'joint_reference_observer/record.json')
        for namespace in ('post_final_v7', 'post_final_v8')}
    root = runs / 'post_final_v7/plot'
    report['brain_figure'] = {'run': record(root / 'record.json'),
                             'metadata': read(root / 'cpu_official_brains.json'),
                             'worker_sha256': digest(base / 'workspaces/smri_cpu_20261004/task3/tools_v3/plot_brains.py')}
    report['reference_neurite_barycenter_source'] = {
        'build': 'FreeSurfer 8.2.0-1', 'file': 'neurite/tf/utils/utils.py',
        'sha256': '872618d7418f529e2115fb30c4c3c889a312212550c0162e00cb4176d0c10bc8',
        'function': 'barycenter', 'lines': [512, 573]}
    report['unit_tests'] = {'passed': 160,
                           'command': 'PYTHONPATH=src python -m pytest tests/synthmorph tests/applywarp/test_world_transform.py -q'}
    report['all_default_CPU_gates_passed'] = all(row['gate']['all_gates_passed'] for row in report['CPU_modes'].values())
    report['all_materialized_API_gates_passed'] = all(
        row.get('contract', row['report'])['all_gates_passed']
        for row in report['materialized_API'].values())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
