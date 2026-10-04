#!/usr/bin/env python3
"""File-format/layout refusal controls; these two-frame fixtures are not MRI benchmarks."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import nibabel as nib
from nibabel.cifti2.cifti2_axes import SeriesAxis
import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--metric-helper', type=Path, required=True, help='Actual server path; original bindings retain the local path and pinned SHA.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.code))
    collector = load_module(args.code / 'collect_continuous.py', 'layout_collector')
    endpoint = load_module(args.code / 'endpoint_compare.py', 'layout_endpoint')
    bindings = json.loads(args.bindings.read_text())
    bindings['metric_helper'] = {'path': str(args.metric_helper.resolve(strict=True)),
                                 'sha256': bindings['metric_helper']['sha256']}
    metric = load_module(Path(bindings['metric_helper']['path']), 'layout_metric')
    assert endpoint.sha256(args.code / 'endpoint_compare.py') == collector.ENDPOINT_SHA
    assert endpoint.sha256(bindings['metric_helper']['path']) == collector.METRIC_SHA
    roots = [args.code, args.bindings.parent, Path(bindings['metric_helper']['path']).parent]
    roots += [Path(x['path']).parent for x in bindings['cifti_axis_assets'].values()]
    base = collector.fresh_output(args.output, roots)
    base.mkdir(parents=True, exist_ok=False)
    inputs = base / 'inputs'
    inputs.mkdir()
    axis, _ = metric.fixed_brain_axis(bindings['cifti_axis_assets'])
    shape, affine = metric.volume_grid(type('Grid', (), {'shape': axis.volume_shape, 'affine': axis.affine})())

    def nifti(path, data, matrix):
        image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), matrix)
        image.header.set_xyzt_units('mm', 'sec')
        if image.ndim == 4:
            image.header.set_zooms((*image.header.get_zooms()[:3], 1.0))
        nib.save(image, path)
        return collector.entry(path)

    raw = {'t1w': nifti(inputs / 'raw_t1w.nii.gz', np.ones((2, 2, 2)), np.eye(4)),
           'bold': nifti(inputs / 'raw_bold.nii.gz', np.broadcast_to([1, 2], (2, 2, 2, 2)), np.eye(4))}
    mask = np.zeros(shape, dtype=np.uint8)
    mask[tuple(np.asarray(shape) // 2)] = 1
    mask_entry = nifti(inputs / 'mask.nii.gz', mask, affine)
    volume = nifti(inputs / 'volume.nii.gz', np.broadcast_to([1, 2], (*shape, 2)), affine)
    data = np.broadcast_to(np.asarray([[1], [2]], dtype=np.float32), (2, 91282))
    cifti = nib.Cifti2Image(data.copy(), nib.Cifti2Header.from_axes((SeriesAxis(0, 1, 2), axis)))
    cifti_path = inputs / 'series.dtseries.nii'
    nib.save(cifti, cifti_path)
    report_path = inputs / 'fixture_report.json'
    collector.write(report_path, {'status': 'scientific_complete', 'case_id': 'LAYOUT_FIXTURE',
                                  'frames': 2, 'tr_seconds': 1.0,
                                  'raw_hashes': {k: v['sha256'] for k, v in raw.items()},
                                  'fixture_scope': 'file_format_control_not_MRI_benchmark'})
    spec = {'label': 'FILE_FIXTURE', 'report': collector.entry(report_path),
            'report_fields': {'case_id': 'case_id', 'frames': 'frames', 'tr_seconds': 'tr_seconds', 'raw_hashes': 'raw_hashes'},
            'outputs': {'preproc_mni': volume, 'dtseries': collector.entry(cifti_path)}}
    manifest = {'case_id': 'LAYOUT_FIXTURE', 'cohort_id': 'layout_control', 'frames': 2, 'tr_seconds': 1.0,
                'raw': raw, 'brain_mask': mask_entry, 'cifti_axis_assets': bindings['cifti_axis_assets'],
                'metric_helper': bindings['metric_helper'], 'candidate': spec, 'reference': copy.deepcopy(spec)}
    controls = []
    good = base / 'good'
    (good / 'manifests').mkdir(parents=True)
    manifest_path = good / 'manifests' / 'fixture.private.json'
    collector.write(manifest_path, manifest)
    manifest['_manifest_entry'] = collector.entry(manifest_path)
    good_target = good / 'endpoints' / 'LAYOUT_FIXTURE' / 'middle'
    result = endpoint.compare(manifest, good_target)
    assert result['status'] == 'comparison_complete' and result['input_guards_equal']
    assert result['volume']['normalized_rmse'] == 0 and result['cifti']['normalized_rmse'] == 0
    assert (good_target / 'comparison.public.json').is_file() and (good_target / 'metrics.private.npz').is_file()
    record = collector.save_endpoint_result(result, good_target, {}, good / 'unused_failure.public.json')
    assert collector.completion_code([record]) == 0 and not (good / 'unused_failure.public.json').exists()
    controls.append('sibling_manifest_endpoint_full91k_fixture_saved_exit0')

    # Reproduce the exact old layout failure without writing anything into the rejected endpoint tree.
    bad = base / 'bad'
    bad.mkdir()
    bad_manifest = bad / 'fixture.private.json'
    payload = copy.deepcopy(manifest)
    payload.pop('_manifest_entry')
    collector.write(bad_manifest, payload)
    payload['_manifest_entry'] = collector.entry(bad_manifest)
    bad_target = bad / 'LAYOUT_FIXTURE' / 'middle'
    rejected = endpoint.compare(payload, bad_target)
    assert rejected['status'] == 'validation_failed' and rejected['failure']['code'] == 'output_overlaps_original_input'
    assert not bad_target.exists() and rejected['input_guards_equal']
    failure_path = bad / 'preflight_failure.public.json'
    record = collector.save_endpoint_result(rejected, bad_target, {}, failure_path)
    assert collector.completion_code([record]) == 2 and record['endpoint_output_created'] is False
    assert json.loads(failure_path.read_text())['failure']['code'] == 'output_overlaps_original_input'
    controls.append('old_layout_rejected_no_endpoint_created_failure_saved_exit2')

    wrong = copy.deepcopy(manifest)
    wrong['candidate']['outputs']['dtseries']['sha256'] = '0' * 64
    wrong_target = good / 'endpoints' / 'LAYOUT_FIXTURE' / 'wrong_sha'
    rejected = endpoint.compare(wrong, wrong_target)
    assert rejected['status'] == 'validation_failed' and rejected['failure']['code'] == 'input_sha256_mismatch'
    assert not wrong_target.exists()
    record = collector.save_endpoint_result(rejected, wrong_target, {}, good / 'wrong_sha.public.json')
    assert collector.completion_code([record]) == 2
    controls.append('bad_bound_output_sha_rejected_before_mkdir_failure_saved_exit2')
    assert collector.completion_code([{'status': 'pending'}]) == 3
    controls.append('pending_is_exit3_not_complete')
    summary = {'status': 'passed', 'scope': 'actual_nifti_cifti_file_format_and_output_layout_controls_not_MRI_benchmark',
               'fixture_frames': 2, 'fixture_cifti_points': 91282, 'controls': controls,
               'endpoint_sha256': endpoint.sha256(args.code / 'endpoint_compare.py'),
               'collector_sha256': endpoint.sha256(args.code / 'collect_continuous.py'),
               'metric_sha256': endpoint.sha256(bindings['metric_helper']['path'])}
    collector.write(base / 'controls.public.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
