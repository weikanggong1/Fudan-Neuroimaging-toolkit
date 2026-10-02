"""CPU-only audit of provenance-checked SynthSegPlus inputs and saved controls.

Example: python analyze_coarse_inputs.py --root /server/benchmark_root --output audit.json
The public defaced ds000114 example is hash-checked; no fitting or image export.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_labels(path, target):
    image = nib.load(path)
    matrix = np.linalg.inv(image.affine) @ target.affine
    return ndimage.affine_transform(np.asarray(image.dataobj, dtype=np.int32), matrix[:3, :3],
                                    offset=matrix[:3, 3], output_shape=target.shape,
                                    order=0, prefilter=False, output=np.int32)


def mask_metrics(reference, prediction):
    r, p = int(reference.sum()), int(prediction.sum())
    overlap = int((reference & prediction).sum())
    return {'reference_voxels': r, 'prediction_voxels': p, 'intersection_voxels': overlap,
            'dice': 2 * overlap / (r + p) if r + p else None,
            'reference_coverage': overlap / r if r else None}


def intensity(data, mask):
    values = data[mask & np.isfinite(data) & (data > 0)]
    if not len(values):
        return {'positive_voxels': 0, 'median': None, 'MAD': None}
    median = float(np.median(values))
    return {'positive_voxels': int(values.size), 'median': median,
            'MAD': float(np.median(np.abs(values - median)))}


def stage_summary(stage):
    history = stage.get('objective_history', [])
    keys = ('alpha_sigma_voxels', 'alpha_sigma_mm', 'resolution_mm', 'mesh_steps',
            'mesh_evaluations', 'mesh_iteration_limit', 'outer_iteration_limit', 'stable_mesh_fitting',
            'precise_mesh_matrices', 'mesh_line_search')
    row = {key: stage.get(key) for key in keys}
    row.update(history_length=len(history), first_cost=history[0] if history else None,
               final_cost=history[-1] if history else None,
               final_cost_repeated=history[-1] == history[-2] if len(history) > 1 else None,
               hit_total_mesh_iteration_budget=stage['mesh_steps'] >= (
                   stage['mesh_iteration_limit'] * stage.get('outer_iteration_limit', 1)),
               cost_history_sha256=hashlib.sha256(json.dumps(history).encode()).hexdigest())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root.parent
    control = root / 'raw_precision_controls_20261001'
    cache = control / 'shared_cache'
    record = json.loads((cache / 'preprocessing.json').read_text())
    raw_path = root.parent.parent / 'examples/data/sub-01_T1w.nii.gz'
    expected = 'f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a'
    assert raw_path.stat().st_size == 3847853 and sha256(raw_path) == expected
    assert record['input_sha256'] == expected
    for name in ('coarse', 'cortical', 'wmparc'):
        assert sha256(cache / f'{name}.nii.gz') == record['files'][name]['sha256']
    for name, weight in record['verified_weights'].items():
        path = root / 'weights' / name
        assert path.stat().st_size == weight['bytes'] and sha256(path) == weight['sha256']
    raw_image = nib.load(raw_path)
    data = np.asarray(raw_image.dataobj, dtype=np.float32)
    coarse = aligned_labels(cache / 'coarse.nii.gz', raw_image)
    cortical = aligned_labels(cache / 'cortical.nii.gz', raw_image)
    proxy = aligned_labels(cache / 'wmparc.nii.gz', raw_image)
    mri = work / 'reconall_reference_gpucw1/fs_sub01/mri'
    oracle = aligned_labels(mri / 'aseg.mgz', raw_image)
    oracle_wm = aligned_labels(mri / 'wmparc.mgz', raw_image)
    cases = ('synthseg_raw', 'official_raw', 'official_wm110', 'official_norm_native')
    analyses = {case: json.loads((control / case / 'analysis.json').read_text()) for case in cases}
    first = analyses[cases[0]]
    assert hashlib.sha256(coarse.tobytes()).hexdigest() == first['prepared_coarse_sha256']
    for case, analysis in analyses.items():
        assert analysis['source_sha256'] == first['source_sha256']
        assert analysis['validation_driver_sha256'] == first['validation_driver_sha256']
        for path in (raw_path, mri / 'aseg.mgz', mri / 'wmparc.mgz', mri / 'norm.mgz'):
            assert sha256(path) == analysis['input_sha256'][str(path)]
    voxel_size = np.linalg.norm(raw_image.affine[:3, :3], axis=0)
    width = max(256, math.ceil(float(np.max(np.asarray(raw_image.shape) * voxel_size))))
    if width > 256 and (width - 256) / 256 < 0.1:
        width = 256
    # Same default coronal geometry as recon_all/conform_gpu._coronal_affine;
    # derive the center from this NIfTI's own affine, not official grid metadata.
    center = raw_image.affine @ np.r_[np.asarray(raw_image.shape) / 2, 1]
    derived = np.eye(4)
    derived[:3, :3] = np.array([[-1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    derived[:3, 3] = center[:3] - derived[:3, :3] @ np.full(3, width / 2)
    corners = np.asarray(list(product((0, width - 1), repeat=3)))
    coronal_grid = {'width': width, 'spacing_mm': [1., 1., 1.],
                    'center_source': 'raw affine @ (raw shape / 2, 1)',
                    'intensity_processing': 'None; geometry audit only, no uint8 quantization',
                    'comparison': {}}
    for name in ('orig', 'norm'):
        reference_image = nib.load(mri / f'{name}.mgz')
        ras_delta = corners @ (derived[:3, :3] - reference_image.affine[:3, :3]).T + (
            derived[:3, 3] - reference_image.affine[:3, 3])
        coronal_grid['comparison'][name] = {
            'reference_shape': [int(n) for n in reference_image.shape],
            'maximum_affine_element_difference': float(np.max(np.abs(derived - reference_image.affine))),
            'maximum_corner_RAS_distance_mm': float(np.linalg.norm(ras_delta, axis=1).max()),
            'maximum_corner_RAS_coordinate_difference_mm': float(np.abs(ras_delta).max())}
    ball = ndimage.generate_binary_structure(3, 1)
    tissues = {}
    for label, name in ((10, 'left-thalamus'), (49, 'right-thalamus'),
                        (17, 'left-hippocampus'), (53, 'right-hippocampus'),
                        (18, 'left-amygdala'), (54, 'right-amygdala'),
                        (2, 'left-WM'), (41, 'right-WM')):
        r, p = oracle == label, coarse == label
        row = mask_metrics(r, p)
        if r.any() and p.any():
            delta = np.argwhere(p).mean(0) - np.argwhere(r).mean(0)
            row['centroid_distance_mm'] = float(np.linalg.norm(raw_image.affine[:3, :3] @ delta))
        row['reference_eroded_intensity'] = intensity(data, ndimage.binary_erosion(r, ball, border_value=1))
        row['predicted_eroded_intensity'] = intensity(data, ndimage.binary_erosion(p, ball, border_value=1))
        tissues[name] = row
    wm = {}
    competition = coarse.copy()
    parcel_sources = {}
    for side, white, lower, upper in (('left', 2, 1000, 2000), ('right', 41, 2000, 3000)):
        source = (cortical > lower) & (cortical < upper)
        parcel_sources[side] = [int(value) for value in np.unique(cortical[source])]
        white_mask = coarse == white
        points = np.argwhere(white_mask | source)
        low = np.maximum(points.min(0) - 1, 0)
        high = np.minimum(points.max(0) + 2, coarse.shape)
        crop = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        distance, nearest = ndimage.distance_transform_edt(
            ~source[crop], sampling=voxel_size, return_indices=True)
        labels = cortical[crop][tuple(nearest)] + 2000
        select = white_mask[crop] & (distance <= 15)
        competition[crop][select] = labels[select]
    for side, labels, white in (('left', (3006, 3007, 3016), 2),
                                ('right', (4006, 4007, 4016), 41)):
        reference, predicted = np.isin(oracle_wm, labels), np.isin(proxy, labels)
        row = mask_metrics(reference, predicted)
        row['outside_same_hemisphere_coarse_WM_voxels'] = int((predicted & (coarse != white)).sum())
        row['proxy_volume_mm3'] = int(predicted.sum()) * float(abs(np.linalg.det(raw_image.affine[:3, :3])))
        row['official_union_eroded_intensity'] = intensity(data, ndimage.binary_erosion(reference, ball, border_value=1))
        row['proxy_union_eroded_intensity'] = intensity(data, ndimage.binary_erosion(predicted, ball, border_value=1))
        row['per_label'] = {str(label): mask_metrics(oracle_wm == label, proxy == label) for label in labels}
        competing = np.isin(competition, labels)
        candidate = mask_metrics(reference, competing)
        candidate['all_same_side_cortical_source_ids'] = parcel_sources[side]
        candidate['outside_same_hemisphere_coarse_WM_voxels'] = int((competing & (coarse != white)).sum())
        candidate['union_eroded_intensity'] = intensity(data, ndimage.binary_erosion(competing, ball, border_value=1))
        candidate['per_label'] = {str(label): mask_metrics(oracle_wm == label, competition == label) for label in labels}
        row['candidate_all_cortical_competition_CPU_only'] = candidate
        wm[side] = row
    official = work / 'fnit_subregions_plus_20260928'
    supports = {}
    for family, source, ids, iterations in (
        ('thalamus', official / 'official_thalamus_gpucw1_full_sub01_20260929/ThalamicNuclei.FSvoxelSpace.mgz', (10, 49), 5),
        ('hippo-amygdala-left', official / 'official_hippo_gpucw1_full_sub01_20260929/lh.hippoAmygLabels.FSvoxelSpace.mgz', (17, 18), 2),
        ('hippo-amygdala-right', official / 'official_hippo_gpucw1_full_sub01_20260929/rh.hippoAmygLabels.FSvoxelSpace.mgz', (53, 54), 2)):
        fine = aligned_labels(source, raw_image)
        groups = ({'thalamus': (fine >= 8100) & (fine < 8300) & ~np.isin(fine, (8125, 8225))}
                  if family == 'thalamus' else
                  {'hippocampus': (fine >= 200) & (fine <= 246) & (fine != 201),
                   'amygdala': (fine >= 7000) & (fine < 8000)})
        p = ndimage.binary_dilation(np.isin(coarse, ids), iterations=iterations)
        r = ndimage.binary_dilation(np.isin(oracle, ids), iterations=iterations)
        supports[family] = {name: {'reference_fine_voxels': int(mask.sum()),
                                  'excluded_by_current_synthseg_support': int((mask & ~p).sum()),
                                  'excluded_by_official_coarse_support': int((mask & ~r).sum())}
                             for name, mask in groups.items()}
    traces = {}
    for case, analysis in analyses.items():
        initialization = analysis['initialization']['thalamus']
        segmentation = initialization['segmentation_fit']
        hypers = [{'means': h['means'], 'counts': h['counts']} for h in analysis['hyperparameters']]
        traces[case] = {'precision_metrics': analysis['metrics'],
                        'alignment_dice': initialization['alignment_dice'],
                        'synthetic_min_jacobian': segmentation['min_jacobian'],
                        'synthetic_mean_displacement_voxels': segmentation['mean_displacement_voxels'],
                        'synthetic_stages': [stage_summary(s) for s in segmentation['mesh_solver']['stages']],
                        'hyperparameters': hypers,
                        'intensity_stages': [stage_summary(s) for s in initialization['mesh_solver']['stages']],
                        'prepared_coarse_sha256': analysis['prepared_coarse_sha256'],
                        'prepared_data_sha256': analysis['prepared_data_sha256'],
                        'analysis_json_sha256': sha256(control / case / 'analysis.json')}
    result = {'kind': 'CPU_only_current_provenance_coarse_and_control_trace_audit',
              'analysis_script_sha256': sha256(__file__), 'public_example_sha256': expected,
              'cache_manifest_sha256': sha256(cache / 'preprocessing.json'),
              'cache_file_sha256': {name: record['files'][name]['sha256'] for name in ('coarse', 'cortical', 'wmparc')},
              'cache_metadata': record['metadata'], 'verified_weight_records': record['verified_weights'],
              'runtime_source_manifest_sha256': hashlib.sha256(json.dumps(first['source_sha256'], sort_keys=True).encode()).hexdigest(),
              'validation_driver_sha256': first['validation_driver_sha256'],
              'native_spacing_mm': voxel_size.tolist(), 'coarse_tissues': tissues,
              'derived_default_coronal_1mm_grid': coronal_grid,
              'temporal_WM_proxy': wm, 'official_fine_label_support_coverage': supports,
              'controlled_thalamus_fits': traces,
              'limitations': ['This is an input/trace audit, not a fit-performance benchmark.',
                              'Uses only the new provenance-checked SynthSegPlus shared_cache, not historical arrays.',
                              'Supplied controls use official coarse on raw grid. They do not isolate physical-grid effects without a separate grid intervention.',
                              'Fine-label support exclusions are upper-bound input-mask checks; fitting can move labels inside or outside those masks.']}
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
