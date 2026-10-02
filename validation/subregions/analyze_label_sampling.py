"""CPU-only sampling audit of a verified public FNIT example; no image export."""
from pathlib import Path
import argparse
from itertools import product
import hashlib
import json
import numpy as np
import nibabel as nib
from scipy import ndimage

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Existing FNIT subregions benchmark root with weights and outputs')
    parser.add_argument('--stage-run', type=Path, help='Saved full_stage run; default: root/segment4_backtracking_final_full_stage_20261001')
    parser.add_argument('--output', type=Path, help='Write only aggregate JSON; default: stdout')
    args = parser.parse_args()
    work = args.root.resolve().parent
    raw = work.parent / 'examples/data/sub-01_T1w.nii.gz'
    expected = 'f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a'
    assert raw.stat().st_size == 3847853
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == expected
    official = work / 'fnit_subregions_plus_20260928'
    run = args.stage_run.resolve() if args.stage_run else args.root.resolve() / 'segment4_backtracking_final_full_stage_20261001'
    api = json.loads((run / 'api_report.json').read_text())
    meta = api['labels']
    native_image = nib.load(run / 'subregions_native.nii.gz')
    native_array = np.asarray(native_image.dataobj, dtype=np.int32)
    validation = json.loads((run / 'report.json').read_text())
    known = {r['id']: r for c in validation['comparisons'].values() for r in c['regions']}
    base_image = nib.load(api['input'])
    base_coarse = np.asarray(nib.load(work / 'reconall_reference_gpucw1/fs_sub01/mri/aseg.mgz').dataobj, dtype=np.int32)

    def load(path):
        image = nib.load(path)
        return image, np.asarray(image.dataobj, dtype=np.int32)

    def resample(array, affine, target_affine, target_shape):
        transform = np.linalg.inv(affine) @ target_affine
        return ndimage.affine_transform(array, transform[:3, :3], offset=transform[:3, 3],
                                        output_shape=target_shape, order=0, mode='constant',
                                        cval=0, prefilter=False, output=np.int32)

    def dice_metrics(reference, prediction, identifiers):
        max_id = max(int(reference.max()), int(prediction.max()), max(identifiers))
        rc = np.bincount(reference.ravel(), minlength=max_id + 1)
        pc = np.bincount(prediction.ravel(), minlength=max_id + 1)
        matched = reference[reference == prediction]
        ic = np.bincount(matched.ravel(), minlength=max_id + 1)
        rows = []
        for label in identifiers:
            r, p, i = int(rc[label]), int(pc[label]), int(ic[label])
            rows.append({'label': label, 'reference_voxels': r, 'prediction_voxels': p,
                         'intersection_voxels': i, 'dice': 2 * i / (r + p) if r + p else None})
        valid = [r for r in rows if r['dice'] is not None]
        ref_n = sum(r['reference_voxels'] for r in valid)
        return {'rows': rows, 'mean_dice': float(np.mean([r['dice'] for r in valid])),
                'reference_voxel_weighted_dice': sum(r['dice'] * r['reference_voxels']
                                                    for r in valid) / ref_n if ref_n else None}

    def union_grid(first, second):
        mapped = []
        for image in (first, second):
            corners = np.array(list(product(*[(0, int(n) - 1) for n in image.shape])))
            transform = np.linalg.inv(first.affine) @ image.affine
            mapped.append(corners @ transform[:3, :3].T + transform[:3, 3])
        points = np.concatenate(mapped)
        low = np.floor(points.min(0) - 1e-6).astype(int)
        high = np.ceil(points.max(0) + 1e-6).astype(int)
        affine = first.affine.copy()
        affine[:3, 3] += affine[:3, :3] @ low
        return affine, tuple((high - low + 1).tolist())

    def row_map(metrics):
        return {r['label']: r for r in metrics['rows']}

    result = {
        'kind': 'CPU_only_public_example_high_resolution_sampling_audit',
        'public_example': {'dataset': 'OpenNeuro ds000114', 'license': 'CC0',
                           'defaced_published_example_sha256': expected,
                           'published_bytes_verified': 3847853},
        'run': run.name,
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method': {
            'labels': 'Existing saved high-resolution hard labels; no fitting or GPU execution.',
            'geometry': 'Scanner-RAS voxel-to-world affine; nearest neighbor, order=0, constant background.',
            'common_highres': 'Official high-resolution orientation/spacing, integer-expanded union FOV of both inputs; no clipping to official ROI.',
            'native_grid': 'The exact official FSvoxelSpace grid, preserved for both methods.',
            'roundtrip': 'Highres hard labels -> official 1mm grid -> same common highres grid; self Dice measures information lost by hard downsampling.',
            'phase_probe': 'All 27 fixed 1mm grid phases from {-0.25,0,+0.25} voxel per axis, sampled equally for both inputs; not a registration search or new benchmark.',
        },
        'families': {},
        'limitations': ['Highres hard labels are not posterior volumes.',
                        'Highres/native Dice changes and self roundtrip loss do not establish that remaining deformation differences are correct.',
                        'Only one existing public example and the frozen full_stage run are evaluated.'],
    }

    def native_sampling_diagnostic(family, image, array, identifiers):
        report = api['initialization'][family]['working_image']
        low = np.asarray(report['crop_start_native'])
        high = np.asarray(report['crop_stop_native'])
        shape = np.asarray(report['working_shape'])
        step = report['resolution_mm'] / np.linalg.norm(base_image.affine[:3, :3], axis=0)
        origin = low + ((high - low) - shape * step) / 2
        transform = np.eye(4)
        transform[:3, :3] = np.diag(step)
        transform[:3, 3] = origin
        working = base_image.affine @ transform
        fit_crop = np.rint((np.linalg.inv(working) @ image.affine)[:3, 3]).astype(int)
        exact = working.copy()
        exact[:3, 3] += working[:3, :3] @ fit_crop
        assert np.max(np.abs(exact - image.affine)) < 1e-4
        support_ids = (10, 49) if family == 'thalamus' else (17, 18) if family.endswith('left') else (53, 54)
        support = ndimage.binary_dilation(np.isin(base_coarse, support_ids), iterations=5 if family == 'thalamus' else 2)
        saved = np.where(np.isin(native_array, identifiers), native_array, 0)
        metrics = {}
        for role, affine in [('reloaded_NIfTI_affine', image.affine), ('reconstructed_runtime_float64_affine', exact)]:
            pred = resample(array, affine, base_image.affine, base_image.shape)
            masked = np.where(support, pred, 0)
            selected = np.where(np.isin(pred, identifiers), pred, 0)
            selected_masked = np.where(np.isin(masked, identifiers), masked, 0)
            offset = 10000 if family.endswith('right') else 0
            targets = (7006, 7010, 7007, 203, 240, 243, 244) if family != 'thalamus' else (8110, 8130)
            metrics[role] = {'family_label_changed_voxels_before_support': int(np.count_nonzero(selected != saved)),
                             'family_label_changed_voxels_after_support': int(np.count_nonzero(selected_masked != saved)),
                             'support_removed_family_voxels': int(np.count_nonzero((selected > 0) & ~support)),
                             'target_labels': [{'label': label + offset,
                                'unmasked': int(np.count_nonzero(pred == label + offset)),
                                'support_masked': int(np.count_nonzero(masked == label + offset)),
                                'saved_native': int(np.count_nonzero(native_array == label + offset))} for label in targets]}
        return {'reconstructed_runtime_grid_contract': 'Original working/crop geometry and integer fit crop inferred from highres header, max difference checked <1e-4mm',
                'max_affine_rounding_mm': float(np.max(np.abs(exact - image.affine))), 'metrics': metrics}

    families = [
        ('thalamus', official / 'official_thalamus_gpucw1_full_sub01_20260929', 'ThalamicNuclei', 0),
        ('hippo-amygdala-left', official / 'official_hippo_gpucw1_full_sub01_20260929', 'lh.hippoAmygLabels', 0),
        ('hippo-amygdala-right', official / 'official_hippo_gpucw1_full_sub01_20260929', 'rh.hippoAmygLabels', 10000),
    ]
    for family, folder, stem, offset in families:
        oi, oa = load(folder / f'{stem}.mgz')
        ni, na = load(folder / f'{stem}.FSvoxelSpace.mgz')
        fi, fa = load(run / f'highres/{family}.nii.gz')
        saved_highres = fa
        if offset:
            fa = np.where(fa > 0, fa - offset, 0).astype(np.int32)
        ids = sorted(int(k) - offset for k, m in meta.items() if m['source'] == family)
        target_a, target_s = union_grid(oi, fi)
        oh = resample(oa, oi.affine, target_a, target_s)
        fh = resample(fa, fi.affine, target_a, target_s)
        on = resample(oa, oi.affine, ni.affine, ni.shape)
        fn = resample(fa, fi.affine, ni.affine, ni.shape)
        saved = native_array if not offset else np.where(native_array >= offset, native_array - offset, 0)
        saved = np.where(np.isin(saved, ids), saved, 0).astype(np.int32)
        saved = resample(saved, native_image.affine, ni.affine, ni.shape)
        official_round = resample(on, ni.affine, target_a, target_s)
        fnit_round = resample(fn, ni.affine, target_a, target_s)
        high = dice_metrics(oh, fh, ids)
        hard = dice_metrics(na, saved, ids)
        direct = dice_metrics(on, fn, ids)
        self_off = dice_metrics(oh, official_round, ids)
        self_fn = dice_metrics(fh, fnit_round, ids)
        official_consistency = dice_metrics(na, on, ids)
        fnit_consistency = dice_metrics(saved, fn, ids)
        hm, nm, dm, om, fm = map(row_map, (high, hard, direct, self_off, self_fn))
        points = np.argwhere(np.isin(on, ids) | np.isin(fn, ids))
        low = np.maximum(points.min(0) - 3, 0)
        high_native = np.minimum(points.max(0) + 4, ni.shape)
        phase_shape = tuple((high_native - low).tolist())
        phase_affine = ni.affine.copy()
        phase_affine[:3, 3] += phase_affine[:3, :3] @ low
        phase_values = {label: [] for label in ids}
        for shift in product((-0.25, 0.0, 0.25), repeat=3):
            shifted = phase_affine.copy()
            shifted[:3, 3] += shifted[:3, :3] @ np.asarray(shift)
            rs = resample(oa, oi.affine, shifted, phase_shape)
            ps = resample(fa, fi.affine, shifted, phase_shape)
            for row in dice_metrics(rs, ps, ids)['rows']:
                if row['dice'] is not None:
                    phase_values[row['label']].append(row['dice'])
        voxel_volume = float(abs(np.linalg.det(target_a[:3, :3])))
        cr = ndimage.center_of_mass(np.ones(target_s, np.float32), oh, ids)
        cp = ndimage.center_of_mass(np.ones(target_s, np.float32), fh, ids)
        label_rows = []
        for index, label in enumerate(ids):
            m = meta[str(label + offset)]
            r, p = hm[label]['reference_voxels'], hm[label]['prediction_voxels']
            distance = float(np.linalg.norm(target_a[:3, :3] @ (np.asarray(cr[index]) - cp[index]))) if r and p else None
            phases = phase_values[label]
            label_rows.append({'label': label, 'output_label': label + offset, 'name': m['name'],
                              'parent': m['parent'], 'highres_reference_voxels': r,
                              'highres_prediction_voxels': p,
                              'highres_reference_hard_mm3': r * voxel_volume,
                              'highres_prediction_hard_mm3': p * voxel_volume,
                              'highres_hard_volume_fraction_difference': (p-r)/r if r else None,
                              'highres_dice': hm[label]['dice'],
                              'official_1mm_reference_voxels': nm[label]['reference_voxels'],
                              'saved_1mm_prediction_voxels': nm[label]['prediction_voxels'],
                              'saved_1mm_dice': nm[label]['dice'],
                              'both_highres_to_1mm_dice': dm[label]['dice'],
                              'official_self_1mm_roundtrip_dice': om[label]['dice'],
                              'fnit_self_1mm_roundtrip_dice': fm[label]['dice'],
                              'centroid_distance_mm': distance,
                              'fixed_phase_dice_min': min(phases) if phases else None,
                              'fixed_phase_dice_median': float(np.median(phases)) if phases else None,
                              'fixed_phase_dice_max': max(phases) if phases else None,
                              'phase_count': len(phases)})
        result['families'][family] = {
            'official_original_highres_exists': True,
            'official_highres_shape': [int(n) for n in oi.shape],
            'official_spacing_mm': np.linalg.norm(oi.affine[:3, :3], axis=0).tolist(),
            'fnit_highres_shape': [int(n) for n in fi.shape],
            'fnit_spacing_mm': np.linalg.norm(fi.affine[:3, :3], axis=0).tolist(),
            'comparison_union_highres_shape': list(target_s),
            'input_sha256_by_role': {'official_highres': hashlib.sha256((folder/f'{stem}.mgz').read_bytes()).hexdigest(),
                                     'official_native': hashlib.sha256((folder/f'{stem}.FSvoxelSpace.mgz').read_bytes()).hexdigest(),
                                     'fnit_highres': hashlib.sha256((run/f'highres/{family}.nii.gz').read_bytes()).hexdigest()},
            'highres_mean_dice': high['mean_dice'],
            'highres_reference_voxel_weighted_dice': high['reference_voxel_weighted_dice'],
            'saved_1mm_mean_dice': hard['mean_dice'],
            'saved_1mm_reference_voxel_weighted_dice': hard['reference_voxel_weighted_dice'],
            'official_highres_to_1mm_consistency_mean_dice': official_consistency['mean_dice'],
            'official_highres_to_1mm_consistency_weighted_dice': official_consistency['reference_voxel_weighted_dice'],
            'fnit_highres_to_saved_1mm_consistency_weighted_dice': fnit_consistency['reference_voxel_weighted_dice'],
            'label_metrics': label_rows,
            'native_sampling_consistency_diagnostic': native_sampling_diagnostic(family, fi, saved_highres, [label + offset for label in ids]),
        }
    for family, values in result['families'].items():
        all_rows = values['label_metrics']
        values['highres_evaluated_labels'] = sum(r['highres_dice'] is not None for r in all_rows)
        values['saved_1mm_evaluated_labels'] = sum(r['saved_1mm_dice'] is not None for r in all_rows)
        aggregates = {}
        for parent in sorted({r['parent'] for r in all_rows}):
            rows = [r for r in all_rows if r['parent'] == parent]
            hr = [r for r in rows if r['highres_dice'] is not None]
            nr = [r for r in rows if r['saved_1mm_dice'] is not None]
            shared = [r for r in rows if r['highres_dice'] is not None and r['saved_1mm_dice'] is not None]
            def aggregate(selected, cost, ref):
                denominator = sum(r[ref] for r in selected)
                return {'evaluated_labels': len(selected), 'reference_voxels': denominator,
                        'mean_dice': sum(r[cost] for r in selected) / len(selected) if selected else None,
                        'reference_voxel_weighted_dice': sum(r[cost] * r[ref] for r in selected) / denominator if denominator else None}
            aggregates[parent] = {'highres_all_evaluable': aggregate(hr, 'highres_dice', 'highres_reference_voxels'),
                                  'saved_1mm_all_evaluable': aggregate(nr, 'saved_1mm_dice', 'official_1mm_reference_voxels'),
                                  'matched_evaluation_set': {'label_count': len(shared),
                                     'highres': aggregate(shared, 'highres_dice', 'highres_reference_voxels'),
                                     'saved_1mm': aggregate(shared, 'saved_1mm_dice', 'official_1mm_reference_voxels')}}
        values['aggregates_by_parent'] = aggregates
        if family == 'thalamus':
            selected = [r for r in all_rows if r['saved_1mm_dice'] is None or r['saved_1mm_dice'] < 0.9]
        else:
            selected = [r for r in all_rows if any(part in r['name'] for part in ('parasubiculum', 'CA3-body', 'GC-ML-DG', 'Medial-nucleus', 'Cortical-nucleus', '-AAA'))]
        for row in selected:
            existing = known.get(row['output_label'], {})
            row['official_soft_mm3'] = existing.get('reference_soft_volume_mm3')
            row['fnit_soft_mm3'] = existing.get('fnit_soft_volume_mm3')
            row['soft_volume_fraction_difference'] = ((row['fnit_soft_mm3'] - row['official_soft_mm3']) / row['official_soft_mm3'] if row['official_soft_mm3'] else None)
        values['label_metrics'] = selected
        values['label_metrics_selection'] = 'Requested small hippo/amyg labels; thalamus native Dice<0.9 or absent. Aggregates include all evaluable labels.'
    result['limitations'] += ['Mean Dice denominators differ when labels vanish at 1mm. Matched-evaluation aggregates are reported separately.', 'Finite-precision saved NIfTI affine headers can alter nearest-neighbor ties during highres-to-native resampling; runtime-geometry reconstruction is a diagnostic.', 'Right AAA has one official highres hard voxel and no native voxels, despite nonzero official soft volume; hard and soft agreement remain separate.']
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
