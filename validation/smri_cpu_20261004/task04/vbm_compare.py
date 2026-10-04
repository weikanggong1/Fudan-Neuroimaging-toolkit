"""Score all thirteen saved FastVBM maps without rerunning the pipeline."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


NATIVE_FILES = (
    'T1_brain.nii.gz', 'brain_mask.nii.gz',
    'T1_brain_pve_0.nii.gz', 'T1_brain_pve_1.nii.gz',
    'T1_brain_pve_2.nii.gz', 'T1_brain_seg.nii.gz',
    'T1_brain_pveseg.nii.gz', 'T1_brain_mixeltype.nii.gz',
    'T1_brain_bias.nii.gz', 'T1_brain_restore.nii.gz',
)
STANDARD_FILES = ('T1_GM_to_template_GM.nii.gz', 'T1_GM_JAC_nl.nii.gz',
                  'T1_GM_to_template_GM_mod.nii.gz')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def affine_comparison(candidate_root, reference_root, template):
    """Compare report pull with independent original FLIRT over source corners."""
    def scaled_mm(image):
        zooms = image.header.get_zooms()[:3]
        matrix = np.diag((*zooms, 1.)).astype(np.float64)
        if np.linalg.det(image.affine[:3, :3]) > 0:
            matrix[0, 0] *= -1
            matrix[0, 3] = (image.shape[0] - 1) * zooms[0]
        return matrix
    candidate_report = json.loads((candidate_root / 'fast_vbm_report.json').read_text())
    candidate_forward = np.linalg.inv(candidate_report['registration']['pull_world_affine'])
    moving = nib.load(reference_root / 'T1_brain_pve_1.nii.gz')
    candidate_moving = nib.load(candidate_root / 'T1_brain_pve_1.nii.gz')
    original_flirt = np.loadtxt(reference_root / 'gm_affine.mat')
    original_forward = (template.affine @ np.linalg.inv(scaled_mm(template)) @
                        original_flirt @ scaled_mm(moving) @ np.linalg.inv(moving.affine))
    corners = np.array(np.meshgrid(*[(0, size-1) for size in moving.shape],
                                  indexing='ij')).reshape(3, -1)
    corners = np.vstack((corners, np.ones((1, corners.shape[1]))))
    difference = (candidate_forward @ candidate_moving.affine @ corners -
                  original_forward @ moving.affine @ corners)[:3]
    return {'candidate_forward_world': candidate_forward.tolist(),
            'original_forward_world': original_forward.tolist(),
            'whole_source_grid_maximum_world_error_mm': float(np.max(np.linalg.norm(difference, axis=0))),
            'source_grid_shape': moving.shape,
            'scope': 'affine norm maximum on complete convex source grid is attained at one of its eight corners'}


def metrics(candidate, reference, selected):
    a = candidate[selected].astype(np.float64)
    b = reference[selected].astype(np.float64)
    difference = a - b
    scale = float(np.percentile(b, 99) - np.percentile(b, 1))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    return {'values': int(a.size), 'mae': float(np.abs(difference).mean()),
            'rmse': rmse, 'p99_absolute_error': float(np.percentile(np.abs(difference), 99)),
            'maximum_absolute_error': float(np.max(np.abs(difference))),
            'reference_p99_minus_p1': scale,
            'nrmse_reference_p99_minus_p1': rmse / scale if scale else None,
            'pearson_r': float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None,
            'different_numerical_values': int(np.count_nonzero(difference))}


def label_metrics(a, b, selected, label, voxel_volume):
    aa = (a == label) & selected
    bb = (b == label) & selected
    n1, n2 = int(aa.sum()), int(bb.sum())
    return {'dice': float(2 * np.count_nonzero(aa & bb) / (n1 + n2)) if n1 + n2 else 1.,
            'candidate_voxels': n1, 'reference_voxels': n2,
            'candidate_volume_mm3': n1 * voxel_volume,
            'reference_volume_mm3': n2 * voxel_volume,
            'volume_difference_percent': 100 * (n1 / n2 - 1) if n2 else None}


def compare(candidate_path, reference_path, selected):
    candidate = nib.load(candidate_path)
    reference = nib.load(reference_path)
    row = {'candidate_sha256': digest(candidate_path),
           'reference_sha256': digest(reference_path),
           'candidate_shape': candidate.shape, 'reference_shape': reference.shape,
           'candidate_dtype': str(candidate.get_data_dtype()),
           'reference_dtype': str(reference.get_data_dtype()),
           'shape_equal': candidate.shape == reference.shape,
           'affine_equal': bool(np.array_equal(candidate.affine, reference.affine)),
           'affine_maximum_absolute_error': float(np.max(np.abs(candidate.affine - reference.affine))),
           'header_equal': {key: bool(np.array_equal(candidate.header[key], reference.header[key]))
                            for key in ('dim', 'pixdim', 'datatype', 'xyzt_units',
                                        'qform_code', 'sform_code', 'intent_code')},
           'qform_maximum_absolute_error': float(np.max(np.abs(candidate.get_qform() - reference.get_qform()))),
           'sform_maximum_absolute_error': float(np.max(np.abs(candidate.get_sform() - reference.get_sform())))}
    if not row['shape_equal'] or candidate.shape != selected.shape or not np.allclose(
            candidate.affine, reference.affine, rtol=0, atol=1e-3):
        row['comparison_status'] = 'geometry_mismatch'
        return row
    a, b = np.asanyarray(candidate.dataobj), np.asanyarray(reference.dataobj)
    row['finite'] = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not row['finite']:
        row['comparison_status'] = 'nonfinite_output'
        return row
    row['comparison_status'] = 'complete'
    row['whole_grid'] = metrics(a, b, np.ones(a.shape, dtype=bool))
    row['official_brain_region'] = metrics(a, b, selected)
    row['values_exact_equal'] = bool(np.array_equal(a, b))
    row['dtype_equal'] = a.dtype == b.dtype
    if row['dtype_equal']:
        row['different_bit_patterns'] = int(np.count_nonzero(
            np.ascontiguousarray(a).view('u' + str(a.dtype.itemsize)) !=
            np.ascontiguousarray(b).view('u' + str(b.dtype.itemsize))))
    voxel_volume = abs(float(np.linalg.det(reference.affine[:3, :3])))
    name = reference_path.name
    if name == 'brain_mask.nii.gz':
        row['whole_grid_binary_mask'] = label_metrics(
            a > 0, b > 0, np.ones(a.shape, dtype=bool), True, voxel_volume)
    elif any(suffix in name for suffix in ('_seg.', '_pveseg.', '_mixeltype.')):
        labels = range(6) if '_mixeltype.' in name else range(1, 4)
        row['labels'] = {str(label): label_metrics(a, b, selected, label, voxel_volume)
                         for label in labels}
    if '_pve_' in name or name in (STANDARD_FILES[0], STANDARD_FILES[2]):
        row['soft_volume_mm3'] = {}
        for region, inclusion in (('whole_grid', np.ones(a.shape, dtype=bool)),
                                   ('official_brain_region', selected)):
            candidate_volume = float(a[inclusion].sum(dtype=np.float64) * voxel_volume)
            reference_volume = float(b[inclusion].sum(dtype=np.float64) * voxel_volume)
            row['soft_volume_mm3'][region] = {
                'candidate': candidate_volume, 'reference': reference_volume,
                'difference_percent': 100 * (candidate_volume / reference_volume - 1)
                if reference_volume else None}
        row['threshold_0.5'] = label_metrics(a > .5, b > .5, selected, True, voxel_volume)
    if name == STANDARD_FILES[1]:
        row['candidate_nonpositive_jacobian_voxels'] = int(np.count_nonzero(a <= 0))
        row['reference_nonpositive_jacobian_voxels'] = int(np.count_nonzero(b <= 0))
        row['candidate_range'] = [float(a.min()), float(a.max())]
        row['reference_range'] = [float(b.min()), float(b.max())]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--reference-native', type=Path,
                        help='official upstream directory if only nonlinear branch was rerun')
    parser.add_argument('--template', required=True)
    parser.add_argument('--reference-mask', required=True)
    parser.add_argument('--scope', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    native = args.reference_native or args.reference
    native_mask = np.asanyarray(nib.load(native / 'brain_mask.nii.gz').dataobj) > 0
    native_brain = np.asanyarray(nib.load(native / 'T1_brain.nii.gz').dataobj)
    tissue_mask = native_mask & (native_brain > 0)
    template = nib.load(args.template)
    mask_image = nib.load(args.reference_mask)
    if mask_image.shape != template.shape or not np.allclose(mask_image.affine, template.affine, atol=1e-5, rtol=0):
        raise ValueError('reference mask must match the complete template grid')
    template_mask = (np.asanyarray(mask_image.dataobj) > 0) & (np.asanyarray(template.dataobj) > 0)
    rows = {}
    for name in (*NATIVE_FILES, *STANDARD_FILES):
        reference_dir = native if name in NATIVE_FILES else args.reference
        included = template_mask if name in STANDARD_FILES else (
            native_mask if name in ('T1_brain.nii.gz', 'brain_mask.nii.gz') else tissue_mask)
        rows[name] = compare(args.candidate / name, reference_dir / name, included)
    report = {'scope': args.scope, 'output_count': len(rows),
              'regions': {'official_native_brain_voxels': int(native_mask.sum()),
                          'official_positive_tissue_voxels': int(tissue_mask.sum()),
                          'explicit_template_mask_and_positive_template_voxels': int(template_mask.sum())},
              'outputs': rows,
              'linear_registration': affine_comparison(args.candidate, native, template),
              'all_numerical_values_equal': all(row.get('values_exact_equal', False) for row in rows.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output_count': len(rows), 'all_numerical_values_equal': report['all_numerical_values_equal']}))


if __name__ == '__main__':
    main()
