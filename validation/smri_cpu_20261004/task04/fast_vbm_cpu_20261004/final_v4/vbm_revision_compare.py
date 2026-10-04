"""Compare thirteen FastVBM outputs between two frozen FNIT revisions.

This performs saved-image analysis only. It does not invoke FNIT, FSL, or models.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

FILES = (
    'T1_brain.nii.gz', 'brain_mask.nii.gz', 'T1_brain_pve_0.nii.gz',
    'T1_brain_pve_1.nii.gz', 'T1_brain_pve_2.nii.gz', 'T1_brain_seg.nii.gz',
    'T1_brain_pveseg.nii.gz', 'T1_brain_mixeltype.nii.gz', 'T1_brain_bias.nii.gz',
    'T1_brain_restore.nii.gz', 'T1_GM_to_template_GM.nii.gz',
    'T1_GM_JAC_nl.nii.gz', 'T1_GM_to_template_GM_mod.nii.gz',
)
HEADER_FIELDS = ('dim', 'pixdim', 'datatype', 'xyzt_units', 'qform_code',
                 'sform_code', 'intent_code', 'quatern_b', 'quatern_c',
                 'quatern_d', 'qoffset_x', 'qoffset_y', 'qoffset_z',
                 'srow_x', 'srow_y', 'srow_z')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def compare(candidate_path, reference_path):
    candidate, reference = nib.load(candidate_path), nib.load(reference_path)
    row = {
        'candidate_sha256': digest(candidate_path), 'reference_sha256': digest(reference_path),
        'candidate_shape': list(candidate.shape), 'reference_shape': list(reference.shape),
        'candidate_dtype': str(candidate.get_data_dtype()),
        'reference_dtype': str(reference.get_data_dtype()),
        'shape_equal': candidate.shape == reference.shape,
        'dtype_equal': candidate.get_data_dtype() == reference.get_data_dtype(),
        'affine_equal': bool(np.array_equal(candidate.affine, reference.affine)),
        'affine_maximum_absolute_error': float(np.max(np.abs(candidate.affine - reference.affine))),
        'header_equal': {key: bool(np.array_equal(candidate.header[key], reference.header[key]))
                         for key in HEADER_FIELDS},
        'qform_equal': bool(np.array_equal(candidate.get_qform(), reference.get_qform())),
        'sform_equal': bool(np.array_equal(candidate.get_sform(), reference.get_sform())),
        'extensions_equal': [(e.get_code(), e.content) for e in candidate.header.extensions]
                            == [(e.get_code(), e.content) for e in reference.header.extensions],
    }
    row['geometry_equal'] = all((row['shape_equal'], row['dtype_equal'], row['affine_equal'],
                                row['qform_equal'], row['sform_equal'],
                                all(row['header_equal'].values())))
    if not row['shape_equal']:
        row['comparison_status'] = 'geometry_mismatch'
        return row
    a, b = np.asanyarray(candidate.dataobj), np.asanyarray(reference.dataobj)
    row['finite'] = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not row['finite']:
        row['comparison_status'] = 'nonfinite_output'
        return row
    d = a.astype(np.float64) - b.astype(np.float64)
    row.update({
        'comparison_status': 'complete', 'values_exact_equal': bool(np.array_equal(a, b)),
        'different_numerical_values': int(np.count_nonzero(a != b)),
        'mae': float(np.mean(np.abs(d))), 'rmse': float(np.sqrt(np.mean(d * d))),
        'maximum_absolute_error': float(np.max(np.abs(d))),
    })
    if a.dtype == b.dtype:
        dtype = np.dtype('u' + str(a.dtype.itemsize))
        row['different_bit_patterns'] = int(np.count_nonzero(
            np.ascontiguousarray(a).view(dtype) != np.ascontiguousarray(b).view(dtype)))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {name: compare(args.candidate / name, args.reference / name) for name in FILES}
    report = {
        'scope': 'saved thirteen maps: final integrated v4 versus prior frozen v2; no rerun of v2',
        'output_count': len(rows), 'outputs': rows,
        'all_numerical_values_equal': all(r.get('values_exact_equal', False) for r in rows.values()),
        'all_data_bit_patterns_equal': all(r.get('different_bit_patterns', -1) == 0 for r in rows.values()),
        'all_geometry_equal': all(r.get('geometry_equal', False) for r in rows.values()),
        'all_extensions_equal': all(r.get('extensions_equal', False) for r in rows.values()),
    }
    ca = json.loads((args.candidate / 'fast_vbm_report.json').read_text())
    rb = json.loads((args.reference / 'fast_vbm_report.json').read_text())
    pull_a = np.asarray(ca['registration']['pull_world_affine'], dtype=np.float64)
    pull_b = np.asarray(rb['registration']['pull_world_affine'], dtype=np.float64)
    report['affine_pull_world'] = {
        'exact_equal': bool(np.array_equal(pull_a, pull_b)),
        'maximum_element_absolute_error': float(np.max(np.abs(pull_a - pull_b))),
    }
    report['registration_qc_field_equal'] = {
        key: ca['registration'].get(key) == rb['registration'].get(key)
        for key in ['pre_nonlinear_signature', 'shared_pre_nonlinear_inputs',
                    'residual_fsl_sha256', 'nonlinear_backend', 'jacobian_convention']
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k.startswith('all_') or k == 'output_count'}))


if __name__ == '__main__':
    main()
