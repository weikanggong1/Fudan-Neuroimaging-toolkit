#!/usr/bin/env python3
"""Compare independent official and actual FNIT rawprep outputs on the native grid.

This compares the accumulated rawprep chain: the two masks/fields are independent.
It is not a fixed-field/mask EDDY solver comparison or an equivalence test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b''): value.update(data)
    return value.hexdigest()


def statistics(reference, candidate):
    if reference.shape != candidate.shape:
        return {'compatible_shape': False, 'reference_shape': list(reference.shape), 'candidate_shape': list(candidate.shape)}
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    absolute = np.abs(difference)
    return {'compatible_shape': True, 'shape': list(reference.shape), 'neq': int(np.count_nonzero(difference)),
            'max_abs': float(absolute.max()), 'p99_abs': float(np.percentile(absolute, 99)),
            'rmse': float(np.sqrt(np.mean(difference**2))),
            'reference_min_max': [float(reference.min()), float(reference.max())],
            'candidate_min_max': [float(candidate.min()), float(candidate.max())]}


def image_comparison(reference_path, candidate_path, mask=None):
    reference_image, candidate_image = nib.load(reference_path), nib.load(candidate_path)
    reference = np.asarray(reference_image.dataobj, dtype=np.float32)
    candidate = np.asarray(candidate_image.dataobj, dtype=np.float32)
    result = {'reference_path': str(reference_path), 'candidate_path': str(candidate_path),
              'reference_sha256': sha(reference_path), 'candidate_sha256': sha(candidate_path),
              'affine_max_abs': float(np.max(np.abs(reference_image.affine - candidate_image.affine))),
              'reference_shape': list(reference.shape), 'candidate_shape': list(candidate.shape),
              'full_grid': statistics(reference, candidate)}
    if mask is not None and reference.shape == candidate.shape:
        if mask.shape != reference.shape[:3]: raise ValueError('mask grid mismatch')
        result['official_brain_mask'] = statistics(reference[mask], candidate[mask])
    return result


def load_sidecar(path):
    """Recognize the documented native outlier header; numeric rows remain strict."""
    with path.open() as stream:first=stream.readline().strip()
    skip = 1 if path.name.startswith("data.eddy_outlier_") and first.startswith("One row per scan, one column per slice.") else 0
    data = np.loadtxt(path, ndmin=2, skiprows=skip)
    if not np.isfinite(data).all():raise ValueError("nonfinite numeric sidecar: " + str(path))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-case', type=Path, required=True)
    parser.add_argument('--fnit-preproc', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    official, fnit = args.official_case, args.fnit_preproc
    official_report = json.loads((official / 'report.json').read_text())
    if not official_report['completed']:
        raise ValueError('official rawprep has not completed; do not compare partial output as a finished chain')
    reference_mask = nib.load(official / 'mask/nodif_brain_mask.nii.gz')
    candidate_mask = nib.load(fnit / 'eddy/nodif_brain_mask.nii.gz')
    a, b = np.asarray(reference_mask.dataobj) > 0, np.asarray(candidate_mask.dataobj) > 0
    if a.shape != b.shape or not np.allclose(reference_mask.affine, candidate_mask.affine, rtol=0, atol=1e-5):
        raise ValueError('mask grids differ')
    mask_metrics = {'reference_voxels': int(a.sum()), 'candidate_voxels': int(b.sum()),
                    'neq': int(np.count_nonzero(a != b)), 'dice': float(2 * np.count_nonzero(a & b) / (a.sum() + b.sum())),
                    'reference_sha256': sha(official / 'mask/nodif_brain_mask.nii.gz'),
                    'candidate_sha256': sha(fnit / 'eddy/nodif_brain_mask.nii.gz')}
    eddy_command = next(x for x in official_report['commands'] if x['stage'] in ('official_EDDY_GPU', 'official_EDDY_CPU'))
    report = {'scope': 'accumulated rawprep: independent official TOPUP+official SynthStrip+official EDDY vs actual FNIT outputs',
              'equivalence_assessed': False, 'same_EDDY_field_and_mask': False, 'official_report_sha256': sha(official / 'report.json'),
              'official_host': official_report['host'], 'official_GPU_UUID': official_report['GPU_UUID'],
              'official_EDDY_solver': official_report.get('EDDY_solver', 'gpu'), 'official_CPU_threads': official_report['threads'],
              'official_EDDY_command': {key: eddy_command.get(key) for key in ('stage', 'command', 'env', 'program_sha256', 'returncode', 'wall_seconds', 'lock_wait_seconds', 'sampled_GPU_process_peak_bytes', 'sampled_RSS_peak_bytes')}, 
              'official_GP_seed': official_report['gp_seed'], 'official_ref_scan_no': official_report['selection']['ap_index'],
              'brain_mask': mask_metrics, 'TOPUP': {}, 'EDDY': {}}
    qc_path = fnit / 'eddy/data.eddy_qc.json'
    if qc_path.exists():
        qc = json.loads(qc_path.read_text())
        report['FNIT_QC_SHA256'] = sha(qc_path)
        report['FNIT_GP_seed'] = qc.get('gp_seed_override')
        report['GP_seed_matches'] = qc.get('gp_seed_override') == official_report['gp_seed']
        report['FNIT_QC_device'] = qc.get('device')
        report['FNIT_QC_TF32'] = qc.get('tf32')
        report['FNIT_actual_GPU_UUID'] = 'must be read from outer actual run report; logical device is insufficient'
    else:
        report['GP_seed_matches'] = 'not captured; no same-seed assertion'
    for name in ['B0_AP_PA.nii.gz', 'fieldmap_fout.nii.gz', 'fieldmap_iout.nii.gz', 'fieldmap_out_fieldcoef.nii.gz']:
        report['TOPUP'][name] = image_comparison(official / 'topup' / name, fnit / 'topup' / name,
                                               a if name in ('fieldmap_fout.nii.gz', 'fieldmap_iout.nii.gz') else None)
    packing = report['TOPUP']['B0_AP_PA.nii.gz']
    report['selected_pair_same_voxels_and_affine'] = packing['full_grid']['compatible_shape'] and packing['full_grid']['neq'] == 0 and packing['affine_max_abs'] == 0
    report['EDDY']['data.nii.gz'] = image_comparison(official / 'eddy/data.nii.gz', fnit / 'eddy/data.nii.gz', a)
    paths = ['topup/fieldmap_out_movpar.txt', 'eddy/data.eddy_parameters', 'eddy/data.eddy_movement_rms',
             'eddy/data.eddy_restricted_movement_rms', 'eddy/data.eddy_outlier_map',
             'eddy/data.eddy_outlier_n_stdev_map', 'eddy/data.eddy_outlier_n_sqr_stdev_map']
    report['numeric_sidecars'] = {}
    for relative in paths:
        reference_path, candidate_path = official / relative, fnit / relative
        reference, candidate = load_sidecar(reference_path), load_sidecar(candidate_path)
        metrics = statistics(reference, candidate)
        metrics.update(reference_sha256=sha(reference_path), candidate_sha256=sha(candidate_path))
        if reference.shape == candidate.shape:
            metrics['per_column'] = [statistics(reference[:, index], candidate[:, index]) for index in range(reference.shape[1])]
        report['numeric_sidecars'][relative] = metrics
    bvals = np.loadtxt(official / 'raw/AP.bval').reshape(-1)
    reference = np.loadtxt(official / 'eddy/data.eddy_rotated_bvecs')
    candidate = np.loadtxt(fnit / 'eddy/data.eddy_rotated_bvecs')
    if reference.shape != candidate.shape or reference.shape != (3, len(bvals)): raise ValueError('gradient shape mismatch')
    selected = bvals >= 100
    left, right = reference[:, selected], candidate[:, selected]
    dot = np.sum(left * right, axis=0) / (np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0))
    angles = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    report['gradient_angle_degrees'] = {'frames': int(selected.sum()), 'max': float(angles.max()),
                                       'p99': float(np.percentile(angles, 99)), 'rms': float(np.sqrt(np.mean(angles**2))),
                                       'mean': float(angles.mean()), 'b0_excluded': True}
    report['official_rawprep_time'] = {key: official_report[key] for key in ('observed_rawprep_total_wall_seconds', 'rawprep_wall_excluding_GPU_queue_seconds', 'GPU_lock_wait_seconds', 'CPU_reference_activation_wall_seconds') if key in official_report}
    if 'source_CPU_stage_lineage' in official_report:
        report['source_CPU_stage_lineage'] = official_report['source_CPU_stage_lineage']
        report['timing_policy'] = official_report['timing_policy']
    report['timing_comparison'] = 'FNIT stage time/source/GPU must be supplied from its actual run report; no speedup claim inferred here'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
