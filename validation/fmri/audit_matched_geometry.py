"""Audit saved native/FNIT registration outputs without rerunning algorithms.

--run-root selects one private matched-benchmark directory. It contains native
outputs, candidate_fnirt outputs and case.native.private.json. Alternative
private case and candidate directories can be supplied explicitly. Only
anonymous checks, scalar errors and hashes are written to --report-out.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import nibabel as nib
import numpy as np


def digest(path):
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def scaled(image):
    spacing = image.header.get_zooms()[:3]
    matrix = np.diag([*spacing, 1.])
    if np.linalg.det(image.affine[:3, :3]) > 0:
        matrix[0, 0] *= -1
        matrix[0, 3] = spacing[0] * (image.shape[0] - 1)
    return matrix


def forward(matrix, moving, fixed):
    return fixed.affine @ np.linalg.inv(scaled(fixed)) @ matrix @ scaled(moving) @ np.linalg.inv(moving.affine)


def summaries(vectors):
    values = np.linalg.norm(vectors, axis=-1)
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)),
            'rms': float(np.sqrt(np.mean(values**2))), 'maximum': float(values.max())}


def ras_from_relative(path, source, target):
    voxels = np.indices(target.shape[:3], dtype=np.float64).reshape(3, -1)
    reference_scaled = scaled(target)[:3, :3] @ voxels + scaled(target)[:3, 3:4]
    field = np.asarray(nib.load(path).dataobj, dtype=np.float64)
    source_scaled = reference_scaled + field.reshape(-1, 3).T
    inverse = source.affine @ np.linalg.inv(scaled(source))
    source_world = inverse[:3, :3] @ source_scaled + inverse[:3, 3:4]
    target_world = target.affine[:3, :3] @ voxels + target.affine[:3, 3:4]
    return (source_world - target_world).T.reshape(*target.shape[:3], 3), target_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--report-out', type=Path, required=True)
    parser.add_argument('--case-json', type=Path,
                        help='Private input manifest; default RUN_ROOT/case.native.private.json')
    parser.add_argument('--candidate-root', type=Path,
                        help='Candidate benchmark directory; default RUN_ROOT/candidate_fnirt')
    parser.add_argument('--candidate-anatomical', type=Path,
                        help='Exact candidate anatomical cache directory if more than one exists')
    args = parser.parse_args()
    root = args.run_root
    native = root / 'native'
    reg = native / 'reg_fnirt'
    case = json.loads((args.case_json or root / 'case.native.private.json').read_text())
    source_report = json.loads((reg / 'registration.public.json').read_text())
    candidate_root = args.candidate_root or root / 'candidate_fnirt'
    if args.candidate_anatomical:
        candidate = args.candidate_anatomical
    else:
        candidate_manifests = list((candidate_root / 'derivatives').rglob('manifest.json'))
        if len(candidate_manifests) != 1:
            raise ValueError('Specify --candidate-anatomical when the cache is ambiguous')
        candidate = candidate_manifests[0].parent
    t1 = nib.load(native / 'anat/T1_brain.nii.gz')
    raw_t1 = nib.load(case['t1w'])
    epi = nib.load(native / 'feat/example_func.nii.gz')
    raw_sbref = nib.load(case['sbref'])
    template = nib.load(case['mni_template'])
    template_mask = np.asarray(nib.load(case['mni_mask']).dataobj) > 0
    wm = np.asarray(nib.load(native / 'anat/T1_fast_pve_2.nii.gz').dataobj)
    wmseg = np.asarray(nib.load(reg / 'T1_wmseg.nii.gz').dataobj) > 0
    epi_mask = np.asarray(nib.load(native / 'masks/epi_mask.nii.gz').dataobj) > 0
    brain = nib.load(reg / 'MNI_brain.nii.gz')
    candidate_brain = nib.load(candidate / 'MNI_brain.nii.gz')
    coefficients = nib.load(reg / 'T1_to_MNI_coeff.nii.gz')
    affine = np.loadtxt(reg / 'T1_to_MNI152_2mm_affine.mat')
    checks = {
        'raw_t1_and_brain_shape_equal': raw_t1.shape == t1.shape,
        'raw_t1_and_brain_affine_equal': np.array_equal(raw_t1.affine, t1.affine),
        'raw_t1_and_brain_pixdim_equal': raw_t1.header.get_zooms() == t1.header.get_zooms(),
        'epi_reference_equals_original_sbref_values': np.array_equal(np.asarray(epi.dataobj), np.asarray(raw_sbref.dataobj)),
        'epi_reference_equals_original_sbref_affine': np.array_equal(epi.affine, raw_sbref.affine),
        'mni_brain_equals_full_times_mask': np.array_equal(np.asarray(brain.dataobj), np.asarray(template.dataobj, dtype=np.float32) * template_mask),
        'native_candidate_template_arrays_equal': np.array_equal(np.asarray(brain.dataobj), np.asarray(candidate_brain.dataobj)),
        'native_candidate_template_affines_equal': np.array_equal(brain.affine, candidate_brain.affine),
        'native_candidate_template_header_equal': brain.header.binaryblock == candidate_brain.header.binaryblock,
        'wm_boundary_matches_native_fast_pve_ge_0p5': np.array_equal(wmseg, wm >= .5),
        'coefficient_intent_2007': int(coefficients.header['intent_code']) == 2007,
        'coefficient_sform_contains_t1_affine_float32': np.array_equal(coefficients.get_sform(), affine.astype(np.float32)),
        'coefficient_qoffset_contains_target_shape': np.array_equal(np.asarray([coefficients.header[x] for x in ['qoffset_x','qoffset_y','qoffset_z']]), template.shape),
        'coefficient_intent_params_contain_target_pixdim': np.array_equal(np.asarray([coefficients.header[x] for x in ['intent_p1','intent_p2','intent_p3']]), template.header.get_zooms()[:3]),
        'all_command_outputs_match_published_hashes': all(digest(reg / name) == checksum for name, checksum in source_report['sha256']['outputs'].items()),
        'all_actual_registration_children_exit_zero': all(stage['actual_child_exit_code'] == 0 for stage in source_report['stages'].values()),
    }
    for name, path in {
        'native_t1_fsl': reg / 'T1_to_MNI_fsl_relative.nii.gz',
        'native_epi_fsl': reg / 'EPI_to_MNI_fsl_relative.nii.gz',
        'native_t1_ras': reg / 'MNI152_2mm_to_T1_pull_ras.nii.gz',
        'native_epi_ras': reg / 'MNI_to_EPI_pull_ras.nii.gz',
        'candidate_t1_ras': candidate_root / 'resampling_inputs/mni_to_t1_pull_ras.nii.gz',
    }.items():
        field = nib.load(path)
        checks[name + '_grid_matches_template'] = (field.shape == (*template.shape, 3)
                                                   and np.allclose(field.affine, template.affine, atol=1e-4, rtol=0))
    for tissue in ['wm', 'csf']:
        probability = np.asarray(nib.load(reg / (tissue + '_pve_epi.nii.gz')).dataobj)
        saved = np.asarray(nib.load(reg / (tissue + '_epi.nii.gz')).dataobj) > 0
        checks[tissue + '_epi_mask_matches_pve_ge_0p8_intersection'] = np.array_equal(saved, (probability >= .8) & epi_mask)
    warped_mask = np.asarray(nib.load(reg / 'brain_MNI_raw.nii.gz').dataobj)
    mni_mask = np.asarray(nib.load(reg / 'brain_MNI152_2mm.nii.gz').dataobj) > 0
    checks['mni_mask_matches_warped_epi_and_template_intersection'] = np.array_equal(mni_mask, (warped_mask > .5) & template_mask)
    t1_ras, mni_world = ras_from_relative(reg / 'T1_to_MNI_fsl_relative.nii.gz', t1, template)
    epi_ras, _ = ras_from_relative(reg / 'EPI_to_MNI_fsl_relative.nii.gz', epi, template)
    saved_t1_ras = np.asarray(nib.load(reg / 'MNI152_2mm_to_T1_pull_ras.nii.gz').dataobj, dtype=np.float64)
    saved_epi_ras = np.asarray(nib.load(reg / 'MNI_to_EPI_pull_ras.nii.gz').dataobj, dtype=np.float64)
    inverse_bbr = np.linalg.inv(forward(np.loadtxt(reg / 'example_func2highres.mat'), epi, t1))
    saved_inverse = np.loadtxt(reg / 'reference_to_source_world.txt')
    composed = inverse_bbr[:3, :3] @ (mni_world + t1_ras.reshape(-1, 3).T) + inverse_bbr[:3, 3:4]
    composed -= mni_world
    composition_delta = composed.T.reshape(*template.shape, 3) - epi_ras
    checks['bbr_world_inverse_file_matches_independent_conversion'] = np.allclose(inverse_bbr, saved_inverse, atol=1e-12, rtol=0)
    candidate_t1_ras = np.asarray(nib.load(candidate_root / 'resampling_inputs/mni_to_t1_pull_ras.nii.gz').dataobj, dtype=np.float64)
    candidate_affine = np.loadtxt(candidate / 'T1_to_MNI152_2mm_affine.mat')
    candidate_t1 = nib.load(candidate / 'T1_brain.nii.gz')
    native_inverse_affine = np.linalg.inv(forward(affine, t1, brain))
    candidate_inverse_affine = np.linalg.inv(forward(candidate_affine, candidate_t1, candidate_brain))
    affine_delta = ((native_inverse_affine[:3, :3] - candidate_inverse_affine[:3, :3]) @ mni_world
                    + native_inverse_affine[:3, 3:4] - candidate_inverse_affine[:3, 3:4])
    report = {
        'schema_version': 1,
        'scope': 'Independent saved-output geometry audit on the completed real native FSL registration and FNIT FNIRT run; no registration, resampling or decomposition rerun.',
        'native_source_revision': source_report['source_revision'],
        'software': {'python': platform.python_version(), 'numpy': np.__version__,
                     'nibabel': nib.__version__},
        'inverse_diagnostic_enabled': source_report['inverse_diagnostic_enabled'],
        'inverse_scope': 'Forward EPI-to-MNI composition uses inverse of the BBR affine only. It does not invert a nonlinear warp. Optional native invwarp is a separate diagnostic excluded from primary registration timing.',
        'checks': {key: bool(value) for key, value in checks.items()},
        'saved_t1_ras_vs_independent_fsl_conversion_mm': summaries((saved_t1_ras - t1_ras)[template_mask]),
        'saved_epi_ras_vs_independent_fsl_conversion_mm': summaries((saved_epi_ras - epi_ras)[template_mask]),
        'native_convertwarp_vs_independent_bbr_t1_composition_mm': summaries(composition_delta[template_mask]),
        'native_vs_fnit_t1_affine_pull_mm': summaries(affine_delta.T.reshape(*template.shape, 3)[template_mask]),
        'native_vs_fnit_complete_t1_pull_mm': summaries((t1_ras - candidate_t1_ras)[template_mask]),
        'coefficient_shape': list(coefficients.shape),
        'coefficient_knot_spacing_voxels': list(map(float, coefficients.header.get_zooms()[:3])),
        'sha256': {
            'audit_script': digest(Path(__file__)),
            'native_registration_report': digest(reg / 'registration.public.json'),
            'native_template': digest(reg / 'MNI_brain.nii.gz'),
            'candidate_template': digest(candidate / 'MNI_brain.nii.gz'),
            'candidate_t1_brain': digest(candidate / 'T1_brain.nii.gz'),
            'candidate_t1_affine': digest(candidate / 'T1_to_MNI152_2mm_affine.mat'),
            'candidate_t1_pull': digest(candidate_root / 'resampling_inputs/mni_to_t1_pull_ras.nii.gz'),
        },
        'privacy': 'Only anonymous booleans, scalar metrics and hashes; all arrays and paths remain private.',
    }
    if not all(checks.values()):
        raise ValueError('A saved-output geometric contract failed: ' + ','.join(k for k,v in checks.items() if not v))
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
