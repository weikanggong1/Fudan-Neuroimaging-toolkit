"""只从本轮已完成的 FNIT 和原软件文件映射建立配对，不查找旧结果。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-json', type=Path, required=True)
    parser.add_argument('--candidate-files', type=Path, required=True)
    parser.add_argument('--candidate-report', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--preproc-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--fnit-source', type=Path, required=True)
    args = parser.parse_args()
    case = read(args.case_json)
    candidate = read(args.candidate_files)
    candidate_report = read(args.candidate_report)
    native = args.native_root
    preproc = args.preproc_root
    if read(native / 'status.private.json') != {'stage': 'complete', 'status': 'passed'}:
        raise RuntimeError('The fresh original clean chain must complete first')
    if read(preproc / 'status.private.json') != {'stage': 'complete', 'status': 'passed'}:
        raise RuntimeError('The new original preproc stage must complete first')
    if not candidate_report['source_unchanged_during_run']:
        raise RuntimeError('Candidate source changed during its actual API execution')
    args.output_root.mkdir(parents=True, exist_ok=True)
    inputs = {key: case[key] for key in ('bold', 'sbref', 't1w', 'mni_template', 'mni_mask', 'synthstrip_weights')}
    hashes = {key: sha256(path) for key, path in inputs.items()}
    candidate_hashes = candidate_report['input_sha256']
    for key, digest in hashes.items():
        actual_key = 'mni_brain_mask' if key == 'mni_mask' else key
        if candidate_hashes[actual_key] != digest:
            raise ValueError('The completed candidate used a different raw input or resource')
    native_report = read(native / 'pipeline.public.json')
    if native_report['input_sha256'] != hashes:
        raise ValueError('The new native chain raw inputs or resources do not match')
    preproc_report = read(preproc / 'preproc.public.json')
    if any(preproc_report['input_sha256'][key] != hashes[key]
           for key in ('bold', 'sbref', 't1w', 'mni_template', 'mni_mask')):
        raise ValueError('The original preproc is not tied to this same new raw input')
    # Final output files must be the completed API's exact saved products.
    for key in ('clean_native', 'clean_mni', 'preproc_t1w', 'preproc_mni'):
        if sha256(candidate[key]) != candidate_report['checks']['volume'][key]['sha256']:
            raise ValueError('Candidate final output differs from its completed API record')
    registered = native / 'reg_fnirt'
    epi_mask, t1_mask = native / 'masks/epi_mask.nii.gz', native / 'anat/T1_mask.nii.gz'
    common_epi = {'candidate_mask': candidate['epi_mask'], 'reference_mask': str(epi_mask)}
    common_t1 = {'candidate_mask': candidate['t1_mask'], 'reference_mask': str(t1_mask)}
    images = {}

    def add(name, key, original, kind, masks=None):
        first, second = Path(candidate[key]), Path(original)
        if not first.is_file() or not second.is_file():
            raise FileNotFoundError('A required new-stage comparison output is absent')
        images[name] = {'kind': kind, 'candidate': str(first), 'reference': str(second), **(masks or {})}

    add('epi_synthstrip_brain', 'epi_brain', native / 'masks/epi_brain.nii.gz', 'scalar', common_epi)
    add('t1_synthstrip_brain', 't1_brain', native / 'anat/T1_brain.nii.gz', 'scalar', common_t1)
    add('epi_synthstrip_mask', 'epi_mask', epi_mask, 'mask')
    add('t1_synthstrip_mask', 't1_mask', t1_mask, 'mask')
    for key, index in (('fast_csf', 0), ('fast_gm', 1), ('fast_wm', 2)):
        add(key, key, native / f'anat/T1_fast_pve_{index}.nii.gz', 'scalar', common_t1)
    add('fast_bias', 'fast_bias', native / 'anat/T1_fast_bias.nii.gz', 'scalar', common_t1)
    add('fast_restored', 'fast_restored', native / 'anat/T1_fast_restore.nii.gz', 'scalar', common_t1)
    add('native_wm_mask', 'native_wm_mask', registered / 'wm_epi.nii.gz', 'mask')
    add('native_csf_mask', 'native_csf_mask', registered / 'csf_epi.nii.gz', 'mask')
    add('mni_brain_mask', 'mask_mni', registered / 'brain_MNI152_2mm.nii.gz', 'mask')
    add('motion_corrected', 'motion_corrected', native / 'feat/mc/prefiltered_func_data_mcf.nii.gz', 'bold', common_epi)
    add('feat_filtered', 'feat_filtered', native / 'feat/filtered_func_data.nii.gz', 'bold', common_epi)
    add('aroma_native', 'aroma_native', native / 'aroma_fnirt/filtered_func_data_aroma.nii.gz', 'bold', common_epi)
    add('clean_native', 'clean_native', native / 'clean_native_fnirt.nii.gz', 'bold', common_epi)
    add('clean_mni', 'clean_mni', native / 'clean_mni_fnirt.nii.gz', 'bold', {
        'candidate_mask': candidate['mask_mni'], 'reference_mask': str(registered / 'brain_MNI152_2mm.nii.gz')})
    add('t1_affine_registered', 't1_affine_moved', registered / 'T1_affine_in_MNI.nii.gz', 'scalar', {'mask': case['mni_mask']})
    add('t1_registered', 't1_nonlinear_moved', registered / 'T1_in_MNI.nii.gz', 'scalar', {'mask': case['mni_mask']})
    add('epi_registered', 'bbr_final_moved', registered / 'example_func2highres.nii.gz', 'scalar', common_t1)
    original_preproc = read(preproc / 'outputs.private.json')
    for key in ('preproc_mni', 'preproc_t1w'):
        if sha256(original_preproc[key]) != original_preproc['output_sha256'][key]:
            raise ValueError('Original preproc output differs from its newly completed stage record')
    add('preproc_mni', 'preproc_mni', original_preproc['preproc_mni'], 'bold', {'mask': case['mni_mask']})
    target_pair = [nib.load(path) for path in (candidate['preproc_t1w'], original_preproc['preproc_t1w'])]
    target_equal = (target_pair[0].shape == target_pair[1].shape
                    and np.array_equal(target_pair[0].affine, target_pair[1].affine))
    if not target_equal:
        raise ValueError('Independently generated T1 target grids differ; no implicit resampling allowed')
    fixed_t1 = nib.load(original_preproc['t1w_reference'])
    mask_path = args.output_root / 'original_fixed_T1w_brain_mask.private.nii.gz'
    header = fixed_t1.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image((np.asarray(fixed_t1.dataobj) > 0).astype(np.uint8), fixed_t1.affine, header), mask_path)
    add('preproc_t1w', 'preproc_t1w', original_preproc['preproc_t1w'], 'bold', {'mask': str(mask_path)})
    transforms = {}
    for name, key, original in (
        ('bbr_initial', 'bbr_initial', registered / 'example_func2highres_init.mat'),
        ('bbr_final', 'bbr_matrix', registered / 'example_func2highres.mat')):
        transforms[name] = {
            'candidate_matrix': candidate[key], 'reference_matrix': str(original),
            'moving': candidate['feat_reference'], 'reference': str(native / 'anat/T1_brain.nii.gz'),
            'candidate_moving': candidate['feat_reference'], 'reference_moving': str(native / 'feat/example_func.nii.gz'),
            'candidate_fixed': candidate['t1_brain'], 'reference_fixed': str(native / 'anat/T1_brain.nii.gz'),
            'mask': str(t1_mask), 'convention': 'fsl'}
    transforms['t1_affine'] = {
        'candidate_matrix': candidate['t1_affine'], 'reference_matrix': str(registered / 'T1_to_MNI152_2mm_affine.mat'),
        'moving': candidate['t1_brain'], 'reference': case['mni_template'],
        'candidate_moving': candidate['t1_brain'], 'reference_moving': str(native / 'anat/T1_brain.nii.gz'),
        'mask': case['mni_mask'], 'convention': 'fsl'}
    sys.path.insert(0, str(args.fnit_source))
    from fnit.flirt.coordinates import flirt_to_world_affine
    epi_image, t1_image = [nib.load(candidate[key]) for key in ('feat_reference', 't1_brain')]
    bbr_world = flirt_to_world_affine(np.loadtxt(candidate['bbr_matrix']), epi_image.affine, t1_image.affine,
        epi_image.shape, t1_image.shape, epi_image.header.get_zooms()[:3], t1_image.header.get_zooms()[:3])
    to_epi = args.output_root / 'candidate_reference_to_epi_world.private.txt'
    np.savetxt(to_epi, np.linalg.inv(bbr_world), fmt='%.17g')
    manifest = {
        'candidate_revision': candidate_report['source_revision'],
        'reference_revision': native_report['source_revision'],
        'validated_on': '2026-10-02',
        'input_files': inputs, 'reference_input_files': inputs, 'expected_input_sha256': hashes,
        'images': images, 'transforms': transforms,
        'motion_reference_headers': {'candidate': candidate['feat_reference'],
                                     'reference': str(native / 'feat/example_func.nii.gz')},
        'motion': {'moving': case['bold'], 'reference': str(native / 'feat/example_func.nii.gz'),
                   'candidate_reference': candidate['feat_reference'],
                   'reference_reference': str(native / 'feat/example_func.nii.gz'),
                   'mask': str(epi_mask), 'candidate_matrices': candidate['motion_matrices'],
                   'reference_matrices': str(native / 'feat/mc/prefiltered_func_data_mcf.mat')},
        'pull': {'candidate': candidate['mni_pull'],
                 'reference': str(registered / 'MNI152_2mm_to_T1_pull_ras.nii.gz'),
                 'mni_template': case['mni_template'], 'mask': case['mni_mask'],
                 'convention': 'ras_mm_pull_displacement',
                 'candidate_to_epi_world': str(to_epi),
                 'reference_to_epi_world': str(registered / 'reference_to_source_world.txt')},
    }
    path = args.output_root / 'matched_manifest.private.json'
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    guard = {'schema_version': 1,
        'candidate_completed_api_sha256': sha256(args.candidate_report),
        'candidate_source_revision': candidate_report['source_revision'],
        'original_reference_harness_base_revision': native_report['source_revision'],
        'native_completed_clean_report_sha256': sha256(native / 'pipeline.public.json'),
        'native_completed_preproc_report_sha256': sha256(preproc / 'preproc.public.json'),
        'candidate_source_unchanged': True, 'all_raw_inputs_and_resources_identical': True,
        'final_candidate_outputs_match_completed_api': True,
        'final_original_preproc_outputs_match_completed_stage': True,
        'independent_t1w_target_grid_exact_equal': True,
        'image_stages': sorted(images),
        'reference_revision_scope': 'FNIT benchmark harness base Git revision; original software has its separate actual package/executable identity.',
        'preproc_fixed_mask_rule': 'MNI uses the full original supplied template brain mask; T1 uses nonzero brain voxels of the original NiWorkflows fixed target. Candidate/reference BOLD zero series are retained in these domains.',
        'manifest_builder_sha256': sha256(__file__),
        'privacy': 'Anonymous checks and hashes only; manifest and individual transforms stay private.'}
    (args.output_root / 'identity_guard.public.json').write_text(json.dumps(guard, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'manifest_complete': True, 'image_stages': len(images), 'transform_stages': len(transforms)}))


if __name__ == '__main__':
    main()
