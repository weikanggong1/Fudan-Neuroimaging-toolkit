"""相同raw的两套原软件协议比较：FSL单次采样和完整fMRIPrep，不含FNIT输出。"""
import argparse
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_volume_e2e as volume_helper
from compare_matched_pipeline import correlations, load_mask
from compare_volume_e2e import common_mni_views, sha256


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report-out', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = read(args.manifest)
    stage = read(manifest['fsl_preproc_report'])
    full = read(manifest['fmriprep_report'])
    proof = read(manifest['original_trace_verification'])
    if (full['exit_code'] != 0 or full.get('validation_complete') is not True
            or full.get('raw_inputs_unchanged') is not True or full.get('raw_t1_unchanged') is not True
            or full.get('STC') is not False or full.get('SDC') is not False
            or full.get('input_frames') != 490):
        raise ValueError('The complete original fMRIPrep protocol is not verified')
    if read(manifest['fsl_preproc_status']) != {'stage': 'complete', 'status': 'passed'}:
        raise ValueError('The original FSL sampling stage is not complete')
    if (proof['all_scientific_results_verified_with_independent_controls'] is not True
            or proof['source_records_sha256']['preproc_timed'] != sha256(manifest['fsl_preproc_report'])):
        raise ValueError('The original FSL process proof is not bound to this completed stage')
    sampling_rows = [row for row in proof['commands'] if row['group'] == 'preproc_timed']
    if len(sampling_rows) != 2 or any(row['original_process_accepted'] is not True for row in sampling_rows):
        raise ValueError('The original sampling stage has an unresolved command')
    identities = {name: sha256(path) for name, path in manifest['raw_inputs'].items()}
    if set(identities) != {'bold', 'sbref', 't1w'}:
        raise ValueError('The three raw source files must be explicitly enumerated')
    for name, full_key in (('bold', 'raw_bold_sha256'), ('sbref', 'raw_sbref_sha256'), ('t1w', 'raw_t1_sha256')):
        if identities[name] != stage['input_sha256'][name] or identities[name] != full[full_key]:
            raise ValueError('The original protocols did not use the same raw data')
    first, second, mask_path = (Path(manifest[key]) for key in ('fsl_preproc_mni', 'fmriprep_preproc_mni', 'mni_mask'))
    hashes = {'original_fsl': sha256(first), 'original_fmriprep': sha256(second)}
    fmp_outputs = [row['sha256'] for row in full['outputs'] if row['kind'] == 'MNI152NLin6Asym']
    if hashes['original_fsl'] != stage['checks']['preproc_mni']['sha256'] or hashes['original_fmriprep'] not in fmp_outputs:
        raise ValueError('Original comparison files differ from their actual completed output records')
    if sha256(mask_path) != stage['input_sha256']['mni_mask']:
        raise ValueError('Use the original fixed template mask, independent of either output')
    original_images = [nib.load(path) for path in (first, second)]
    images, orientation = common_mni_views(*original_images)
    if (images[0].shape[-1] != 490 or any(image.header.get_xyzt_units() != ('mm', 'sec') for image in images)
            or any(not np.isclose(image.header.get_zooms()[3], .735, atol=1e-6) for image in images)):
        raise ValueError('The complete original time axes do not match this full490 run')
    arrays = [np.asarray(image.dataobj, dtype=np.float32) for image in images]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError('Nonfinite complete original protocol output')
    mask = load_mask(mask_path, images[0])
    values = [array[mask] for array in arrays]
    metrics = correlations(*values, temporal=True)
    varying = [np.std(value, axis=1, dtype=np.float64) > 1e-6 for value in values]
    report = {'schema_version': 1, 'validated_on': '2026-10-02',
        'protocols': ['Original FSL one-pass raw-to-MNI sampling using fresh original MCFLIRT/BBR/FNIRT',
                      'Original independent complete fMRIPrep25.2.4 published raw-to-MNI preproc'],
        'contains_fnit_processed_output': False,
        'both_complete_outputs_bound_to_successful_execution_records': True,
        'same_raw_input_sha256': identities, 'shape': list(images[0].shape),
        'output_sha256': hashes, 'saved_dtype': [str(image.get_data_dtype()) for image in original_images],
        'fixed_comparison_mask_sha256': sha256(mask_path), 'fixed_mask_voxels': int(mask.sum()),
        'axis_index_alignment': orientation, 'metrics': metrics,
        'temporal_coverage_in_fixed_mask': {
            'original_fsl_varying_voxels': int(varying[0].sum()),
            'original_fmriprep_varying_voxels': int(varying[1].sum()),
            'both_varying_voxels': int((varying[0] & varying[1]).sum()),
            'only_original_fsl_varying_voxels': int((varying[0] & ~varying[1]).sum()),
            'only_original_fmriprep_varying_voxels': int((~varying[0] & varying[1]).sum())},
        'execution_report_sha256': {key: sha256(manifest[key]) for key in
            ('fsl_preproc_report', 'fmriprep_report', 'original_trace_verification')},
        'comparison_script_sha256': sha256(__file__),
        'common_grid_helper_sha256': sha256(volume_helper.__file__),
        'comparison_wall_seconds': time.perf_counter() - started,
        'sampling_stage_scope': 'The original FSL reference is the separately timed completed sampling stage with registration reused from this round; it is not a separate complete raw-to-CIFTI wall clock.',
        'comparison_scope': 'Both original protocols process the same full490 raw input. A lossless axis-index permutation/flip aligns the same physical MNI lattice. Numerical errors retain all voxels of the fixed input template mask, including zero series; temporal correlations omit constant series only. No new interpolation, registration, smoothing, fitted intensity scaling or FNIT output is used. This control measures protocol differences without identifying a causal component.',
        'privacy': 'Anonymous statistics and hashes only; full original images and private paths remain on the server.'}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'original_protocol_metrics': metrics, 'mask_voxels': int(mask.sum())}))


if __name__ == '__main__':
    main()
