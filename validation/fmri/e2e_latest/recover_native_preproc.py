"""后验核验原 preproc 成功子程序；保留旧 driver 失败和未记录时间。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_exec import trace_exit_evidence


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_image(path, reference, frames, tr):
    image = nib.load(path)
    if image.shape != (*reference.shape, frames) or not np.allclose(
            image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError('Original preproc output grid/frame count mismatch')
    values = np.asarray(image.dataobj)
    if image.get_data_dtype() != np.dtype('float32') or not np.isfinite(values).all():
        raise ValueError('Original preproc must be finite float32')
    if not np.isclose(image.header.get_zooms()[3], tr, atol=1e-6) or image.header.get_xyzt_units()[1] != 'sec':
        raise ValueError('Original preproc time axis mismatch')
    return {'shape': list(image.shape), 'dtype': str(image.get_data_dtype()),
            'tr_seconds': float(image.header.get_zooms()[3]), 'time_unit': 'sec',
            'all_finite': True, 'finite_value_count': int(values.size), 'sha256': sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-json', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--preproc-root', type=Path, required=True)
    parser.add_argument('--original-driver-source', type=Path, required=True)
    parser.add_argument('--candidate-revision', required=True)
    args = parser.parse_args()
    output = args.preproc_root
    if (output / 'outputs.private.json').exists():
        raise FileExistsError('Do not overwrite a previously finalized preproc report')
    case = json.loads(args.case_json.read_text())
    root = args.native_root
    registration = root / 'reg_fnirt'
    t1_reference = output / 'T1w_native_bold_reference.nii.gz'
    images = {'preproc_mni': output / 'preproc_MNI.nii.gz',
              'preproc_t1w': output / 'preproc_T1w.nii.gz'}
    commands = json.loads((output / 'commands.private.json').read_text())
    if [row['name'] for row in commands] != ['sampling_reference', 'preproc_MNI', 'preproc_T1w']:
        raise ValueError('Unexpected original command identity')
    timings = {}
    controller = output.parent / 'native_preproc_controller.private.log'
    for line in controller.read_text().splitlines():
        if line.startswith('{'):
            row = json.loads(line)
            if 'completed' in row:
                timings[row.pop('completed')] = row
    if 'preproc_T1w' in timings:
        raise ValueError('Expected the known trace-parser failure before T1 timing was saved')
    for stage in ('preproc_MNI', 'preproc_T1w'):
        proof = trace_exit_evidence(output / (stage + '.exec.private.log'), 255)
        if not proof['original_process_accepted']:
            raise ValueError('Missing successful original child evidence')
        timings.setdefault(stage, {'wall_seconds_including_io': None})['exit_evidence'] = proof
    (output / 'failed_attempt_status.private.json').write_bytes((output / 'status.private.json').read_bytes())
    raw = nib.load(case['bold'])
    if raw.shape[3] != 490:
        raise ValueError('This acceptance run requires all 490 frames')
    validation_started = time.perf_counter()
    checks = {'preproc_mni': check_image(images['preproc_mni'], nib.load(case['mni_template']), raw.shape[3], case['tr']),
              'preproc_t1w': check_image(images['preproc_t1w'], nib.load(t1_reference), raw.shape[3], case['tr'])}
    matrices = sorted((root / 'feat/mc/prefiltered_func_data_mcf.mat').glob('MAT_*'))
    if len(matrices) != raw.shape[3]:
        raise ValueError('Incomplete original motion matrix sequence')
    report = {'schema_version': 1, 'validated_on': '2026-10-02',
        'candidate_revision': args.candidate_revision, 'frames': raw.shape[3], 'tr_seconds': case['tr'],
        'protocol': 'Original FSL4D applywarp: one frame-specific MCFLIRT+BBR premat, optional original FNIRT coefficients, one spline interpolation of raw 4D BOLD per target.',
        'timing_boundary': 'Stage-only sampling from freshly estimated original registration. Original child programs completed; the benchmark driver incorrectly rejected padded strace PID whitespace after the T1 command. T1 wall and combined stage wall were not saved and remain null. No timestamp-derived or summed replacement wall time.',
        'stage_only_wall_seconds': None, 'timings': timings, 'checks': checks,
        'post_validation_recovery': {'original_driver_completed': False,
            'reason': 'strace PID padding was rejected by benchmark evidence parser; original T1 applywarp child exited zero',
            'original_child_commands_completed': True, 'exit_codes_rewritten': False,
            'failed_controller_log_sha256': sha256(controller),
            'original_trace_sha256': {name: sha256(output / (name + '.exec.private.log')) for name in ('preproc_MNI', 'preproc_T1w')},
            'recovery_script_sha256': sha256(__file__),
            'corrected_evidence_parser_sha256': sha256(Path(__file__).resolve().parents[1] / 'native_exec.py'),
            'full_output_validation_seconds': time.perf_counter() - validation_started},
        'original_sampling_reference': json.loads((output / 'sampling_reference.public.json').read_text()),
        'input_sha256': {key: sha256(case[key]) for key in ('bold', 'sbref', 't1w', 'mni_template', 'mni_mask')},
        'reused_fresh_registration_sha256': {
            'bbr': sha256(registration / 'example_func2highres.mat'),
            'fnirt_coefficients': sha256(registration / 'T1_to_MNI_coeff.nii.gz'),
            'motion_sequence': hashlib.sha256('\n'.join(sha256(path) for path in matrices).encode()).hexdigest(),
            'registration_report': sha256(registration / 'registration.public.json')},
        'matrix_series_sha256': {'MNI': sha256(output / 'motion_bbr_4d.private.mat'), 'T1w': sha256(output / 'motion_bbr_t1grid_4d.private.mat')},
        'executables_sha256': {'applywarp_launcher': sha256(Path(case['fsl_root']) / 'bin/applywarp')},
        'original_driver_sha256': {name: sha256(args.original_driver_source / name) for name in ('e2e_latest/run_native_preproc.py', 'e2e_latest/make_original_sampling_reference.py', 'native_exec.py')},
        'limits': ['Stage-only recovered outputs; not a successful continuous raw-to-CIFTI run.',
            'FSL NEWIMAGE spline/extraslice edge rules differ from FNIT preproc grid-constant sampling.',
            'One real subject on a shared host; native FSL uses CPU, configured thread environment eight.'],
        'privacy': 'Anonymous scalar checks and hashes only; raw paths, matrices, traces and 4D data remain private.'}
    (output / 'preproc.public.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'outputs.private.json').write_text(json.dumps({
        **{key: str(path) for key, path in images.items()}, 't1w_reference': str(t1_reference),
        'report': str(output / 'preproc.public.json'), 'frames': raw.shape[3], 'tr_seconds': case['tr'],
        'input_sha256': report['input_sha256'], 'output_sha256': {key: value['sha256'] for key, value in checks.items()},
        'fresh_registration_report': str(registration / 'registration.public.json')}, indent=2) + '\n')
    (output / 'status.private.json').write_text(json.dumps({'stage': 'complete', 'status': 'passed'}) + '\n')
    print(json.dumps({'recovered_completed_original_outputs': True,
                      'checks': checks, 'original_driver_completed': False}))


if __name__ == '__main__':
    main()
