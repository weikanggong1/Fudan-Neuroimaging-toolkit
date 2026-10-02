"""原 FSL 的一次插值 preproc；复用本轮新估计配准，仅记录本阶段时间。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_exec import run_traced


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def scaled_mm(image):
    sizes = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    matrix = np.diag([*sizes, 1.0])
    if np.linalg.det(image.affine[:3, :3]) > 0:
        matrix[0, 0] *= -1
        matrix[0, 3] = sizes[0] * (image.shape[0] - 1)
    return matrix


def check_image(path, reference, frames):
    image = nib.load(path)
    if image.shape != (*reference.shape, frames) or not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError('Original preproc output has wrong reference grid or frame count')
    values = np.asarray(image.dataobj)
    if image.get_data_dtype() != np.dtype('float32') or not np.isfinite(values).all():
        raise ValueError('Original preproc must be finite float32')
    return {'shape': list(image.shape), 'dtype': str(image.get_data_dtype()),
            'tr_seconds': float(image.header.get_zooms()[3]),
            'time_unit': image.header.get_xyzt_units()[1],
            'all_finite': True, 'sha256': sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-json', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--reference-launch-plan', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--candidate-revision', required=True)
    parser.add_argument('--wait-input-seconds', type=float, default=600)
    args = parser.parse_args()
    case = json.loads(args.case_json.read_text())
    root = args.native_root
    output = args.output_root
    if output.exists():
        raise FileExistsError('Use a fresh native preproc directory')
    output.mkdir(parents=True)
    registration = root / 'reg_fnirt'
    needed = [root / 'feat/mc/prefiltered_func_data_mcf.mat/MAT_0489',
              registration / 'example_func2highres.mat',
              registration / 'T1_to_MNI_coeff.nii.gz',
              registration / 'registration.public.json']
    deadline = time.monotonic() + args.wait_input_seconds
    while not all(path.is_file() for path in needed):
        if time.monotonic() > deadline:
            raise FileNotFoundError('The fresh native registration is not ready')
        time.sleep(2)
    fsl = Path(case['fsl_root'])
    environment = dict(os.environ, FSLDIR=str(fsl), FSLOUTPUTTYPE='NIFTI_GZ',
                       OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    environment['LD_LIBRARY_PATH'] = str(fsl / 'lib') + ':' + environment.get('LD_LIBRARY_PATH', '')
    commands = []
    timings = {}

    def run(name, command, *, original=True):
        (output / 'status.private.json').write_text(json.dumps({'stage': name, 'status': 'running'}) + '\n')
        commands.append({'name': name, 'argv': list(map(str, command))})
        (output / 'commands.private.json').write_text(json.dumps(commands, indent=2) + '\n')
        started = time.perf_counter()
        with (output / (name + '.private.log')).open('wb') as log:
            if original:
                result, evidence = run_traced(command, trace_path=output / (name + '.exec.private.log'),
                                               env=environment, stdout=log, stderr=subprocess.STDOUT)
                if not evidence['original_process_accepted']:
                    raise RuntimeError('An original sampling command has no successful child evidence')
            else:
                result = subprocess.run(list(map(str, command)), env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
                evidence = {'launcher_exit_code': result.returncode}
                if result.returncode:
                    raise RuntimeError('Original NiWorkflows sampling grid generation failed')
        timings[name] = {'wall_seconds_including_io': time.perf_counter() - started,
                         'exit_evidence': evidence}
        print(json.dumps({'completed': name, **timings[name]}), flush=True)

    plan = json.loads(args.reference_launch_plan.read_text())['argv']
    container = Path(plan[plan.index('--container-image') + 1])
    singularity = Path(plan[plan.index('--singularity') + 1])
    t1_reference = output / 'T1w_native_bold_reference.nii.gz'
    source = Path(__file__).resolve().parent
    started = time.perf_counter()
    run('sampling_reference', [singularity, 'exec', '--cleanenv', '--bind', '/cwStorage,/home1,/public',
        container, 'python', source / 'make_original_sampling_reference.py',
        '--fixed', root / 'anat/T1_brain.nii.gz', '--moving', case['sbref'],
        '--mask', root / 'anat/T1_mask.nii.gz', '--work', output / 'sampling_reference_work',
        '--output', t1_reference, '--report', output / 'sampling_reference.public.json'], original=False)
    raw = nib.load(case['bold'])
    target = nib.load(case['mni_template'])
    original_t1 = nib.load(root / 'anat/T1_brain.nii.gz')
    t1_grid = nib.load(t1_reference)
    matrices = sorted((root / 'feat/mc/prefiltered_func_data_mcf.mat').glob('MAT_*'))
    if len(matrices) != raw.shape[3] or raw.shape[3] != 490:
        raise ValueError('This acceptance run requires all 490 native motion matrices')
    motion = np.stack([np.loadtxt(path) for path in matrices])
    bbr = np.loadtxt(registration / 'example_func2highres.mat')
    # Old T1-scaled-mm -> world RAS -> native-resolution T1 reference-scaled-mm.
    bridge = scaled_mm(t1_grid) @ np.linalg.inv(t1_grid.affine) @ original_t1.affine @ np.linalg.inv(scaled_mm(original_t1))
    mni_series = bbr @ motion
    t1_series = bridge @ mni_series
    if not np.isfinite(t1_series).all() or not np.isfinite(mni_series).all():
        raise ValueError('Invalid composite matrix sequence')
    series = {'MNI': output / 'motion_bbr_4d.private.mat',
              'T1w': output / 'motion_bbr_t1grid_4d.private.mat'}
    np.savetxt(series['MNI'], mni_series.reshape(-1, 4), fmt='%.17g')
    np.savetxt(series['T1w'], t1_series.reshape(-1, 4), fmt='%.17g')
    outputs = {'preproc_mni': output / 'preproc_MNI.nii.gz',
               'preproc_t1w': output / 'preproc_T1w.nii.gz'}
    # Original applywarp accepts a 4T x 4 ASCII premat and selects one 4x4 block
    # for each raw frame. The input is never an already corrected BOLD.
    run('preproc_MNI', [fsl / 'bin/applywarp', '--in=' + case['bold'],
        '--ref=' + case['mni_template'], '--warp=' + str(registration / 'T1_to_MNI_coeff.nii.gz'),
        '--premat=' + str(series['MNI']), '--interp=spline', '--datatype=float',
        '--out=' + str(outputs['preproc_mni'])])
    run('preproc_T1w', [fsl / 'bin/applywarp', '--in=' + case['bold'],
        '--ref=' + str(t1_reference), '--premat=' + str(series['T1w']), '--interp=spline',
        '--datatype=float', '--out=' + str(outputs['preproc_t1w'])])
    wall = time.perf_counter() - started
    checks = {'preproc_mni': check_image(outputs['preproc_mni'], target, raw.shape[3]),
              'preproc_t1w': check_image(outputs['preproc_t1w'], t1_grid, raw.shape[3])}
    if any(not np.isclose(check['tr_seconds'], case['tr'], atol=1e-6) or
           check['time_unit'] != 'sec' for check in checks.values()):
        raise ValueError('Original preproc changed the raw time axis')
    report = {'schema_version': 1, 'validated_on': '2026-10-02',
        'candidate_revision': args.candidate_revision, 'frames': raw.shape[3], 'tr_seconds': case['tr'],
        'protocol': 'Original FSL4D applywarp: one frame-specific MCFLIRT+BBR premat, optional original FNIRT coefficients, one spline interpolation of the raw 4D BOLD per target.',
        'timing_boundary': 'Stage-only continuous sampling call, including original NiWorkflows target-grid generation, Python matrix composition, original process startup/I/O and tracing; source registration estimation is reused from this fresh original run, and final integrity/hash checks are outside this timer.',
        'stage_only_wall_seconds': wall, 'timings': timings, 'checks': checks,
        'original_sampling_reference': json.loads((output / 'sampling_reference.public.json').read_text()),
        'input_sha256': {key: sha256(case[key]) for key in ('bold', 'sbref', 't1w', 'mni_template', 'mni_mask')},
        'reused_fresh_registration_sha256': {
            'bbr': sha256(registration / 'example_func2highres.mat'),
            'fnirt_coefficients': sha256(registration / 'T1_to_MNI_coeff.nii.gz'),
            'motion_sequence': hashlib.sha256('\n'.join(sha256(path) for path in matrices).encode()).hexdigest(),
            'registration_report': sha256(registration / 'registration.public.json')},
        'matrix_series_sha256': {name: sha256(path) for name, path in series.items()},
        'executables_sha256': {'applywarp_launcher': sha256(fsl / 'bin/applywarp'),
                              'fmriprep_container': sha256(container)},
        'driver_sha256': sha256(__file__),
        'grid_driver_sha256': sha256(source / 'make_original_sampling_reference.py'),
        'limits': ['Stage-only runtime; not a new raw-to-CIFTI or combined clean+preproc end-to-end wall clock.',
            'FSL NEWIMAGE third-order spline with extraslice and validity-mask edge rules differs from FNIT preproc fMRIPrep grid-constant sampling.',
            'T1 target is independently generated from the fresh original T1 brain mask; its crop may differ from the candidate T1 crop.',
            'Single real run on a shared host; original FSL uses CPU, configured thread environment limit eight.'],
        'privacy': 'Anonymous scalar checks and hashes only; full matrix series, raw paths and 4D images stay private.'}
    (output / 'preproc.public.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'outputs.private.json').write_text(json.dumps({
        **{key: str(path) for key, path in outputs.items()}, 't1w_reference': str(t1_reference),
        'report': str(output / 'preproc.public.json'), 'frames': raw.shape[3], 'tr_seconds': case['tr'],
        'input_sha256': report['input_sha256'], 'output_sha256': {key: value['sha256'] for key, value in checks.items()},
        'fresh_registration_report': str(registration / 'registration.public.json')}, indent=2) + '\n')
    (output / 'status.private.json').write_text(json.dumps({'stage': 'complete', 'status': 'passed'}) + '\n')
    print(json.dumps({'preproc_complete': True, 'seconds': wall}), flush=True)


if __name__ == '__main__':
    main()
