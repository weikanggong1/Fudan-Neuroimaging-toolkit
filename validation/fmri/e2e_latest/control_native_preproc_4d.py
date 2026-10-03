"""真实帧控制：核对安装的原 FSL4D premat 与独立3D调用。"""
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case-json', type=Path, required=True)
parser.add_argument('--native-root', type=Path, required=True)
parser.add_argument('--preproc-root', type=Path, required=True)
parser.add_argument('--output-root', type=Path, required=True)
args = parser.parse_args()
case = json.loads(args.case_json.read_text())
output = args.output_root
if output.exists():
    raise FileExistsError('Use a new real-frame control directory')
output.mkdir(parents=True)
fsl = Path(case['fsl_root'])
environment = dict(os.environ, FSLDIR=str(fsl), FSLOUTPUTTYPE='NIFTI_GZ',
                   OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
environment['LD_LIBRARY_PATH'] = str(fsl / 'lib') + ':' + environment.get('LD_LIBRARY_PATH', '')
raw = nib.load(case['bold'])
frames = (0, 100, 489)
values = np.asarray(raw.dataobj)
selected = values[..., list(frames)]
raw3 = output / 'raw_three_frames.private.nii.gz'
nib.save(nib.Nifti1Image(selected, raw.affine, raw.header.copy()), raw3)
series = np.loadtxt(args.preproc_root / 'motion_bbr_4d.private.mat').reshape(-1, 4, 4)
matrices = output / 'three_premats.private.mat'
np.savetxt(matrices, series[list(frames)].reshape(-1, 4), fmt='%.17g')
records = []

def run(name, source, matrix_path, destination):
    command = [fsl / 'bin/applywarp', '--in=' + str(source),
        '--ref=' + case['mni_template'],
        '--warp=' + str(args.native_root / 'reg_fnirt/T1_to_MNI_coeff.nii.gz'),
        '--premat=' + str(matrix_path), '--interp=spline', '--datatype=float',
        '--out=' + str(destination)]
    with (output / (name + '.private.log')).open('wb') as log:
        started = time.perf_counter()
        result, evidence = run_traced(command, trace_path=output / (name + '.exec.private.log'),
                                     env=environment, stdout=log, stderr=subprocess.STDOUT)
        wall = time.perf_counter() - started
    if not evidence['original_process_accepted']:
        raise RuntimeError('Original real-frame control failed')
    records.append({'name': name, 'seconds_including_io': wall, 'exit_evidence': evidence})

four_d = output / 'four_d_result.private.nii.gz'
run('four_d', raw3, matrices, four_d)
full_values = np.asarray(nib.load(four_d).dataobj, dtype=np.float32)
per_frame = []
for offset, frame in enumerate(frames):
    source = output / f'raw_frame_{frame:04d}.private.nii.gz'
    matrix = output / f'premat_{frame:04d}.private.mat'
    destination = output / f'result_{frame:04d}.private.nii.gz'
    nib.save(nib.Nifti1Image(values[..., frame], raw.affine, raw.header.copy()), source)
    np.savetxt(matrix, series[frame], fmt='%.17g')
    run(f'frame_{frame:04d}', source, matrix, destination)
    single = np.asarray(nib.load(destination).dataobj, dtype=np.float32)
    previous = full_values[..., offset]
    delta = previous.astype(np.float64) - single.astype(np.float64)
    per_frame.append({'frame': frame,
        'evaluated_values': int(single.size),
        'float32_bitwise_equal': bool(np.array_equal(previous.view(np.uint32), single.view(np.uint32))),
        'bitwise_unequal_values': int(np.count_nonzero(previous.view(np.uint32) != single.view(np.uint32))),
        'max_absolute_difference': float(np.abs(delta).max()),
        'rmse': float(np.sqrt(np.square(delta).mean()))})
report = {'schema_version': 1, 'protocol': 'Original FSL applywarp full3-frame premat series compared with three independent original single-frame commands, using unchanged real frames 0/100/489 and this fresh original registration.',
    'source_real_frames': list(frames), 'frames': per_frame, 'commands': records,
    'installed_4d_premat_supported': True,
    'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope': 'Real-data contract control, separate from whole-pipeline and full490 stage timings.',
    'privacy': 'Only anonymous metrics; images and command paths remain private.'}
(output / 'control.public.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({'installed_4d_premat_supported': True, 'frames': per_frame}), flush=True)
