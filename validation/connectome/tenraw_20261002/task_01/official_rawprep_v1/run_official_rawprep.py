#!/usr/bin/env python3
"""Independent official raw AP/PA -> TOPUP -> SynthStrip CPU -> EDDY GPU.

FNIT is used only to make explicit, matching b0 selection/packing proposals on CPU.
The solver chain consumes newly generated official ROI/merge, TOPUP and mask files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import nibabel as nib
import numpy as np

GPU_UUID = 'GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e'
LOCK = '/tmp/fnit-connectome-tenraw-gongwk.gpu0.lock'
EDDY_FLAGS = ['--flm=quadratic', '--resamp=jac', '--slm=linear', '--niter=8',
              '--fwhm=10,8,4,2,0,0,0,0', '--ff=10', '--sep_offs_move',
              '--nvoxhp=1000', '--repol', '--rms', '--initrand=12345']
EXPECTED_PILOT_SELECTION = {'CON01': (76, 0), 'CON03': (0, 0)}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def descendants(pid):
    found = {pid}
    parents = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            pass
    changed = True
    while changed:
        added = {child for child, parent in parents.items() if parent in found} - found
        changed = bool(added)
        found.update(added)
    return found


def process_snapshot(pid, gpu):
    pids = descendants(pid)
    rss = 0
    for child in pids:
        try:
            rows = Path(f'/proc/{child}/status').read_text().splitlines()
            rss += next(int(row.split()[1]) * 1024 for row in rows if row.startswith('VmRSS:'))
        except (OSError, StopIteration):
            pass
    row = {'monotonic': time.monotonic(), 'pids': sorted(pids), 'aggregate_RSS_bytes': rss}
    if gpu:
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                          '--format=csv,noheader,nounits'], text=True, timeout=10)
        all_rows = [line.split(',') for line in output.splitlines()]
        own = [parts for parts in all_rows if int(parts[0]) in pids]
        row.update(own_GPU_bytes=sum(float(parts[2]) * 1024**2 for parts in own),
                   own_GPU_UUIDs=sorted({parts[1].strip() for parts in own}),
                   shared_GPU_processes=[{'pid': int(parts[0]), 'uuid': parts[1].strip(),
                                          'bytes': float(parts[2]) * 1024**2} for parts in all_rows])
    return row


def run_command(argv, stage, output, env, report, gpu=False):
    record = {'stage': stage, 'command': list(map(str, argv)), 'host': socket.gethostname(),
              'started_utc': utc(), 'GPU': gpu, 'env': {key: env.get(key) for key in
              ['FSLDIR', 'FREESURFER_HOME', 'FSLOUTPUTTYPE', 'OMP_NUM_THREADS',
               'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES']},
              'program_sha256': sha(argv[0]), 'memory_samples': [], 'budget_exceeded': False,
              'sample_period_nominal_seconds': .5}
    report['commands'].append(record)
    save(output / 'report.json', report)
    lock = None
    record['lock_wait_seconds'] = 0.
    if gpu:
        lock = open(env['FNIT_REFERENCE_GPU_LOCK'], 'a')
        requested = time.perf_counter()
        fcntl.flock(lock, fcntl.LOCK_EX)
        record['lock_wait_seconds'] = time.perf_counter() - requested
        report['GPU_lock_wait_seconds'] += record['lock_wait_seconds']
        record['lock_acquired_utc'] = utc()
    started = time.perf_counter()
    try:
        with (output / f'{stage}.log').open('w') as log:
            proc = subprocess.Popen(record['command'], env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            record['pid'] = proc.pid
            if gpu: report['state'] = 'official_EDDY_GPU_running'
            last_save = time.monotonic()
            save(output / 'report.json', report)
            while proc.poll() is None:
                try:
                    sample = process_snapshot(proc.pid, gpu)
                    record['memory_samples'].append(sample)
                    if gpu and sample.get('own_GPU_bytes', 0) >= 20e9:
                        record['budget_exceeded'] = True
                        # Only this command's process group, created above.
                        import signal
                        os.killpg(proc.pid, signal.SIGTERM)
                    if gpu and any(value != GPU_UUID for value in sample.get('own_GPU_UUIDs', [])):
                        record['wrong_GPU_UUID'] = True
                        import signal
                        os.killpg(proc.pid, signal.SIGTERM)
                except Exception as error:
                    record['memory_samples'].append({'monotonic': time.monotonic(), 'error': repr(error)})
                if time.monotonic() - last_save >= 30:
                    save(output / 'report.json', report)
                    last_save = time.monotonic()
                time.sleep(.5)
            record['returncode'] = proc.returncode
        record['wall_seconds'] = time.perf_counter() - started
        record['completed_utc'] = utc()
        record['sampled_GPU_process_peak_bytes'] = max((row.get('own_GPU_bytes', 0) for row in record['memory_samples']), default=0)
        record['sampled_RSS_peak_bytes'] = max((row.get('aggregate_RSS_bytes', 0) for row in record['memory_samples']), default=0)
        if record['returncode'] or record['budget_exceeded'] or record.get('wrong_GPU_UUID'):
            raise RuntimeError(f'{stage} failed: returncode={record["returncode"]}')
    finally:
        if lock:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        save(output / 'report.json', report)


def check_grid(path, source, mask=False):
    image = nib.load(path)
    if image.shape[:3] != source.shape[:3] or not np.allclose(image.affine, source.affine, rtol=0, atol=1e-5):
        raise ValueError(f'grid mismatch: {path}')
    if mask:
        values = np.asarray(image.dataobj)
        if not np.any(values) or not np.isin(values, (0, 1)).all():
            raise ValueError(f'empty or nonbinary official mask: {path}')
    return {'shape': list(image.shape), 'affine': image.affine.tolist(), 'sha256': sha(path)}


def finish_eddy(case, output, args, env, report):
    raw, topup, mask, eddy = [output / name for name in ('raw', 'topup', 'mask', 'eddy')]
    ap = nib.load(raw / 'AP.nii.gz')
    ap_index = report['selection']['ap_index']
    bvals = np.loadtxt(raw / 'AP.bval').reshape(-1)
    inputs = {'imain': raw / 'AP.nii.gz', 'mask': mask / 'nodif_brain_mask.nii.gz',
              'acqp': topup / 'acqparams.txt', 'index': eddy / 'eddy_index.txt',
              'bvecs': raw / 'AP.bvec', 'bvals': raw / 'AP.bval', 'topup': topup / 'fieldmap_out'}
    use_gpu = args.solver == 'gpu'
    eddy_stage = 'official_EDDY_GPU' if use_gpu else 'official_EDDY_CPU'
    report['state'] = eddy_stage + ('_waiting' if use_gpu else '_running'); save(output / 'report.json', report)
    solver_env = dict(env, CUDA_VISIBLE_DEVICES=GPU_UUID if use_gpu else '', FNIT_REFERENCE_GPU_LOCK=str(args.gpu_lock))
    command = [args.fsl_dir / 'bin' / args.eddy_binary, *[f'--{key}={value}' for key, value in inputs.items()],
               f'--out={eddy}/data', *EDDY_FLAGS, f'--ref_scan_no={ap_index}']
    run_command(command, eddy_stage, output, solver_env, report, use_gpu)
    report['EDDY_output_geometry'] = check_grid(eddy / 'data.nii.gz', ap)
    if tuple(report['EDDY_output_geometry']['shape']) != ap.shape:
        raise ValueError('official EDDY altered spatial grid or DWI frame count')
    rotated = np.loadtxt(eddy / 'data.eddy_rotated_bvecs')
    if rotated.shape != (3, bvals.size):
        raise ValueError('official rotated bvec shape mismatch')
    report['output_sha256'] = {str(path.relative_to(output)): sha(path) for directory in (topup, mask, eddy) for path in directory.iterdir() if path.is_file()}
    report.update(completed=True, state='completed', completed_utc=utc())
    contract = {'completed': True, 'subject': report['subject'], 'session': report['session'],
                'origin': 'canonical raw -> own official TOPUP -> own official SynthStrip -> own official EDDY',
                'data': str(eddy / 'data.nii.gz'), 'mask': str(mask / 'nodif_brain_mask.nii.gz'),
                'rotated_bvecs': str(eddy / 'data.eddy_rotated_bvecs'), 'bvals': str(raw / 'AP.bval'),
                'topup_prefix': str(topup / 'fieldmap_out'), 'T1w': case['t1w'],
                'output_sha256': report['output_sha256'], 'raw_sha256': report['input_sha256'],
                'GP_seed': report['gp_seed'], 'ref_scan_no': ap_index, 'GPU_UUID': report['GPU_UUID'], 'EDDY_solver': args.solver}
    save(output / 'completed_contract.json', contract)


def run_subject(case, args, prepare):
    subject = case['subject'].removeprefix('sub-')
    output = args.output_root / f'sub-{subject}'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = {'subject': subject, 'session': case['session'], 'completed': False, 'state': 'started',
              'scope': 'independent official rawprep; not anatomy/modeling/tracking/full connectome',
              'host': socket.gethostname(), 'started_utc': utc(), 'commands': [],
              'GPU_UUID': GPU_UUID if args.solver == 'gpu' else None, 'GPU_lock': str(args.gpu_lock) if args.solver == 'gpu' else None, 'GPU_lock_wait_seconds': 0., 'EDDY_solver': args.solver,
              'gp_seed': 12345, 'threads': 8, 'input_sha256': {}, 'source_sha256': args.source_hashes,
              'timings': {}}
    save(output / 'report.json', report)
    env = dict(os.environ, FSLDIR=str(args.fsl_dir), FREESURFER_HOME=str(args.freesurfer_dir),
               FSLOUTPUTTYPE='NIFTI_GZ', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='8')
    env['PATH'] = f'{args.freesurfer_dir}/bin:{args.fsl_dir}/bin:' + env.get('PATH', '')
    try:
        stage = time.perf_counter()
        raw = output / 'raw'; raw.mkdir()
        selected = {name: value for name, value in case['input_sha256'].items()}
        for relative, expected_sha in selected.items():
            path = args.raw_root / relative
            actual = sha(path)
            if actual != expected_sha:
                raise ValueError(f'canonical raw SHA mismatch: {path}')
            report['input_sha256'][str(path)] = actual
            if '/dwi/' in relative:
                stem = 'AP' if '_acq-AP_' in relative else 'PA' if '_acq-PA_' in relative else None
                if stem:
                    extension = '.nii.gz' if relative.endswith('.nii.gz') else '.' + Path(relative).suffix.removeprefix('.')
                    (raw / f'{stem}{extension}').symlink_to(path)
        report['timings']['raw_verify_stage_seconds'] = time.perf_counter() - stage
        stage = time.perf_counter()
        proposal = prepare(raw, output / 'selection', device='cpu', pair_geometry='fslmerge-first')
        ap_index, pa_index = proposal['ap_index'], proposal['pa_index']
        if subject in EXPECTED_PILOT_SELECTION and (ap_index, pa_index) != EXPECTED_PILOT_SELECTION[subject]:
            raise ValueError('CPU b0 selection differs from recorded FNIT pilot; no silent substitution')
        report['selection'] = {key: str(value) if isinstance(value, Path) else value for key, value in proposal.items()}
        report['selection']['device'] = 'cpu; same FNIT policy, FP32 CPU scores recorded; compare selected frames to FNIT per case'
        report['selection']['original_JSON'] = {stem: json.loads((raw / f'{stem}.json').read_text()) for stem in ('AP', 'PA')}
        report['selection']['effective_acqparams'] = np.loadtxt(proposal['datain']).tolist()
        report['timings']['b0_selection_seconds'] = time.perf_counter() - stage
        selection_reference = None
        if args.fnit_selection_root is not None:
            fnit_topup = args.fnit_selection_root / f'sub-{subject}/connectome/preproc/topup'
            selection_reference = fnit_topup / 'B0_AP_PA.nii.gz'
            wait_started = time.perf_counter()
            report['state'] = 'waiting_actual_FNIT_b0_packing'
            save(output / 'report.json', report)
            while not selection_reference.is_file() or not (fnit_topup / 'acqparams.txt').is_file():
                time.sleep(15)
            selected_pair = nib.load(selection_reference)
            selected_values = np.asarray(selected_pair.dataobj)
            raw_ap = nib.load(raw / 'AP.nii.gz')
            if selected_values.shape != (*raw_ap.shape[:3], 2) or not np.array_equal(selected_pair.affine, raw_ap.affine):
                raise ValueError('actual FNIT selection pair grid differs from canonical AP')
            exact_indices = []
            for stem, pair_index in [('AP', 0), ('PA', 1)]:
                original_values = np.asarray(nib.load(raw / f'{stem}.nii.gz').dataobj)
                original_bvals = np.loadtxt(raw / f'{stem}.bval').reshape(-1)
                matches = [int(index) for index in np.where(original_bvals < 100)[0]
                           if np.array_equal(original_values[..., index], selected_values[..., pair_index])]
                if len(matches) != 1: raise ValueError('actual FNIT b0 does not match a unique canonical raw frame')
                exact_indices.append(matches[0])
            if not np.array_equal(np.loadtxt(proposal['datain']), np.loadtxt(fnit_topup / 'acqparams.txt')):
                raise ValueError('actual FNIT effective readout/PE differs from raw policy')
            report['CPU_selection_proposal'] = dict(report['selection'])
            ap_index, pa_index = exact_indices
            if subject in EXPECTED_PILOT_SELECTION and tuple(exact_indices) != EXPECTED_PILOT_SELECTION[subject]:
                raise ValueError('actual pilot selection differs from frozen expected indices')
            report['selection'].update(ap_index=ap_index, pa_index=pa_index,
                                       device='actual formal FNIT GPU/TF32 selection; raw frame index identified by exact native packing, CPU proposal preserved separately')
            report['FNIT_selection_evidence'] = {'packed_pair': str(selection_reference), 'packed_pair_sha256': sha(selection_reference),
                                               'effective_acqparams': str(fnit_topup / 'acqparams.txt'), 'effective_acqparams_sha256': sha(fnit_topup / 'acqparams.txt'),
                                               'raw_indices': exact_indices, 'every_selected_voxel_matches_canonical_raw': True,
                                               'scope': 'b0 choice metadata only; no FNIT field/mask/corrected DWI reused'}
            report['timings']['actual_FNIT_selection_wait_and_read_seconds'] = time.perf_counter() - wait_started
        topup = output / 'topup'; topup.mkdir()
        mask = output / 'mask'; mask.mkdir()
        eddy = output / 'eddy'; eddy.mkdir()
        shutil.copyfile(proposal['datain'], topup / 'acqparams.txt')
        run_command([args.fsl_dir / 'bin/fslroi', raw / 'AP.nii.gz', topup / 'AP_b0.nii.gz', str(ap_index), '1'], 'roi_AP', output, env, report)
        run_command([args.fsl_dir / 'bin/fslroi', raw / 'PA.nii.gz', topup / 'PA_b0.nii.gz', str(pa_index), '1'], 'roi_PA', output, env, report)
        run_command([args.fsl_dir / 'bin/fslmerge', '-t', topup / 'B0_AP_PA.nii.gz', topup / 'AP_b0.nii.gz', topup / 'PA_b0.nii.gz'], 'merge_pair', output, env, report)
        packed = nib.load(topup / 'B0_AP_PA.nii.gz'); suggested = nib.load(selection_reference if selection_reference is not None else proposal['imain'])
        if not np.array_equal(np.asarray(packed.dataobj), np.asarray(suggested.dataobj)) or not np.array_equal(packed.affine, suggested.affine):
            raise ValueError('official fslroi/merge pair differs from frozen FNIT selection proposal')
        report['packed_pair_equals_FNIT_selection_voxels_affine'] = True
        report['TOPUP_inputs_sha256'] = {str(path): sha(path) for path in [topup / 'B0_AP_PA.nii.gz', topup / 'acqparams.txt']}
        report['state'] = 'official_topup_running'; save(output / 'report.json', report)
        config = args.fsl_dir / 'etc/flirtsch/b02b0.cnf'
        run_command([args.fsl_dir / 'bin/topup', f'--imain={topup}/B0_AP_PA.nii.gz',
                     f'--datain={topup}/acqparams.txt', f'--config={config}',
                     f'--out={topup}/fieldmap_out', f'--fout={topup}/fieldmap_fout',
                     f'--iout={topup}/fieldmap_iout'], 'official_topup', output, env, report)
        ap = nib.load(raw / 'AP.nii.gz')
        report['TOPUP_iout_geometry'] = check_grid(topup / 'fieldmap_iout.nii.gz', ap)
        run_command([args.fsl_dir / 'bin/fslmaths', topup / 'fieldmap_iout.nii.gz', '-Tmean', mask / 'b0_mean.nii.gz'], 'official_b0_mean', output, env, report)
        report['mask_input_geometry'] = check_grid(mask / 'b0_mean.nii.gz', ap)
        model = args.freesurfer_dir / 'models/synthstrip.1.pt'
        report['SynthStrip_model'] = {'path': str(model), 'bytes': model.stat().st_size, 'sha256': sha(model), 'border_mm': 1, 'no_csf': False, 'device': 'CPU'}
        if model.stat().st_size != 30851709 or sha(model) != '37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33':
            raise ValueError('official SynthStrip model size/SHA mismatch')
        report['state'] = 'official_synthstrip_running'; save(output / 'report.json', report)
        run_command([args.freesurfer_dir / 'bin/mri_synthstrip', '-i', mask / 'b0_mean.nii.gz',
                     '-m', mask / 'nodif_brain_mask.nii.gz', '-o', mask / 'nodif_brain.nii.gz',
                     '--model', model, '-b', '1', '-t', '8'], 'official_synthstrip_CPU', output, env, report)
        report['mask_geometry'] = check_grid(mask / 'nodif_brain_mask.nii.gz', ap, True)
        bvals = np.loadtxt(raw / 'AP.bval').reshape(-1)
        index = eddy / 'eddy_index.txt'; np.savetxt(index, np.ones(bvals.size, dtype=int)[None], fmt='%d')
        inputs = {'imain': raw / 'AP.nii.gz', 'mask': mask / 'nodif_brain_mask.nii.gz',
                  'acqp': topup / 'acqparams.txt', 'index': index, 'bvecs': raw / 'AP.bvec',
                  'bvals': raw / 'AP.bval', 'topup': topup / 'fieldmap_out'}
        dependency_paths = [value for key, value in inputs.items() if key != 'topup'] + [topup / 'fieldmap_out_fieldcoef.nii.gz', topup / 'fieldmap_out_movpar.txt']
        report['EDDY_inputs_sha256'] = {str(path): sha(path) for path in dependency_paths}
        report['EDDY_input_origin'] = 'this subject/new namespace official TOPUP fieldcoef+movpar and official SynthStrip mask; no FNIT field/mask used'
        report.update(CPU_prepared=True, CPU_completed_utc=utc(), CPU_wall_seconds=time.perf_counter() - started)
        if args.phase == 'cpu':
            report['state'] = 'CPU_prepared_EDDY_not_run'
            return report
        finish_eddy(case, output, args, env, report)
    except BaseException as error:
        report.update(completed=False, state='failed', error=repr(error), traceback=traceback.format_exc(), completed_utc=utc())
    finally:
        report['observed_rawprep_total_wall_seconds'] = time.perf_counter() - started
        report['rawprep_wall_excluding_GPU_queue_seconds'] = report['observed_rawprep_total_wall_seconds'] - report['GPU_lock_wait_seconds']
        report['timing_policy'] = 'whole subject measured wall; queue wait separately recorded. CPU stages run on declared host, not nodecw10. Stage times not medians; hash/read/write and startup included.'
        save(output / 'report.json', report)
        if args.phase == 'cpu' and report.get('CPU_prepared'):
            save(output / 'CPU_prepared_report.json', report)
    return report


def run_gpu_subject(case, args):
    subject = case['subject'].removeprefix('sub-')
    output = args.output_root / f'sub-{subject}'
    report = json.loads((output / 'report.json').read_text())
    if report.get('state') != 'CPU_prepared_EDDY_not_run' or not report.get('CPU_prepared'):
        raise ValueError(f'{subject}: GPU activation requires untouched successful CPU preparation')
    started = time.perf_counter()
    activated_utc = utc()
    report['GPU_activation_utc'] = activated_utc
    report['GPU_activation_source_SHA'] = args.source_hashes
    env = dict(os.environ, FSLDIR=str(args.fsl_dir), FREESURFER_HOME=str(args.freesurfer_dir),
               FSLOUTPUTTYPE='NIFTI_GZ', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    env['PATH'] = f'{args.freesurfer_dir}/bin:{args.fsl_dir}/bin:' + env.get('PATH', '')
    try:
        for relative, digest in case['input_sha256'].items():
            if sha(args.raw_root / relative) != digest:
                raise ValueError('canonical raw changed after CPU preparation')
        for path, digest in report['EDDY_inputs_sha256'].items():
            if sha(path) != digest:
                raise ValueError('official field/mask/gradient dependency changed after CPU preparation')
        binary = str(args.fsl_dir / 'bin/eddy_cuda10.2')
        if report['source_sha256'][binary] != args.source_hashes[binary]:
            raise ValueError('frozen native EDDY binary changed')
        finish_eddy(case, output, args, env, report)
    except BaseException as error:
        report.update(completed=False, state='failed', error=repr(error), traceback=traceback.format_exc(), completed_utc=utc())
    finally:
        report['GPU_activation_wall_seconds'] = time.perf_counter() - started
        report['observed_rawprep_total_wall_seconds'] = (datetime.now(timezone.utc) - datetime.fromisoformat(report['started_utc'])).total_seconds()
        report['CPU_to_GPU_phase_gap_seconds'] = (datetime.fromisoformat(activated_utc) - datetime.fromisoformat(report['CPU_completed_utc'])).total_seconds()
        report['rawprep_wall_excluding_GPU_queue_seconds'] = report['observed_rawprep_total_wall_seconds'] - report['GPU_lock_wait_seconds']
        report['active_stage_wall_excluding_GPU_queue_seconds'] = report['CPU_wall_seconds'] + report['GPU_activation_wall_seconds'] - report['GPU_lock_wait_seconds']
        report['timing_policy'] = 'Two-phase rawprep. Measured wall includes CPU-to-GPU phase gap; gap, GPU queue and active phase wall separately recorded. Do not treat active stage sum as uninterrupted end-to-end wall.'
        save(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--fnit-source', type=Path, required=True)
    parser.add_argument('--fsl-dir', type=Path, default=Path('/public/software/apps/FSL/6.0.7.4'))
    parser.add_argument('--freesurfer-dir', type=Path, default=Path('/public/software/apps/Freesurfer/8.2.0-1'))
    parser.add_argument('--gpu-lock', type=Path, default=Path(LOCK))
    parser.add_argument('--subjects', nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--phase', choices=('all', 'cpu', 'gpu'), default='all')
    parser.add_argument('--solver', choices=('gpu', 'cpu'), default='gpu', help='CPU fallback uses installed eddy_cpu, CPU8, same scientific flags; fresh all phase only')
    parser.add_argument('--fnit-selection-root', type=Path, help='actual formal FNIT baseline root; identify raw b0 indices from exact packing, never reuse field/mask')
    args = parser.parse_args()
    args.eddy_binary = 'eddy_cuda10.2' if args.solver == 'gpu' else 'eddy_cpu'
    if args.solver == 'cpu' and args.phase != 'all': parser.error('CPU fallback requires fresh all-phase independent chain')
    if args.workers not in (1, 2): parser.error('CPU reference concurrency must be 1 or 2, each8 threads')
    if args.phase != 'gpu' and args.output_root.exists() and any(args.output_root.iterdir()): parser.error('fresh namespace required')
    if args.phase == 'gpu' and not (args.output_root / 'freeze.json').is_file(): parser.error('GPU phase requires its own frozen CPU namespace')
    args.output_root.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Parent/b0 policy are strictly CPU.
    sys.path.insert(0, str(args.fnit_source / 'src'))
    import torch
    torch.set_num_threads(8)
    from fnit.topup.ukb import prepare_ukb_topup
    paths = [Path(__file__), args.fnit_source / 'src/fnit/topup/ukb.py', args.fnit_source / 'src/fnit/topup/core.py',
             args.fsl_dir / 'etc/flirtsch/b02b0.cnf', args.freesurfer_dir / 'python/scripts/mri_synthstrip',
             args.freesurfer_dir / 'bin/fspython', args.freesurfer_dir / 'models/synthstrip.1.pt']
    paths += [args.fsl_dir / 'bin' / name for name in ('fslroi', 'fslmerge', 'topup', 'fslmaths', args.eddy_binary)]
    paths += [args.freesurfer_dir / 'bin/mri_synthstrip', args.freesurfer_dir / 'build-stamp.txt']
    args.source_hashes = {str(path): sha(path) for path in paths}
    payload = json.loads(args.manifest.read_text())
    wanted = [value.removeprefix('sub-') for value in args.subjects]
    cases = [case for case in payload['cases'] if case['subject'].removeprefix('sub-') in wanted]
    if len(cases) != len(wanted): raise ValueError('subjects absent/duplicate in canonical manifest')
    spec = {'scope': 'independent official rawprep, not full anatomical/downstream connectome',
            'host': socket.gethostname(), 'created_utc': utc(), 'manifest_sha256': sha(args.manifest),
            'subjects': wanted, 'raw_root': str(args.raw_root), 'output_root': str(args.output_root),
            'source_sha256': args.source_hashes, 'FNIT_selection_root': str(args.fnit_selection_root) if args.fnit_selection_root else None, 'CPU_threads_per_subject': 8, 'CPU_concurrency': args.workers,
            'parent_CUDA_VISIBLE_DEVICES': '', 'GPU_UUID': GPU_UUID if args.solver == 'gpu' else None, 'GPU_lock': str(args.gpu_lock) if args.solver == 'gpu' else None, 'EDDY_solver': args.solver,
            'EDDY_flags': EDDY_FLAGS, 'b0_selection_policy': 'FNIT CPU FP32 pairwise rigid; b<100; first if score>=0.98 else best; both pilots must match recorded indices; official ROI/merge voxel+affine check',
            'pair_geometry': 'fslmerge-first; rigid/no shear/same spacing/matrix/handedness; no resampling',
            'readout_policy': 'FNIT floor(TotalReadoutTime*10000)/10000, original JSON retained',
            'model_SHA_expected': '37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33'}
    spec['phase'] = args.phase
    if args.phase == 'gpu':
        frozen = json.loads((args.output_root / 'freeze.json').read_text())
        if frozen['manifest_sha256'] != spec['manifest_sha256'] or frozen['subjects'] != wanted:
            raise ValueError('GPU activation manifest/subjects differ from frozen CPU preparation')
        save(args.output_root / 'GPU_activation_freeze.json', spec)
    else:
        save(args.output_root / 'freeze.json', spec)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = ([pool.submit(run_gpu_subject, case, args) for case in cases] if args.phase == 'gpu'
                   else [pool.submit(run_subject, case, args, prepare_ukb_topup) for case in cases])
        for future in futures:
            report = future.result()
            print(json.dumps({'subject': report['subject'], 'state': report['state'], 'completed': report['completed'], 'error': report.get('error')}), flush=True)
    reports = []
    for case in cases:
        subject = case['subject'].removeprefix('sub-')
        reports.append(json.loads((args.output_root / f'sub-{subject}' / 'report.json').read_text()))
    summary = {'freeze': spec, 'phase': args.phase, 'subjects': reports,
               'all_rawprep_completed': all(value['completed'] for value in reports),
               'all_CPU_prepared': all(value.get('CPU_prepared', False) for value in reports)}
    save(args.output_root / f'summary_{args.phase}.json', summary)
    success = summary['all_CPU_prepared'] if args.phase == 'cpu' else summary['all_rawprep_completed']
    if not success: sys.exit(1)


if __name__ == '__main__': main()
