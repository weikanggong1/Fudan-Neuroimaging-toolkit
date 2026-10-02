"""Real FS-brain AB/BA for omitting unused SynthMorph inverse/moved outputs.

Each arm runs in a fresh process with identical allocator/TF32/model defaults.
Forward warp and CPU NN Tian S1/S4 must match exactly across all four runs.
Output writes and input/weight SHA verification are excluded from compute walls.
Run the entire parent under the shared GPU flock. No production source mutation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import nibabel as nib
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def child(args):
    import torch
    from fnit.synthmorph import SynthMorph, apply_transform
    from fnit.weights import WEIGHT_FILES, verify_file
    args.output_dir.mkdir(parents=True, exist_ok=False)
    verified = {}
    for name in ('synthmorph.affine.2.h5', 'synthmorph.deform.3.h5'):
        _, size, sha = WEIGHT_FILES[name]
        if not verify_file(args.weights / name, size, sha): raise ValueError('weight verification failed')
        verified[name] = {'size': size, 'sha256': sha}
    source = Path(__file__).resolve().parents[1]
    report = {'arm': args.arm, 'verified_weights': verified,
        'inputs': {name: {'path': str(path), 'sha256': digest(path)} for name, path in (('t1', args.t1), ('mni', args.mni))},
        'sources': {name: digest(source / name) for name in ('src/fnit/synthmorph/pipeline.py', 'src/fnit/synthmorph/models.py')},
        'benchmark_sha256': digest(__file__), 'torch_version': torch.__version__, 'threads': torch.get_num_threads(),
        'allocator': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING'), 'torch_memory_stats_valid': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') is None,
        'parameters': {'model': 'joint', 'extent': 256, 'hyper': 0.5, 'steps': 7, 'device': args.device, 'dtype': 'float32', 'TF32': True, 'autocast': False},
        'timing_scope': 'model load and registration synchronized; image files opened before timer, voxel decoding/H2D inside call included; hashes/write excluded; CPU NN separately timed'}
    def gpu_snapshot():
        return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()
    report['gpu_state_before'] = gpu_snapshot()
    samples, failures, other_samples = [], [], []
    stopped = threading.Event()
    def sample():
        while not stopped.is_set():
            try:
                rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=2).splitlines()
                own = [row.split(',') for row in rows if int(row.split(',')[0].strip()) == os.getpid()]
                samples.append((time.monotonic(), sum(int(row[2].strip()) * 1024**2 for row in own), [row[1].strip() for row in own]))
                other_samples.append([(row.split(',')[1].strip(), int(row.split(',')[2].strip()) * 1024**2) for row in rows if int(row.split(',')[0].strip()) != os.getpid()])
            except Exception as e: failures.append(type(e).__name__)
            stopped.wait(.2)
    thread = threading.Thread(target=sample, daemon=True); thread.start()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.empty(1, device=args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    moving, fixed = nib.load(args.mni), nib.load(args.t1)
    torch.cuda.synchronize(args.device); start = time.perf_counter()
    model = SynthMorph(weights=args.weights, device=args.device, model='joint')
    torch.cuda.synchronize(args.device); loaded = time.perf_counter()
    kwargs = {'transform_only': True, 'compute_inverse': False} if args.arm == 'candidate' else {}
    precision = []
    result = model(moving=moving, fixed=fixed, precision_report=precision, **kwargs)
    torch.cuda.synchronize(args.device); ended = time.perf_counter()
    report.update(model_load_seconds=loaded-start, registration_seconds=ended-loaded, registration_with_model_load_seconds=ended-start, precision=precision,
        omitted_outputs={name: getattr(result, name) is None for name in ('inverse', 'moved', 'fixed_moved')})
    result.transform.save(args.output_dir / 'forward.nii.gz')
    report['NN_cpu_seconds'] = {}
    for level in (1, 4):
        path = args.atlas_dir / f'Tian_Subcortex_S{level}_3T.nii.gz'
        atlas = nib.load(path)
        report['inputs'][f'tian_s{level}'] = {'path': str(path), 'sha256': digest(path)}
        start = time.perf_counter()
        labels = apply_transform(image=atlas, transformation=result.transform, method='nearest', dtype='int16')
        report['NN_cpu_seconds'][f's{level}'] = time.perf_counter()-start
        nib.save(labels, args.output_dir / f'tian_s{level}.nii.gz')
    report['torch_peak_bytes'] = {'allocated': torch.cuda.max_memory_allocated(args.device), 'reserved': torch.cuda.max_memory_reserved(args.device)}
    stopped.set(); thread.join()
    report['nvml'] = {'peak_process_bytes': max((r[1] for r in samples), default=None), 'samples': len(samples), 'failures': failures,
        'gpu_uuids': sorted({u for row in samples for u in row[2]}), 'max_interval_seconds': max((samples[i][0]-samples[i-1][0] for i in range(1, len(samples))), default=None)}
    own_uuids = set(report['nvml']['gpu_uuids'])
    report['gpu_state_after'] = gpu_snapshot()
    report['other_processes_on_target_gpu_peak_bytes'] = max((sum(memory for uuid, memory in rows if uuid in own_uuids) for rows in other_samples), default=0)
    report['shared_gpu_timing_contaminated'] = report['other_processes_on_target_gpu_peak_bytes'] > 0
    peak = report['nvml']['peak_process_bytes']
    report['memory_budget_passed'] = peak is not None and 0 < peak < 20_000_000_000 and len(report['nvml']['gpu_uuids']) == 1 and max(report['torch_peak_bytes'].values()) < 20_000_000_000
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(args.arm, report['registration_with_model_load_seconds'], peak, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('t1', 'mni', 'atlas-dir', 'weights', 'output-dir'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--arm', choices=('baseline', 'candidate'))
    args = parser.parse_args()
    if args.arm: return child(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tolerance = {'warp_neq': 0, 'warp_max_abs': 0.0, 'label_neq': 0, 'affine_equal': True, 'dtype_equal': True, 'speed_criterion': 'candidate registration_seconds faster in both AB and BA pairs; model+registration mean faster'}
    (args.output_dir / 'declared_tolerance.json').write_text(json.dumps(tolerance, indent=2)+'\n')
    runs = []
    for index, arm in enumerate(('baseline', 'candidate', 'candidate', 'baseline')):
        folder = args.output_dir / f'{index}_{arm}'
        command = [sys.executable, __file__, '--arm', arm, '--device', args.device]
        for name in ('t1', 'mni', 'atlas-dir', 'weights'): command += ['--'+name, str(getattr(args, name.replace('-', '_')))]
        command += ['--output-dir', str(folder)]
        subprocess.run(command, check=True)
        runs.append(json.loads((folder / 'report.json').read_text()))
    comparisons = {}
    for filename in ('forward.nii.gz', 'tian_s1.nii.gz', 'tian_s4.nii.gz'):
        reference = nib.load(args.output_dir / '0_baseline' / filename)
        a = np.asarray(reference.dataobj)
        for index, arm in enumerate(('candidate', 'candidate', 'baseline'), 1):
            current = nib.load(args.output_dir / f'{index}_{arm}' / filename)
            b = np.asarray(current.dataobj)
            geometry_equal = a.shape == b.shape and np.array_equal(reference.affine, current.affine)
            dtype_equal = a.dtype == b.dtype
            header_equal = reference.header.binaryblock == current.header.binaryblock
            extensions_equal = [(e.get_code(), e.content) for e in reference.header.extensions] == [(e.get_code(), e.content) for e in current.header.extensions]
            row = {'geometry_equal': geometry_equal, 'dtype_equal': dtype_equal, 'header_equal': header_equal, 'extensions_equal': extensions_equal, 'shape': list(a.shape), 'dtype': str(a.dtype)}
            if geometry_equal:
                delta = a.astype(np.float64)-b.astype(np.float64)
                absolute = np.abs(delta)
                row.update(neq=int(np.count_nonzero(a != b)), max_abs=float(absolute.max()), p99_abs=float(np.percentile(absolute, 99)), rmse=float(np.sqrt(np.mean(delta*delta))))
            row['passed'] = geometry_equal and dtype_equal and header_equal and extensions_equal and row.get('neq') == 0
            comparisons[f'baseline0_vs_{index}_{arm}/{filename}'] = row
    report = {'scope': 'new CON03 official FS brain SynthMorph/Tian component ABBA; not raw full pipeline wall',
        'tolerance': tolerance, 'runs': runs, 'comparisons': comparisons,
        'parity_passed': all(row['passed'] for row in comparisons.values()),
        'memory_budget_passed': all(row['memory_budget_passed'] for row in runs)}
    report['passed'] = report['parity_passed'] and report['memory_budget_passed']
    report['baseline_mean_seconds'] = float(np.mean([row['registration_with_model_load_seconds'] for row in runs if row['arm']=='baseline']))
    report['candidate_mean_seconds'] = float(np.mean([row['registration_with_model_load_seconds'] for row in runs if row['arm']=='candidate']))
    report['registration_pairs_faster'] = runs[1]['registration_seconds'] < runs[0]['registration_seconds'] and runs[2]['registration_seconds'] < runs[3]['registration_seconds']
    report['speed_gain_observed'] = report['registration_pairs_faster'] and report['candidate_mean_seconds'] < report['baseline_mean_seconds']
    report['registration_with_load_speedup_ratio'] = report['baseline_mean_seconds'] / report['candidate_mean_seconds']
    report['timing_clean_observed'] = not any(row['shared_gpu_timing_contaminated'] or row['nvml']['failures'] for row in runs)
    report['adoption_evidence_passed'] = report['passed'] and report['speed_gain_observed'] and report['timing_clean_observed']
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']: raise SystemExit('forward warp/labels or memory budget failed')

if __name__ == '__main__': main()
