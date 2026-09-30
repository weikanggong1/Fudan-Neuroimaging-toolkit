"""Run a frozen 30,000-person CPU or GPU public-API benchmark.

Usage: python benchmark_public_real30000.py EXPERIMENT_DIR cpu|cuda:0
Private subjects/config/source provenance stay in EXPERIMENT_DIR. Each backend
starts from raw NIfTI in its own fresh output directory. C20 is never reduced
silently. progress.json and resources.jsonl remain useful after a gate failure.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import resource
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import sklearn
import torch

from fnit.bigflica import apply_model, run_bigflica
import fnit.bigflica.pipeline as pipeline
import fnit.bigflica.pipeline_cpu_stream as cpu_stream
import fnit.bigflica.pipeline_gpu as gpu_pipeline
import fnit.bigflica.streaming as streaming


def main():
    experiment = Path(sys.argv[1])
    device = sys.argv[2]
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('Expected cpu or cuda:0')
    label = 'cpu' if device == 'cpu' else 'gpu'
    run = experiment / label
    run.mkdir(exist_ok=False)
    output = run / 'output'
    config = json.loads((experiment / 'input.json').read_text())
    subjects = (experiment / 'subjects.txt').read_text().splitlines()
    if len(subjects) != 30000 or len(set(subjects)) != 30000:
        raise ValueError('Expected exactly 30000 unique subjects')
    if hashlib.sha256((experiment / 'subjects.txt').read_bytes()).hexdigest() != config['subjects_sha256']:
        raise ValueError('Subject list hash changed')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if device != 'cpu':
        if torch.cuda.mem_get_info(0)[0] < 20 * 2**30:
            raise MemoryError('Selected GPU has less than 20 GiB free')
        torch.cuda.set_per_process_memory_fraction(20 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    state = {'status': 'running', 'stage': 'input_metadata', 'backend': label,
             'pid': os.getpid(), 'subjects': 30000, 'source_commit': config['source_commit'],
             'parameters': {'n_components': 20, 'migp_dim': 100, 'dicl_dim': 200,
                            'dicl_max_iter': 1000, 'flica_max_iter': 1000,
                            'flica_lambda_dims': 'R', 'seed': 0, 'cpu_threads': 8,
                            'feature_block': 2048, 'max_gpu_gb': 19},
             'stage_timings_s': {}, 'images_read': 0,
             'versions': {'torch': torch.__version__, 'numpy': np.__version__,
                          'sklearn': sklearn.__version__, 'nibabel': nib.__version__},
             'normalized_dtype': 'float64' if device == 'cpu' else 'float32',
             'subjects_sha256': config['subjects_sha256'],
             'cold_start': True, 'cpu_gpu_execution': 'sequential; independent output/cache directories',
             'os_page_cache_controlled': False, 'shared_gpu': True}
    lock = threading.RLock()
    stop = threading.Event()

    def write_status():
        with lock:
            state['wall_s'] = time.perf_counter() - start
            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            state['peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            if device != 'cpu':
                state['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
                state['peak_gpu_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
            temporary = run / 'progress.json.tmp'
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(run / 'progress.json')

    def monitor():
        ticks = 0
        while not stop.wait(5):
            write_status()
            sample = {key: state[key] for key in ('updated_utc', 'stage', 'wall_s', 'peak_rss_gib')}
            if device != 'cpu':
                sample.update({key: state[key] for key in ('peak_gpu_allocated_gib', 'peak_gpu_reserved_gib')})
            with (run / 'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(sample) + '\n')
            if state['peak_rss_gib'] >= 32:
                state.update(status='memory_limit_failed', error='Host RSS reached 32 GiB')
                write_status()
                os._exit(88)
            ticks += 1
            if ticks % 12 == 0:
                print(json.dumps({'heartbeat': label, 'stage': state['stage'],
                                  'wall_s': state['wall_s'], 'images_read': state['images_read']}), flush=True)

    def tracked(function, stage):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            state['stage'] = stage
            write_status()
            began = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                state['stage_timings_s'][stage + '_s'] = time.perf_counter() - began
                write_status()
        return wrapper

    original_signature = pipeline._signature
    def signature(records):
        digest = original_signature(records)
        if isinstance(records, dict) and all(key in records for key in ('ids', 'specs', 'masks', 'images')):
            state['input_signature'] = digest
            write_status()
        return digest
    pipeline._signature = signature

    original_read = streaming._read_vector
    def read_vector(*args, **kwargs):
        result = original_read(*args, **kwargs)
        state['images_read'] += 1
        if state['images_read'] % 250 == 0:
            write_status()
        return result
    streaming._read_vector = read_vector
    owner = cpu_stream if device == 'cpu' else gpu_pipeline
    owner.prepare_modalities = tracked(owner.prepare_modalities, 'load_standardize')
    owner.fit_mmigp_streaming = tracked(owner.fit_mmigp_streaming, 'mmigp')
    if device == 'cpu':
        # CPU DicL calls once per modality; add their times instead of replacing.
        original_dicl = owner.fit_dicl
        def fit_dicl(*args, **kwargs):
            state['stage'] = 'dicl'
            write_status()
            began = time.perf_counter()
            result = original_dicl(*args, **kwargs)
            state['stage_timings_s']['dicl_s'] = state['stage_timings_s'].get('dicl_s', 0) + time.perf_counter() - began
            write_status()
            return result
        owner.fit_dicl = fit_dicl
    else:
        owner.fit_dicl_gpu_streaming = tracked(owner.fit_dicl_gpu_streaming, 'dicl')
    owner._fit_flica = tracked(owner._fit_flica, 'flica')
    owner._save_gpu_results = tracked(owner._save_gpu_results, 'maps_and_projection')
    write_status()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    exit_code = 0
    try:
        model = run_bigflica(config['subjects_root'], config['modalities'], output,
                             20, migp_dim=100, dicl_dim=200, subjects=subjects,
                             device=device, dicl_max_iter=1000, flica_max_iter=1000,
                             top_voxels=1000, random_state=0, max_gpu_gb=19,
                             feature_block=2048, flica_lambda_dims='R')
        metadata = json.loads((model / 'model.json').read_text())
        state['stage_timings_s'] = metadata['timings']
        state['flica_diagnostics'] = json.loads((model / 'flica_reconstruction.json').read_text())
        state['input_signature'] = metadata['input_signature']
        course = np.load(model / 'subj_course.npy', mmap_mode='r')
        if course.shape != (30000, 20) or not np.isfinite(course).all():
            raise ValueError('Course shape or values failed')
        state['course_shape'] = list(course.shape)
        for name in config['modalities']:
            mask = np.asarray(nib.load(str(model / f'{name}_mask.nii.gz')).dataobj) > 0
            z = np.load(model / f'{name}_zstat.npy', mmap_mode='r')
            for component in range(20):
                prefix = model / 'maps' / name / f'component-{component+1:03d}'
                volume = np.asarray(nib.load(str(prefix)+'_zstat.nii.gz').dataobj)[mask]
                if not np.array_equal(volume, z[:, component]):
                    raise ValueError('NIfTI z-stat differs from saved array')
                if not Path(str(prefix)+'_top-1000.png').is_file():
                    raise ValueError('Thresholded plot missing')
        held_out = apply_model(model, Path(config['subjects_root']) / config['held_out_subject'], device=device)
        np.save(run / 'held_out_course.npy', held_out)
        if held_out.shape != (20,) or not np.isfinite(held_out).all():
            raise ValueError('Held-out projection failed')
        state.update(status='complete', stage='finished', held_out_finite=True)
    except Exception as error:
        state['error'] = f'{type(error).__name__}: {error}'
        diagnostic = output / 'components_20_lambda_R' / 'flica_reconstruction.json'
        if diagnostic.exists():
            state['flica_diagnostics'] = json.loads(diagnostic.read_text())
        if 'FLICA collapsed or pruned requested components' in str(error):
            state['status'] = 'scientific_gate_failed'
            exit_code = 2
        else:
            state['status'] = 'failed'
            exit_code = 1
        import traceback
        traceback.print_exc()
    finally:
        stop.set()
        thread.join(timeout=6)
        eigen = output / 'mmigp_100' / 'eigen_diagnostics.json'
        if eigen.exists():
            state['mmigp_diagnostics'] = json.loads(eigen.read_text())
        write_status()
        (run / 'aggregate.json').write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
