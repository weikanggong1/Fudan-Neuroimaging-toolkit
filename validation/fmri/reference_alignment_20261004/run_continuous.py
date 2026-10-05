"""真实原始 BOLD+T1w 到 volume/CIFTI；复用自身重建/球面，不计冷 recon-all。"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import nibabel as nib
import torch

from fnit.fmri import fMRISurface_pipeline, locate_bids_inputs
from fnit.fmri.derivatives import fmri_derivative_paths


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def bound(entry):
    p = Path(entry['path']).resolve(strict=True)
    if digest(p) != entry['sha256']:
        raise ValueError('bound file SHA mismatch')
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    source = Path(manifest['source_root']).resolve()
    config = manifest['configuration']
    imported = Path(inspect.getfile(fMRISurface_pipeline)).resolve()
    imported.relative_to(source / 'src')
    guarded = {role: bound(entry) for role, entry in manifest['input_files'].items()}
    helper_path = bound(manifest['validation_helper'])
    spec = importlib.util.spec_from_file_location('fnit_frozen_validation_helpers', helper_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    args.output.mkdir(parents=True, exist_ok=False)
    before = helpers.source_hashes(source)
    source_manifest = bound(manifest['source_manifest'])
    if before != json.loads(source_manifest.read_text()):
        raise ValueError('runtime source differs from the frozen source manifest')
    helpers.write(args.output / 'source.private.json', before)
    monitor = helpers.Monitor(args.output / 'gpu_load.private.jsonl')
    started = None
    report = {'case_id': manifest['case_id'], 'variant': manifest['variant'], 'status': 'initializing',
              'baseline_commit': manifest['baseline_commit'],
              'source_sha256': digest(args.output / 'source.private.json'),
              'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'driver_sha256': digest(__file__),
              'scope': 'Raw paired T1w and all BOLD frames through automatic volume and existing own reconstruction/registered spheres to saved MNI/GIFTI/CIFTI; excludes cold recon-all and MSM estimation.',
              'shared_gpu': True, 'failure': None}
    exit_code = 1
    try:
        inputs = locate_bids_inputs(config['bids_root'], subject=config['subject'], session=config.get('session'))
        if digest(inputs.bold) != manifest['input_files']['bold']['sha256'] or len(inputs.t1w_images) != 1:
            raise ValueError('wrong raw BOLD/T1w identity')
        frames = nib.load(inputs.bold).shape[3]
        if frames != 180 or inputs.tr != manifest['tr_seconds']:
            raise ValueError('requires bound complete180 frames and TR')
        paths = fmri_derivative_paths(inputs, inputs.t1w_images[0], config['derivatives_root'], signal='preproc')
        if paths.root.exists():
            raise FileExistsError('continuous comparison requires a fresh derivatives directory')
        device = config['device']
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('GPU required; no CPU fallback')
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.init()
        torch.cuda.set_per_process_memory_fraction(min(1., 20e9 / torch.cuda.get_device_properties(device).total_memory), device)
        torch.cuda.reset_peak_memory_stats(device)
        report.update(frames=frames, tr_seconds=inputs.tr,
                      raw_hashes={k: manifest['input_files'][k]['sha256'] for k in ('t1w', 'bold')},
                      pre_api_load_average=list(os.getloadavg()))
        monitor.thread.start()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = fMRISurface_pipeline(**config)
        torch.cuda.synchronize(device)
        api_seconds = time.perf_counter() - started
        if not result.volume_executed:
            raise ValueError('raw continuous chain did not automatically execute volume')
        checked = {key: helpers.image_check(path, frames, inputs.tr) for key, path in
                   {'preproc_mni': paths.preproc_mni, 'dtseries': result.dtseries}.items()}
        files = {key: {'path': str(path), 'sha256': digest(path)} for key, path in
                 {'preproc_mni': paths.preproc_mni, 'dtseries': result.dtseries,
                  'volume_metadata': paths.preproc_mni.with_name(paths.preproc_mni.name[:-7]+'.json'),
                  'surface_metadata': result.metadata, 'bold_reference': paths.bold_reference}.items()}
        helpers.write(args.output / 'files.private.json', files)
        if any(digest(path) != manifest['input_files'][role]['sha256'] for role, path in guarded.items()):
            raise ValueError('input file mutated during measurement')
        if before != helpers.source_hashes(source) or args.manifest.read_bytes() != manifest_bytes:
            raise ValueError('source/manifest mutated during measurement')
        report.update(status='scientific_complete', api_seconds=api_seconds,
                      api_plus_validation_seconds=time.perf_counter()-started,
                      phase_seconds=result.timing_seconds, output_checks=checked,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      input_guards_equal=True, source_guards_equal=True)
        exit_code = 0
    except Exception as error:
        report.update(status='failed', failure={'type':type(error).__name__, 'message':str(error)},
                      failed_attempt_seconds=time.perf_counter()-started if started else None)
        traceback.print_exc()
    finally:
        monitor.stop.set()
        if monitor.thread.is_alive():
            monitor.thread.join(10)
        report.update(sampled_owned_simultaneous_peak_bytes=monitor.peak,
                      gpu_sampling_errors=monitor.errors,
                      post_api_load_average=list(os.getloadavg()))
        helpers.write(args.output / 'report.public.json', report)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
