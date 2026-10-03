"""在真实既有被试的私有副本上比较父进程缓存与双侧 register 阶段。"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import nibabel as nib
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--variant', choices=('before', 'after'), required=True)
    args = parser.parse_args()
    if 'PYTORCH_NO_CUDA_MEMORY_CACHING' in os.environ:
        raise ValueError('this comparison requires caching enabled at fresh process entry')
    from fnit.synthseg_parc import SynthSeg
    from fnit.recon_all.hemisphere_parallel import run_hemisphere_group
    import inspect
    imported_module = Path(inspect.getfile(run_hemisphere_group)).resolve()
    imported_module.relative_to((args.source_root / 'src').resolve())
    driver_sha_before = sha256(__file__)
    config = json.loads(args.config.read_text())
    original = Path(config['subject'])
    weights, assets = Path(config['weights']), Path(config['assets'])
    args.output.mkdir(parents=True, exist_ok=False)
    source_files = sorted((args.source_root / 'src/fnit').rglob('*.py'))
    source_before = {p.relative_to(args.source_root).as_posix(): sha256(p) for p in source_files}
    guard_paths = {p.relative_to(original).as_posix(): p for folder in ('mri', 'surf', 'label', 'stats')
                   for p in sorted((original / folder).rglob('*')) if p.is_file()}
    guard_paths.update({'assets/' + str(p.relative_to(assets)): p for p in
                        [assets / 'FreeSurferColorLUT.txt', *map(Path, config['registration_atlases'].values())]})
    guard_paths.update({'weights/' + p.name: p for p in sorted(weights.glob('*')) if p.is_file()})
    guard_paths['native/mrisp_paint'] = Path(config['paint_binary'])
    before = {name: sha256(path) for name, path in guard_paths.items()}
    preparation_started = time.perf_counter()
    subject = args.output / 'subject'
    subject.mkdir()
    for folder in ('mri', 'surf', 'label', 'stats'):
        shutil.copytree(original / folder, subject / folder)
    (subject / 'scripts').mkdir()
    preparation_seconds = time.perf_counter() - preparation_started
    torch.set_num_threads(config['threads'])
    torch.cuda.set_device(config['device'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    state = {'status': 'running', 'variant': args.variant, 'subject': config['public_subject'],
             'dataset': 'ds001226 v5.0.1', 'allocator_policy': 'enabled',
             'precision': {'matmul_tf32': True, 'cudnn_tf32_default': True, 'fp16_bf16': False},
             'copy_seconds_excluded_from_group': preparation_seconds,
             'scope': 'Real SynthSeg warmup, followed by register hemisphere group on an existing complete reconstruction copy; not raw-T1 reconstruction or whole surface pipeline.',
             'timing_scope': 'group includes private copies, optional idle-cache release, fresh exec/bootstrap, both register/avg-curv workers, publication and cleanup; warmup and guards separate',
             'source_hashes_before': source_before, 'input_sha256_before': before}
    state.update(driver_sha256_before=driver_sha_before,
                 imported_hemisphere_module=imported_module.relative_to(args.source_root.resolve()).as_posix(),
                 imported_hemisphere_module_sha256=sha256(imported_module))
    write(args.output / 'report.public.json', state)
    try:
        warm_started = time.perf_counter()
        model = SynthSeg(weights=weights, device=config['device'], threads=config['threads'], cudnn_tf32=False)
        result = model(subject / 'mri/orig.mgz', keep_geometry=True, color_lut=assets / 'FreeSurferColorLUT.txt')
        labels = np.asarray(result.segmentation.data)
        state['warmup'] = {'seconds_including_loading': time.perf_counter() - warm_started,
                           'segmentation_array_sha256': hashlib.sha256(labels.tobytes(order='C')).hexdigest(),
                           'shape': list(labels.shape), 'dtype': str(labels.dtype),
                           'actual_forward_precision': model.segmenter.precision}
        del result, labels, model
        gc.collect()
        # A small live tensor checks the allocator contract; it is not MRI benchmark input.
        live_tensor = torch.arange(1024, device=config['device'], dtype=torch.float32)
        expected_live = live_tensor.cpu().numpy().copy()
        torch.cuda.synchronize(config['device'])
        state['parent_before_group'] = {'allocated_bytes': torch.cuda.memory_allocated(config['device']),
                                        'reserved_bytes': torch.cuda.memory_reserved(config['device'])}
        group = run_hemisphere_group(subject, 'register', device=config['device'],
                                    threads=config['threads'], workers=2,
                                    kwargs={'assets': str(assets),
                                            'binaries': {'paint': config['paint_binary']},
                                            'registration_atlases': config['registration_atlases']})
        write(args.output / 'group.private.json', group)
        torch.cuda.synchronize(config['device'])
        if not np.array_equal(live_tensor.cpu().numpy(), expected_live):
            raise AssertionError('live tensor changed during group/cache release')
        output_checks = {}
        for relative in group['published']:
            if relative.startswith('surf/'):
                path = subject / relative
                entry = {'sha256': sha256(path), 'size_bytes': path.stat().st_size}
                if path.name.endswith('.sphere.reg'):
                    points, faces = nib.freesurfer.read_geometry(str(path))
                    if not np.isfinite(points).all():
                        raise AssertionError('nonfinite registered sphere')
                    entry.update(vertices=int(len(points)), faces=int(len(faces)), all_finite=True)
                output_checks[relative] = entry
        after = {name: sha256(path) for name, path in guard_paths.items()}
        source_after = {p.relative_to(args.source_root).as_posix(): sha256(p) for p in source_files}
        if before != after or source_before != source_after or driver_sha_before != sha256(__file__):
            raise AssertionError('original input or source changed during comparison')
        peak = group['device_process_tree']['peak_tree_total_bytes']
        state.update(status='complete', group_wall_seconds=group['group_wall_seconds'],
                     worker_span_seconds=group['worker_span_seconds'],
                     worker_sum_seconds=group['worker_sum_seconds'], overlap_seconds=group['overlap_seconds'],
                     parent_idle_cuda_cache=group.get('parent_idle_cuda_cache'),
                     simultaneous_job_tree_peak_bytes=peak,
                     simultaneous_job_tree_peak_decimal_gb=None if peak is None else peak / 1e9,
                     simultaneous_job_tree_peak_gib=None if peak is None else peak / 1024**3,
                     memory_sampling_interval_seconds=group['device_process_tree']['sampling_interval_seconds'],
                     memory_sample_count=len(group['device_process_tree']['samples']),
                     memory_sample_errors=group['device_process_tree']['failed_samples'],
                     input_sha256_after=after, inputs_unchanged=True, source_unchanged=True,
                     driver_sha256_after=sha256(__file__), driver_unchanged=True,
                     live_tensor_unchanged=True, outputs=output_checks,
                     parent_after_group={'allocated_bytes': torch.cuda.memory_allocated(config['device']),
                                         'reserved_bytes': torch.cuda.memory_reserved(config['device'])})
    except BaseException as error:
        (args.output / 'failure.private.txt').write_text(traceback.format_exc())
        state.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        write(args.output / 'report.public.json', state)


if __name__ == '__main__':
    main()
