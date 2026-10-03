"""Actual-image FAST/VBM calls for the common cold-process queue.

Original programs run separately through the root reference runner. The
profile mode crops a real T1 for diagnosis; only full-image calls count as
imaging benchmarks. This file is never imported by production FNIT.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


FIELDS = {
    'pve_0': 'pve_csf', 'pve_1': 'pve_gm', 'pve_2': 'pve_wm',
    'seg': 'hard_segmentation', 'pveseg': 'pve_segmentation',
    'mixeltype': 'mixel_type', 'bias': 'bias_field', 'restore': 'restored',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('fast', 'profile', 'vbm'), default='fast')
    parser.add_argument('--image', required=True)
    parser.add_argument('--mask')
    parser.add_argument('--template')
    parser.add_argument('--reference-mask')
    parser.add_argument('--execution', choices=('tensor', 'fsl'), default='fsl')
    parser.add_argument('--backend', choices=('fnirt', 'synthmorph'), default='fnirt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--config', default='{}')
    parser.add_argument('--patch-size', nargs=3, type=int, default=(24, 26, 28))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import nibabel as nib
    import torch
    import fnit
    from fnit.fast import TorchFAST
    from fnit.fast import algorithm
    from fnit.fast import _fsl_scan as scan

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    report = {'mode': args.mode, 'device': args.device, 'execution': args.execution,
              'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'config': json.loads(args.config), 'stages': {},
              'software': {'torch': torch.__version__, 'numpy': np.__version__,
                           'nibabel': nib.__version__},
              'fnit_import': fnit.__file__, 'input_sha256': sha256(args.image),
              'source_sha256': {}}
    for name in ('algorithm.py', 'pipeline.py', '_fsl_scan.py', '_fsl_cpu.py'):
        path = Path(algorithm.__file__).with_name(name)
        if path.exists():
            report['source_sha256'][name] = sha256(path)
    image = args.image
    mask = args.mask
    if args.mask:
        report['mask_sha256'] = sha256(args.mask)
    if args.mode == 'profile':
        image = nib.load(args.image)
        middle = np.array(image.shape[:3]) // 2
        slices = tuple(slice(int(center - size // 2), int(center - size // 2 + size))
                       for center, size in zip(middle, args.patch_size))
        image = image.slicer[slices]
        if args.mask:
            mask = nib.load(args.mask).slicer[slices]
        report['diagnostic_crop'] = {'start': [s.start for s in slices],
                                     'stop': [s.stop for s in slices],
                                     'shape': list(image.shape),
                                     'is_full_image_benchmark': False}
        nib.save(image, args.output_dir / 'real_patch.nii.gz')
        if mask is not None:
            nib.save(mask, args.output_dir / 'real_patch_mask.nii.gz')

        def capture(owner, name):
            original = getattr(owner, name)
            key = owner.__name__ + '.' + name

            def measured(*values, **options):
                started = time.perf_counter()
                result = original(*values, **options)
                if str(args.device).startswith('cuda'):
                    torch.cuda.synchronize(args.device)
                elapsed = time.perf_counter() - started
                entry = report['stages'].setdefault(key, {'calls': 0, 'seconds': 0.0})
                entry['calls'] += 1
                entry['seconds'] += elapsed
                return result

            setattr(owner, name, measured)

        for name in ('_fsl_moments', '_fsl_initial_probabilities', '_fsl_bias',
                     '_fsl_mixel_probabilities', '_fsl_partial_volumes'):
            capture(algorithm, name)
        for name in ('schedule', 'tanaka', 'icm', 'blur'):
            capture(scan, name)
        capture(scan.GlibcRandom, 'raw')
        report['stage_timing_scope'] = 'Nested stages overlap; real cropped-image diagnosis only.'

    if args.mode == 'vbm':
        from fnit.fast_vbm import FastVBM

        config = dict(report['config'])
        # The frozen main predates this optional parameter; its default tensor
        # path remains benchmarkable through the same worker.
        if args.execution == 'fsl' or 'fast_execution' in config:
            config.setdefault('fast_execution', args.execution)
        model = FastVBM(device=args.device, threads=args.threads,
                        registration_backend=args.backend, **config)
        started = time.perf_counter()
        result = model.run(image=image, template=args.template,
                           brain_mask=mask, reference_mask=args.reference_mask,
                           output_dir=args.output_dir / 'artifacts')
        report['api_with_save_seconds'] = time.perf_counter() - started
        report['pipeline_report'] = result.report()
    else:
        started = time.perf_counter()
        model = TorchFAST(device=args.device, threads=args.threads,
                          execution=args.execution, **report['config'])
        result = model(image, mask=mask)
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize(args.device)
        report['loaded_api_seconds'] = time.perf_counter() - started
        saved = time.perf_counter()
        for suffix, name in FIELDS.items():
            volume = getattr(result, name)
            path = args.output_dir / ('fast_' + suffix + '.nii.gz')
            nib.save(volume, path)
        report['save_seconds'] = time.perf_counter() - saved
        report['tissue_means'] = list(result.tissue_means)
        report['tissue_variances'] = list(result.tissue_variances)
        report['outputs'] = {suffix: {'sha256': sha256(args.output_dir / ('fast_' + suffix + '.nii.gz')),
                                     'shape': list(getattr(result, name).shape),
                                     'dtype': str(getattr(result, name).get_data_dtype()),
                                     'finite': bool(np.isfinite(np.asarray(getattr(result, name).dataobj)).all())}
                             for suffix, name in FIELDS.items()}
    report['cuda_initialized'] = torch.cuda.is_initialized()
    (args.output_dir / 'api_record.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'mode': args.mode, 'finished': True,
                      'api_seconds': report.get('loaded_api_seconds', report.get('api_with_save_seconds'))}))


if __name__ == '__main__':
    main()
