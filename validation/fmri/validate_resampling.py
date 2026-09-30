"""Compare linear/spline interpolation with one real BOLD and its exact warp.

All images and spatial arrays stay in --private-output. Only anonymous scalar
metrics, hashes and phase-bin summaries are written to --report-out.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, map_coordinates
import torch

from fnit.fmri.normalization import resample_world


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def agreement(candidate, reference):
    a, b = np.asarray(candidate, dtype=np.float64).ravel(), np.asarray(reference, dtype=np.float64).ravel()
    error = a - b
    return {'values': int(a.size), 'pearson_r': float(np.corrcoef(a, b)[0, 1]),
            'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt(np.mean(error ** 2))),
            'maximum_absolute_difference': float(np.abs(error).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'candidate', 'reference', 'mask', 'pull', 'matrix',
                 'private-output', 'report-out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--oracle-frames', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.private_output.mkdir(parents=True, exist_ok=True)
    if args.device.startswith('cuda'):
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(args.device)
    matrix = np.loadtxt(args.matrix)
    common = dict(pre_affine_pull_ras=args.pull, output_mask=args.mask,
                  batch_size=8, device=args.device)
    timing, outputs = {}, {}
    # Identical input, matrix, displacement field, mask and save policy.
    for interpolation in ('linear', 'spline'):
        output = args.private_output / f'{interpolation}_bold.nii.gz'
        started = time.perf_counter()
        outputs[interpolation] = resample_world(args.native, args.reference, matrix,
                                                output, interpolation=interpolation, **common)
        timing[interpolation + '_read_compute_save_seconds'] = time.perf_counter() - started
    source, target = nib.load(args.native), nib.load(args.reference)
    native = np.asarray(source.dataobj, dtype=np.float32)
    native_sd = native.std(axis=3)
    sd_path = args.private_output / 'native_sd.nii.gz'
    nib.save(nib.Nifti1Image(native_sd, source.affine), sd_path)
    sd_in_mni = resample_world(sd_path, args.reference, matrix,
                              args.private_output / 'native_sd_in_mni.nii.gz',
                              interpolation='linear', **common)
    expected_sd = np.asarray(nib.load(sd_in_mni).dataobj, dtype=np.float32)
    linear = np.asarray(nib.load(outputs['linear']).dataobj, dtype=np.float32)
    spline = np.asarray(nib.load(outputs['spline']).dataobj, dtype=np.float32)
    candidate = np.asarray(nib.load(args.candidate).dataobj, dtype=np.float32)
    if not all(values.shape == (*target.shape, source.shape[3]) and np.isfinite(values).all()
               for values in (linear, spline, candidate)):
        raise ValueError('nonfinite values or unexpected output shape')
    mask = np.asarray(nib.load(args.mask).dataobj) > 0
    world = target.affine[:3, :3] @ np.indices(target.shape, dtype=np.float64).reshape(3, -1)
    world += target.affine[:3, 3:4]
    world += np.asarray(nib.load(args.pull).dataobj, dtype=np.float64).reshape(-1, 3).T
    transform = np.linalg.inv(source.affine) @ matrix
    coords = (transform[:3, :3] @ world + transform[:3, 3:4]).reshape(3, *target.shape)
    valid = mask.copy()
    for axis in range(3):
        valid &= (coords[axis] >= 0) & (coords[axis] <= source.shape[axis] - 1)
    native_interior = binary_erosion(native_sd > 1e-4, iterations=2)
    interior = binary_erosion(mask, iterations=2) & (expected_sd > 1e-4)
    interior &= map_coordinates(native_interior.astype(np.float32), coords,
                                order=0, mode='constant') > .5
    for axis in range(3):
        interior &= (coords[axis] >= 4) & (coords[axis] <= source.shape[axis] - 5)
    fraction = coords - np.floor(coords)
    energy = np.sqrt(np.prod(fraction ** 2 + (1 - fraction) ** 2, axis=0))
    sd_maps = {'linear': linear.std(axis=3), 'spline': spline.std(axis=3)}
    ratios = {name: sd[interior] / expected_sd[interior] for name, sd in sd_maps.items()}
    phase = energy[interior]
    phase_metrics = {}
    edges = np.linspace(np.sqrt(1 / 8), 1, 10)
    for name, ratio in ratios.items():
        slope, intercept = np.polyfit(phase, ratio, 1)
        bins = []
        for low, high in zip(edges[:-1], edges[1:]):
            selected = (phase >= low) & (phase < high)
            bins.append({'weight_energy_low': float(low), 'weight_energy_high': float(high),
                         'voxels': int(selected.sum()),
                         'mean_relative_sd': float(ratio[selected].mean()) if selected.any() else None})
        phase_metrics[name] = {'median_relative_sd': float(np.median(ratio)),
                              'phase_pearson_r': float(np.corrcoef(phase, ratio)[0, 1]),
                              'phase_slope': float(slope), 'phase_intercept': float(intercept), 'bins': bins}
    oracle_frames = min(args.oracle_frames, source.shape[3])
    oracle = np.stack([map_coordinates(native[..., frame], coords, order=3, mode='grid-wrap')
                       for frame in range(oracle_frames)], axis=-1)
    oracle[~valid] = 0
    oracle_metrics = agreement(spline[valid, :oracle_frames], oracle[valid])
    # Spatial arrays are private; this file is used only to render template-space PNGs.
    np.savez_compressed(args.private_output / 'sd_maps.private.npz', mask=mask,
                        native_sd_in_mni=expected_sd, linear=sd_maps['linear'], spline=sd_maps['spline'])
    report = {'schema_version': 1, 'source_revision': args.source_revision,
              'data': {'kind': 'one real UK Biobank run', 'frames': source.shape[3],
                       'native_shape': list(source.shape), 'target_shape': list(spline.shape),
                       'phase_analysis_voxels': int(interior.sum())},
              'input_sha256': {name: sha256(getattr(args, name))
                               for name in ('native', 'candidate', 'reference', 'mask', 'pull', 'matrix')},
              'output_sha256': {name: sha256(path) for name, path in outputs.items()},
              'contracts': {'all_finite': True, 'outside_mask_max_abs':
                            {name: float(np.abs(values[~mask]).max())
                             for name, values in (('linear', linear), ('spline', spline))},
                            'repeated_spline_max_abs': float(np.abs(spline - candidate).max()),
                            'tr_seconds': float(nib.load(outputs['spline']).header.get_zooms()[3])},
              'phase_analysis': phase_metrics,
              'phase_slope_reduction_fraction': float(1 - phase_metrics['spline']['phase_slope'] /
                                                      phase_metrics['linear']['phase_slope']),
              'independent_scipy_oracle': {'frames': oracle_frames, 'order': 3, 'mode': 'grid-wrap',
                                          **oracle_metrics},
              'timing_seconds': timing,
              'memory': {'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(args.device)
                         if args.device.startswith('cuda') else 0},
              'definitions': {'relative_sd': 'SD(resampled 4D BOLD) / linear-resampled native temporal SD',
                              'weight_energy': 'sqrt(product over XYZ of (f^2 + (1-f)^2)), f=source voxel fraction',
                              'phase_slope': 'OLS slope of relative SD versus trilinear weight energy',
                              'interior': 'MNI mask eroded twice, source coordinates >=4 and <=size-5, native nonzero-SD mask eroded twice'},
              'limits': ['Same real BOLD and exact same composed warp; only interpolation changes.',
                         'Cubic interpolation reduces phase-dependent variance loss; it does not guarantee uniform noise variance.',
                         'Single sequential shared-GPU observation; timing includes gzip input/output.',
                         'SciPy is an independent mathematical spline oracle, not a complete FSL/UKB pipeline oracle.'],
              'privacy': 'Anonymous scalar metrics and hashes only; images and spatial arrays stay private.'}
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('contracts', 'phase_slope_reduction_fraction',
                                                'independent_scipy_oracle', 'timing_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
