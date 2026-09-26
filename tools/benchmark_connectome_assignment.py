"""Paired four-matrix benchmark against MRtrix tck2connectome.

The fixture is synthetic and tests the atlas/streamline aggregation stage only.
It does not validate diffusion reconstruction, tracking, or SIFT2 estimation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.assignment import build_connectomes


def _fixture(directory: Path, repeat: int):
    atlas = np.zeros((24, 12, 12), dtype=np.uint32)
    atlas[2:7, 3:9, 3:9] = 1
    atlas[10:15, 3:9, 3:9] = 2
    atlas[18:23, 3:9, 3:9] = 3
    affine = np.eye(4, dtype=np.float32)
    endpoints = np.asarray([
        [[4, 5, 5], [12, 5, 5]],
        [[12, 6, 5], [4, 6, 5]],
        [[4, 5, 6], [5, 5, 6]],
        [[12, 5, 6], [20, 5, 6]],
        [[7.5, 5, 5], [16.4, 5, 5]],
        [[4, 5, 5], [40, 5, 5]],
    ], dtype=np.float32)
    endpoints = np.tile(endpoints, (repeat, 1, 1))
    weights = np.tile(np.asarray([1., 2., 3., 4., 5., 6.], dtype=np.float32), repeat)
    lengths = np.tile(np.asarray([20., 40., 15., 35., 25., 60.], dtype=np.float32), repeat)
    fa = np.tile(np.asarray([0.2, 0.4, 0.8, 0.6, 0.5, 0.1], dtype=np.float32), repeat)
    atlas_path = directory / 'atlas.nii.gz'
    tck_path = directory / 'endpoints.tck'
    nib.save(nib.Nifti1Image(atlas, affine), atlas_path)
    header = (f'mrtrix tracks\ncount: {len(endpoints)}\ndatatype: Float32LE\n'
              'file: . 4096\nEND\n').encode()
    payload = np.empty((len(endpoints), 3, 3), dtype='<f4')
    payload[:, :2] = endpoints
    payload[:, 2] = np.nan
    with tck_path.open('wb') as stream:
        stream.write(header.ljust(4096, b'\0'))
        stream.write(payload.tobytes())
        stream.write(np.full(3, np.inf, dtype='<f4').tobytes())
    for name, array in [('weights', weights), ('lengths', lengths), ('fa', fa)]:
        np.savetxt(directory / f'{name}.txt', array, fmt='%.9g')
    return atlas, affine, endpoints, weights, lengths, fa, atlas_path, tck_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mrtrix', required=True, help='Path to tck2connectome binary')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeat', type=int, default=1,
                        help='Replicate six fixed streamlines to measure larger inputs')
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    atlas, affine, endpoints, weights, lengths, fa, atlas_path, tck_path = _fixture(args.output, args.repeat)
    reference = {}
    commands = {
        'count': [],
        'sift2_fbc': ['-tck_weights_in', str(args.output / 'weights.txt')],
        'mean_length': ['-tck_weights_in', str(args.output / 'weights.txt'),
                        '-scale_file', str(args.output / 'lengths.txt'), '-stat_edge', 'mean'],
        'mean_fa': ['-tck_weights_in', str(args.output / 'weights.txt'),
                    '-scale_file', str(args.output / 'fa.txt'), '-stat_edge', 'mean'],
    }
    reference_seconds = 0.
    for name, options in commands.items():
        output = args.output / f'mrtrix_{name}.csv'
        command = [args.mrtrix, '-quiet', '-symmetric', '-assignment_radial_search', '4',
                   *options, str(tck_path), str(atlas_path), str(output)]
        started = time.perf_counter()
        subprocess.run(command, check=True, capture_output=True, text=True)
        reference_seconds += time.perf_counter() - started
        reference[name] = np.atleast_2d(np.loadtxt(output, delimiter=','))

    device = torch.device(args.device)
    args_tensor = [torch.as_tensor(value, device=device) for value in
                   (endpoints, atlas.astype(np.int64), affine, weights, lengths, fa)]
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    candidate = build_connectomes(args_tensor[0], args_tensor[1], args_tensor[2],
                                   weights=args_tensor[3], lengths=args_tensor[4],
                                   fa=args_tensor[5], radius=4.)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    torch_seconds = time.perf_counter() - started
    metrics = {}
    for name, expected in reference.items():
        actual = candidate[name].detach().cpu().numpy()
        if actual.shape != expected.shape:
            raise AssertionError(f'{name}: shape {actual.shape} != {expected.shape}')
        np.savetxt(args.output / f'torch_{name}.csv', actual, delimiter=',', fmt='%.9g')
        diff = actual - expected
        metrics[name] = {
            'shape': list(actual.shape),
            'max_absolute_error': float(np.max(np.abs(diff))),
            'mean_absolute_error': float(np.mean(np.abs(diff))),
            'exact_elements': int(np.count_nonzero(actual == expected)),
            'elements': int(actual.size),
        }
    report = {
        'scope': 'synthetic atlas and fixed endpoint TCK; matrix assignment only',
        'streamlines': len(endpoints),
        'reference': subprocess.check_output([args.mrtrix, '-version'], text=True).strip(),
        'torch': torch.__version__,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'reference_four_commands_seconds': reference_seconds,
        'torch_four_matrices_seconds': torch_seconds,
        'metrics': metrics,
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(8, 10), constrained_layout=True)
    for row, name in enumerate(commands):
        original = reference[name]
        actual = candidate[name].detach().cpu().numpy()
        for col, (title, values) in enumerate(
            [('MRtrix', original), ('PyTorch', actual), ('absolute difference', np.abs(actual - original))]
        ):
            axes[row, col].imshow(values, cmap='magma')
            axes[row, col].set_title(f'{name}: {title}', fontsize=9)
            axes[row, col].set_xlabel('region')
            axes[row, col].set_ylabel('region')
    fig.savefig(args.output / 'comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
