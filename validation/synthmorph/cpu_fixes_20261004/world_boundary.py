"""Real acquired frames: declared finite-FOV periodic policy and three APIs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
import torch

from fnit._world_resampling import WorldTransformChain, resample_world_image
from fnit.applywarp import TorchApplyWarp
from fnit.synthmorph import apply_transform


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    source = nib.load(args.source)
    if source.ndim != 4 or source.shape[-1] != 2:
        raise ValueError('this diagnostic requires the two acquired DWI frames')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    data = np.asarray(source.dataobj, dtype=np.float32)
    reference = nib.Nifti1Image(np.zeros(source.shape[:3], np.float32), source.affine,
                               header=source.header.copy())
    voxel_pull = np.eye(4)
    voxel_pull[:3, 3] = [-0.25, 0.25, 0.25]
    world_pull = source.affine @ voxel_pull @ np.linalg.inv(source.affine)
    chain = WorldTransformChain(reference, world_pull)
    grid = np.indices(source.shape[:3], dtype=np.float64).reshape(3, -1)
    points = voxel_pull[:3, :3] @ grid + voxel_pull[:3, 3:4]
    # Match the documented <=1e-6 roundoff clamp and finite-FOV mask. The
    # independent circular interpolator is deliberately queried out of FOV;
    # that part of the oracle is then zeroed by the declared public policy.
    for axis, size in enumerate(source.shape[:3]):
        close = (points[axis] >= -1e-6) & (points[axis] <= size - 1 + 1e-6)
        points[axis, close] = np.clip(points[axis, close], 0, size - 1)
    inside = ((points >= 0) & (points <= np.array(source.shape[:3])[:, None] - 1)).all(0)
    report = {'scope': 'full-grid real two-frame DWI finite-FOV periodic-policy control; no CNN',
              'source_sha256': digest(args.source), 'worker_sha256': digest(__file__),
              'shape': list(source.shape), 'outside_query_voxels': int(np.count_nonzero(~inside)),
              'voxel_pull': voxel_pull.tolist(), 'rows': []}
    import fnit._world_resampling as helper
    import fnit.synthmorph.pipeline as pipeline
    report['source_sha256_modules'] = {'world_resampling': digest(helper.__file__),
                                     'synthmorph_pipeline': digest(pipeline.__file__)}
    for method, order in [('nearest', 0), ('linear', 1), ('spline', 3)]:
        oracle = []
        for frame in range(2):
            sampled = map_coordinates(data[..., frame], points, order=order, mode='grid-wrap')
            sampled[~inside] = 0
            oracle.append(sampled.reshape(source.shape[:3]))
        oracle = np.stack(oracle, -1)
        baseline = None
        for entry in ('synthmorph', 'applywarp', 'shared'):
            started = time.perf_counter()
            if entry == 'synthmorph':
                result = apply_transform(source, chain, method=method, boundary='periodic',
                                         device='cpu', frame_chunk_size=1)
            elif entry == 'applywarp':
                result = TorchApplyWarp(device='cpu').apply_world(
                    source, chain, interpolation=method, boundary='periodic', batch_size=1)
            else:
                result = resample_world_image(source, reference, world_pull,
                                              interpolation=method, boundary='periodic',
                                              device='cpu', batch_size=1)
            elapsed = time.perf_counter() - started
            actual = np.asarray(result.dataobj)
            delta = actual.astype(np.float64) - oracle
            span = np.percentile(oracle, 99) - np.percentile(oracle, 1)
            rmse = float(np.sqrt(np.mean(delta * delta)))
            outside = actual.reshape(-1, 2)[~inside]
            row = {'method': method, 'entry': entry, 'api_seconds': elapsed,
                   'rmse': rmse, 'nrmse_p99_minus_p1': None if span == 0 else rmse / float(span),
                   'max_abs': float(np.abs(delta).max()),
                   'different_values': int(np.count_nonzero(delta)),
                   'outside_nonzero': int(np.count_nonzero(outside)),
                   'same_shared_array': True if baseline is None else bool(np.array_equal(actual, baseline)),
                   'same_shared_header': True if baseline is None else result.header.binaryblock == base_header,
                   'tr_equal': bool(result.header.get_zooms()[3] == source.header.get_zooms()[3])}
            row['gate_passed'] = (row['outside_nonzero'] == 0 and row['same_shared_array']
                                  and row['same_shared_header'] and row['tr_equal']
                                  and (row['different_values'] == 0 if order == 0 else
                                       row['nrmse_p99_minus_p1'] is not None and row['nrmse_p99_minus_p1'] <= 1e-3))
            if baseline is None:
                baseline = actual.copy()
                base_header = result.header.binaryblock
            report['rows'].append(row)
        (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    report['all_gates_passed'] = all(row['gate_passed'] for row in report['rows'])
    report['status'] = 'complete'
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_gates_passed']:
        raise RuntimeError('real-input periodic boundary contract failed')


if __name__ == '__main__':
    main()
