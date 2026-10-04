"""Isolated official smoothing on the saved real first-state reference mesh."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sigma', type=float, default=3.)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    sys.path.insert(0, '/public/software/apps/Freesurfer/8.2.0-1/python/lib/python3.8/site-packages/samseg/gems')
    import gemsbindings as gems
    data = np.load(args.shared)
    reference = np.asfortranarray(data['reference'].astype(np.float64))
    grouped = np.zeros_like(data['alphas'])
    for label, group in enumerate(data['classes']):
        grouped[:, group] += data['raw_alphas'][:, label]
    collection = gems.KvlMeshCollection()
    collection.read(str(args.mesh))
    collection.set_positions(reference, [reference.copy()])
    collection.reference_mesh.alphas = np.asfortranarray(grouped)
    assert np.array_equal(collection.reference_mesh.points, reference)
    assert np.array_equal(collection.reference_mesh.alphas, grouped)
    started = time.perf_counter()
    collection.smooth(args.sigma)
    elapsed = time.perf_counter() - started
    actual = np.asarray(collection.reference_mesh.alphas)
    np.save(args.output / 'native_smoothed_alphas.npy', actual)
    expected = data['alphas']
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    report = {'schema': 'fnit.gems.native-alpha-first-state.v1',
              'scope': 'Single isolated native sigma smoothing on the same saved FP32 reference coordinates promoted to FP64; does not fit a mesh or complete a recipe',
              'shared_input_sha256': hashlib.sha256(args.shared.read_bytes()).hexdigest(),
              'mesh_sha256': hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
              'native_binding_sha256': hashlib.sha256(Path(gems.__file__).read_bytes()).hexdigest(),
              'program_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sigma': args.sigma, 'affinity': sorted(os.sched_getaffinity(0)),
              'single_smoothing_observation_seconds': elapsed,
              'actual_dtype': str(actual.dtype), 'fnit_dtype': str(expected.dtype),
              'shape': list(actual.shape), 'exact': bool(np.array_equal(actual, expected)),
              'different_values': int(np.count_nonzero(actual != expected)),
              'maximum_absolute_error': float(np.max(np.abs(difference))),
              'rmse': float(np.sqrt(np.mean(difference * difference))),
              'relative_l2': float(np.linalg.norm(difference) / np.linalg.norm(actual)),
              'max_row_sum_error_native': float(np.max(np.abs(actual.sum(1) - 1))),
              'max_row_sum_error_fnit': float(np.max(np.abs(expected.sum(1) - 1))),
              'output_sha256': hashlib.sha256((args.output / 'native_smoothed_alphas.npy').read_bytes()).hexdigest()}
    (args.output / 'report.public.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
