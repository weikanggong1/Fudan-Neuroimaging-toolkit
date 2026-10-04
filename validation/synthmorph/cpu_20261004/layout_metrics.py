"""Bounded-memory comparison of saved real-input layout diagnostic arrays."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(left, right):
    a = np.load(left, mmap_mode='r').reshape(-1)
    b = np.load(right, mmap_mode='r').reshape(-1)
    maximum = square = 0.0
    mismatch = 0
    close = True
    for start in range(0, a.size, 1_000_000):
        x = np.asarray(a[start:start + 1_000_000], dtype=np.float64)
        y = np.asarray(b[start:start + 1_000_000], dtype=np.float64)
        delta = x - y
        maximum = max(maximum, float(np.abs(delta).max()))
        square += float(np.dot(delta, delta))
        mismatch += int(np.count_nonzero(delta))
        close &= bool(np.allclose(x, y, rtol=1e-5, atol=1e-4))
    return {'max_abs': maximum, 'rmse': float(np.sqrt(square/a.size)),
            'different_values': mismatch, 'allclose_rtol1e_minus5_atol1e_minus4': close}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for mode in ('affine', 'joint'):
        folder = root / ('layout_' + mode) / 'artifacts'
        report = json.loads((folder/'report.json').read_text())
        runs = report['runs']
        times = {policy: [r['api_seconds'] for r in runs if r['policy'] == policy]
                 for policy in ('contiguous', 'channels_last_3d')}
        arrays = {name: compare(folder/('0_'+name+'.npy'), folder/('1_'+name+'.npy'))
                  for name in ('forward', 'inverse')}
        rows.append({'mode': mode, 'times': times, 'arrays': arrays,
                     'units': 'world RAS displacement mm' if mode == 'joint'
                              else 'world affine matrix entries, not displacement eligibility',
                     'production_adopted': False})
    Path(args.output).write_text(json.dumps({'scope': 'CPU-only diagnostic; no GPU layout change; no production adoption', 'rows': rows}, indent=2)+'\n')


if __name__ == '__main__':
    main()
