"""Add measured affine errors over every native source voxel to a comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from fnit._transforms import load_lta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'reference', 'comparison'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    report = json.loads(Path(args.comparison).read_text())
    for name in ('forward', 'inverse'):
        candidate = load_lta(Path(args.candidate) / (name + '.lta'))
        reference = load_lta(Path(args.reference) / (name + '.lta'))
        if candidate.source.shape != reference.source.shape:
            raise ValueError('source shape mismatch')
        shape = candidate.source.shape
        matrix = (candidate.matrix - reference.matrix) @ candidate.source.affine
        total = square = maximum = count = 0
        for start in range(0, shape[2], 16):
            stop = min(start + 16, shape[2])
            grid = np.indices((*shape[:2], stop - start), dtype=np.float64)
            grid[2] += start
            errors = matrix[:3, :3] @ grid.reshape(3, -1) + matrix[:3, 3:4]
            distance = np.linalg.norm(errors, axis=0)
            total += float(distance.sum())
            square += float(np.dot(distance, distance))
            maximum = max(maximum, float(distance.max()))
            count += distance.size
        report['transforms'][name]['complete_source_grid_world_error_mm'] = {
            'voxels': count, 'mean': total / count, 'rmse': float(np.sqrt(square / count)),
            'max': maximum, 'gate_passed': maximum <= 1e-3}
    report['full_affine_grid_worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(args.comparison).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
