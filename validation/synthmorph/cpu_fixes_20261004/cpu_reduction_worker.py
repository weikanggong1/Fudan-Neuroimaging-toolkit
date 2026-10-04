"""Isolate feature reduction layout while preserving both convolution routes."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import torch
from fnit.synthmorph import models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', required=True)
    p.add_argument('--policy', choices=('channels-last', 'reference-shapes'), default='channels-last')
    args, remaining = p.parse_known_args()
    if '--device' in remaining and remaining[remaining.index('--device') + 1] != 'cpu':
        raise ValueError('this isolated diagnostic only supports CPU')
    original = models.barycenter

    def barycenter(features, full_shape):
        if args.policy == 'channels-last':
            return original(features.contiguous(memory_format=torch.channels_last_3d), full_shape)
        # The reference has two distinct reduction layouts: feature masses
        # for confidence weights in NHWDC, and centered XYZ moments after a
        # transpose to NCDHW with a final three-coordinate dimension.
        mass = features.permute(0, 2, 3, 4, 1).contiguous().sum((1, 2, 3))
        coordinates = [
            (torch.arange(size, dtype=features.dtype, device=features.device)
             - (size - 1) / 2) / size for size in features.shape[2:]
        ]
        grid = torch.stack(torch.meshgrid(*coordinates, indexing='ij'), -1)
        values = features.contiguous().unsqueeze(-1)
        denominator = values.sum((2, 3, 4))
        moment = (values * grid).sum((2, 3, 4))
        centers = torch.where(denominator != 0, moment / denominator, 0)
        centers *= torch.as_tensor(full_shape, dtype=features.dtype, device=features.device)
        return centers, mass

    models.barycenter = barycenter
    sys.argv = [args.worker, *remaining]
    runpy.run_path(args.worker, run_name='__main__')
    output = Path(remaining[remaining.index('--output') + 1])
    record = {'scope': 'CPU feature reduction layout only; unchanged CNNs and transform/sampler arithmetic',
              'policy': args.policy,
              'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'worker_sha256': hashlib.sha256(Path(args.worker).read_bytes()).hexdigest()}
    (output / 'reduction.private.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
