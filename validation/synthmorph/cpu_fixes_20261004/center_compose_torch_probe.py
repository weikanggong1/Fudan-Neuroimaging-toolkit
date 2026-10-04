"""Compare center composition on identical official half-affines; no CNN."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    stages = np.load(args.input)
    labels = [key[:-5] for key in stages.files if key.endswith('_half')]
    rows = {}
    arrays = {}
    for label in labels:
        half, center, uncenter, expected = [torch.from_numpy(stages[label + '_' + name])
                                           for name in ('half', 'center', 'uncenter', 'composed')]
        square = torch.cat((half[:3], torch.tensor([[0., 0., 0., 1.]])), dim=0)
        inner = square @ center
        # Original compose discards the homogeneous row after each 3x4 transform.
        inner_square = torch.cat((inner[:3], torch.tensor([[0., 0., 0., 1.]])), dim=0)
        variants = {
            'current_batched_left_inverse': (torch.linalg.inv(center) @ half[None]) @ center,
            'batched_right_direct': uncenter @ (half[None] @ center),
            'unbatched_right_direct': uncenter @ (half @ center),
            'batched_right_square': uncenter @ (square[None] @ center),
            'unbatched_right_square': uncenter @ (square @ center),
            'unbatched_original_3x4_steps': uncenter @ inner_square,
        }
        rows[label] = {'inverse_center_max_abs': float((torch.linalg.inv(center) - uncenter).abs().max()),
                       'half_homogeneous_row': half[-1].tolist(), 'variants': {}}
        for name, actual in variants.items():
            actual = actual.squeeze(0)
            # Public affine representation always reconstructs the last row.
            actual = torch.cat((actual[:3], torch.tensor([[0., 0., 0., 1.]])), dim=0)
            rows[label]['variants'][name] = {'different_values': int(torch.count_nonzero(actual != expected)),
                                            'max_abs': float((actual.double() - expected.double()).abs().max())}
            arrays[label + '_' + name] = actual.numpy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / 'center_variants.npz', **arrays)
    report = {'scope': __doc__, 'rows': rows, 'input_sha256': digest(args.input),
              'worker_sha256': digest(__file__), 'torch_version': torch.__version__,
              'stages_sha256': digest(output / 'center_variants.npz')}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
