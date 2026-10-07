"""No-CNN CPU raw-coordinate interpolation diagnostic on actual live inputs."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from fnit.synthmorph import spatial
from first_layer_torch_probe_v1 import metrics


def raw_linear(volume, locations):
    """Independent eight-corner CPU trilinear interpolation experiment."""
    source_shape = volume.shape[2:]
    lower = [locations[:, axis].floor().clamp(0, size - 1) for axis, size in enumerate(source_shape)]
    upper = [(low + 1).clamp(0, size - 1) for low, size in zip(lower, source_shape)]
    clipped = [locations[:, axis].clamp(0, size - 1) for axis, size in enumerate(source_shape)]
    low_weight = [high - coordinate for high, coordinate in zip(upper, clipped)]
    high_weight = [1 - weight for weight in low_weight]
    indices = [[x.long() for x in lower], [x.long() for x in upper]]
    weights = [low_weight, high_weight]
    result = volume.new_zeros((volume.shape[0], volume.shape[1], *locations.shape[2:]))
    flat = volume.flatten(2)
    for corner in itertools.product((0, 1), repeat=3):
        i, j, k = [indices[choice][axis] for axis, choice in enumerate(corner)]
        index = ((i * source_shape[1] + j) * source_shape[2] + k).flatten(1)
        value = torch.gather(flat, 2, index[:, None].expand(-1, volume.shape[1], -1)).reshape_as(result)
        wi, wj, wk = [weights[choice][axis] for axis, choice in enumerate(corner)]
        result = result + ((wi * wj) * wk)[:, None] * value
    valid = torch.ones_like(locations[:, :1], dtype=torch.bool)
    for axis, size in enumerate(source_shape):
        valid &= (locations[:, axis:axis+1] >= 0) & (locations[:, axis:axis+1] <= size - 1)
    return torch.where(valid, result, torch.zeros_like(result))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('live', 'reference', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    live, reference = Path(args.live), Path(args.reference)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = {}
    with torch.inference_mode():
        for index in range(2):
            volume = torch.from_numpy(np.load(live / ('source_data_' + str(index) + '.npy')).astype(np.float32))[None, None]
            matrix = torch.from_numpy(np.load(live / ('source_to_network_' + str(index) + '.npy')).astype(np.float32))
            raw_path = reference / ('resampled_' + str(index) + '.npy')
            expected_raw = np.load(raw_path) if raw_path.exists() else None
            expected = np.load(live / ('network_input_' + str(index) + '.npy'))
            shape = tuple(expected.shape[1:4])
            coords = spatial.grid(shape, 'cpu')
            products = {'existing_einsum': spatial._dense_from_grid(matrix, coords)}
            products['contiguous_2d_mm'] = (matrix[:3, :3].contiguous() @ coords[0].flatten(1)).reshape(1, 3, *shape) + matrix[:3, 3][None, :, None, None, None] - coords
            direct = torch.stack([matrix[axis, 0] * coords[:, 0] + matrix[axis, 1] * coords[:, 1] + matrix[axis, 2] * coords[:, 2] + matrix[axis, 3] for axis in range(3)], dim=1)
            products['scalar_left_sum'] = direct - coords
            row = {}
            for name, shift in products.items():
                raw = raw_linear(volume, coords + shift)
                normalized = raw - raw.min()
                normalized = normalized / normalized.max()
                row[name] = {'normalized': metrics(normalized.permute(0, 2, 3, 4, 1).numpy(), expected)}
                if expected_raw is not None:
                    row[name]['raw'] = metrics(raw[0].permute(1, 2, 3, 0).numpy(), expected_raw)
            rows[str(index)] = row
    report = {'scope': __doc__, 'rows': rows, 'worker_sha256': digest(__file__),
              'spatial_sha256': digest(spatial.__file__),
              'reference_report_sha256': digest(reference / 'report.private.json'),
              'live_report_sha256': digest(live / 'report.private.json')}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
