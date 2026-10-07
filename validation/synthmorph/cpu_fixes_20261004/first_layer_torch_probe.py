"""Finite CPU convolution layout comparison on one exact original CLI input."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from fnit.synthmorph import models


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, reference):
    actual = np.asarray(actual).reshape(-1)
    reference = np.asarray(reference).reshape(-1)
    different = 0
    square = 0.
    reference_square = 0.
    maximum = 0.
    for start in range(0, actual.size, 1048576):
        a, b = actual[start:start+1048576], reference[start:start+1048576]
        different += int(np.count_nonzero(a != b))
        delta = a.astype(np.float64) - b.astype(np.float64)
        square += float(delta @ delta)
        reference_square += float(b.astype(np.float64) @ b.astype(np.float64))
        maximum = max(maximum, float(np.abs(delta).max()))
    return {'values': int(actual.size), 'different_values': different,
            'max_abs': maximum, 'rmse': (square / actual.size) ** .5,
            'relative_rmse': (square / reference_square) ** .5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'weights', 'reference', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--layer-index', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    with torch.inference_mode():
        input_array = np.load(args.input)
        input_tensor = torch.from_numpy(input_array).permute(0, 4, 1, 2, 3)
        detector = models.FeatureDetector(args.weights)
        layer = detector.layers[args.layer_index]
        before = layer.weight.clone()
        reference = Path(args.reference)
        expected = {name: np.load(reference / (name + '.npy'), mmap_mode='r')
                    for name in ('convolution', 'activation', 'pool')}
        rows = {}
        for input_last, weight_last in ((False, False), (True, False), (False, True), (True, True)):
            image = input_tensor.contiguous(memory_format=torch.channels_last_3d if input_last else torch.contiguous_format)
            weight = layer.weight.contiguous(memory_format=torch.channels_last_3d if weight_last else torch.contiguous_format)
            convolution = F.conv3d(image, weight, layer.bias, padding=1)
            activation = F.leaky_relu(convolution, .2)
            pooled = F.max_pool3d(activation, 2)
            key = 'input_last_' + str(input_last) + '_weight_last_' + str(weight_last)
            rows[key] = {name: metrics(tensor.permute(0, 2, 3, 4, 1).numpy(), expected[name])
                         for name, tensor in [('convolution', convolution), ('activation', activation), ('pool', pooled)]}
            del convolution, activation, pooled
        reference_conv = torch.from_numpy(np.array(expected['convolution'])).permute(0, 4, 1, 2, 3)
        isolated_activation = F.leaky_relu(reference_conv, .2)
        rows['activation_on_identical_original_convolution'] = metrics(
            isolated_activation.permute(0, 2, 3, 4, 1).numpy(), expected['activation'])
        if not torch.equal(before, layer.weight):
            raise RuntimeError('experiment mutated detector parameters')
        report = {'scope': __doc__, 'rows': rows, 'input_sha256': digest(args.input),
                  'weights_sha256': digest(args.weights), 'models_sha256': digest(models.__file__),
                  'worker_sha256': digest(__file__), 'reference_report_sha256': digest(reference / 'report.private.json'),
                  'parameters_unchanged': True, 'torch_version': torch.__version__,
                  'layer_index': args.layer_index}
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
