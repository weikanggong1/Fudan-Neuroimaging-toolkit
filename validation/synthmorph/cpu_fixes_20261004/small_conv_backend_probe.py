"""Finite CPU backend comparison at the first differing small affine convolution."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from fnit.synthmorph import models

try:
    from first_layer_torch_probe import metrics
except ModuleNotFoundError:
    from conv_layer_torch_probe_v2 import metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'weights', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--layer-index', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    reference = Path(args.reference)
    prefix = 'layer_' + str(args.layer_index) + '_'
    input_path = reference / (prefix + 'input.npy')
    expected_path = reference / (prefix + 'convolution.npy')
    detector = models.FeatureDetector(args.weights)
    layer = detector.layers[args.layer_index]
    before = layer.weight.clone()
    enabled_before = torch.backends.mkldnn.enabled
    rows = {}
    with torch.inference_mode():
        image = torch.from_numpy(np.load(input_path)).permute(0, 4, 1, 2, 3)
        expected = np.load(expected_path)
        contiguous = image.contiguous()
        last = image.contiguous(memory_format=torch.channels_last_3d)
        weight = layer.weight
        weight_last = weight.contiguous(memory_format=torch.channels_last_3d)
        common = ([1, 1, 1], [1, 1, 1], [1, 1, 1], 1)
        selections = {name: str(torch._C._select_conv_backend(
            x, w, layer.bias, common[1], common[0], common[2], False, [0, 0, 0], 1))
            for name, x, w in [('contiguous', contiguous, weight), ('channels_last', last, weight_last)]}
        operations = [
            ('established_contiguous', lambda: F.conv3d(contiguous, weight, layer.bias, padding=1)),
            ('channels_last_default', lambda: F.conv3d(last, weight_last, layer.bias, padding=1)),
            ('channels_last_pointwise', lambda: torch.ops.mkldnn._convolution_pointwise(
                last, weight_last, layer.bias, *common, 'none', [], None)),
            ('channels_last_direct_mkldnn', lambda: torch.mkldnn_convolution(
                last, weight_last, layer.bias, *common)),
            ('mkldnn_input', lambda: F.conv3d(contiguous.to_mkldnn(), weight, layer.bias, padding=1).to_dense())]
        for name, operation in operations:
            print('BEGIN_OPERATOR ' + name, flush=True)
            start = time.monotonic()
            try:
                with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
                    result = operation()
                if args.layer_index == 8:
                    result = F.relu(result)
                rows[name] = {'operator_seconds': time.monotonic() - start,
                              'comparison': metrics(result.permute(0, 2, 3, 4, 1).numpy(), expected)}
            except Exception as error:
                rows[name] = {'error_type': type(error).__name__, 'error': str(error)}
            print('END_OPERATOR ' + name, flush=True)
    if not torch.equal(before, layer.weight) or enabled_before != torch.backends.mkldnn.enabled:
        raise RuntimeError('probe changed caller parameters or backend policy')
    report = {'scope': __doc__, 'layer_index': args.layer_index, 'rows': rows,
              'backend_selection': selections, 'parameters_and_backend_policy_unchanged': True,
              'torch_version': torch.__version__, 'input_sha256': digest(input_path),
              'expected_sha256': digest(expected_path), 'weights_sha256': digest(args.weights),
              'models_sha256': digest(models.__file__), 'worker_sha256': digest(__file__),
              'selection_source': 'https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Convolution.cpp#L492-L516'}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
