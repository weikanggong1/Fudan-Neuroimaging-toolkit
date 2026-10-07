"""Independent layer layout controls using identical saved original tail inputs."""
import argparse
import hashlib
import json
from pathlib import Path

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
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    reference = Path(args.reference)
    detector = models.FeatureDetector(args.weights)
    strides = [layer.weight.stride() for layer in detector.layers]
    rows = {}
    with torch.inference_mode():
        for index in range(2, 9):
            layer = detector.layers[index]
            prefix = 'layer_' + str(index) + '_'
            path = reference / (prefix + 'input.npy')
            image = torch.from_numpy(np.load(path)).permute(0, 4, 1, 2, 3)
            names = ['convolution'] if index == 8 else ['convolution', 'activation']
            if index < 4:
                names.append('pool')
            expected = {name: np.load(reference / (prefix + name + '.npy')) for name in names}
            variants = {}
            for input_last, weight_last in ((False, False), (True, False), (False, True), (True, True)):
                source = image.contiguous(memory_format=torch.channels_last_3d if input_last else torch.contiguous_format)
                weight = layer.weight.contiguous(memory_format=torch.channels_last_3d if weight_last else torch.contiguous_format)
                convolution = F.conv3d(source, weight, layer.bias, padding=1)
                if index == 8:
                    convolution = F.relu(convolution)
                    stages = [('convolution', convolution)]
                else:
                    activation = F.leaky_relu(convolution, .2)
                    stages = [('convolution', convolution), ('activation', activation)]
                    if index < 4:
                        stages.append(('pool', F.max_pool3d(activation, 2)))
                variants['input_last_' + str(input_last) + '_weight_last_' + str(weight_last)] = {
                    name: metrics(tensor.permute(0, 2, 3, 4, 1).numpy(), expected[name])
                    for name, tensor in stages}
            rows[str(index)] = {'input_sha256': digest(path), 'variants': variants}
    if strides != [layer.weight.stride() for layer in detector.layers]:
        raise RuntimeError('experiment changed parameter strides')
    report = {'scope': __doc__, 'rows': rows, 'parameters_unchanged': True,
              'models_sha256': digest(models.__file__), 'weights_sha256': digest(args.weights),
              'worker_sha256': digest(__file__),
              'reference_report_sha256': digest(reference / 'report.private.json'),
              'torch_version': torch.__version__, 'final_convolution_includes_relu': True}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
