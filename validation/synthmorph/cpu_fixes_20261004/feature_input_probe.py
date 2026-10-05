"""Compare the affine detector on identical actual original half-resolution inputs."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'weights', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    reference = Path(args.reference)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    detector = models.FeatureDetector(args.weights)
    strides = [layer.weight.stride() for layer in detector.layers]
    rows = {}
    arrays = {}
    with torch.inference_mode():
        for index in range(2):
            path = reference / ('VxmAffineFeatureDetector_vxm_affine_feature_detector_input_' + str(index) + '.npy')
            image = torch.from_numpy(np.load(path)).permute(0, 4, 1, 2, 3)
            expected = np.load(reference / ('Conv3D_conv3d_8_output_' + str(index) + '.npy'))
            variants = {}
            for layout in ('established', 'temporary_channels_last', 'explicit_mkldnn_channels_last'):
                if layout == 'established':
                    actual = detector(image.contiguous())
                else:
                    actual = image.contiguous(memory_format=torch.channels_last_3d)
                    for layer_index, layer in enumerate(detector.layers):
                        weight = layer.weight.contiguous(memory_format=torch.channels_last_3d)
                        if layout == 'explicit_mkldnn_channels_last':
                            actual = torch.mkldnn_convolution(actual, weight, layer.bias,
                                                             [1, 1, 1], [1, 1, 1], [1, 1, 1], 1)
                        else:
                            actual = F.conv3d(actual, weight, layer.bias, padding=1)
                        if layer_index == 8:
                            actual = F.relu(actual)
                        else:
                            actual = F.leaky_relu(actual, .2)
                            if layer_index < 4:
                                actual = F.max_pool3d(actual, 2)
                array = actual.permute(0, 2, 3, 4, 1).numpy()
                delta = array.astype(np.float64) - expected.astype(np.float64)
                variants[layout] = {'different_values': int(np.count_nonzero(array != expected)),
                                    'max_abs': float(np.abs(delta).max()),
                                    'relative_rmse': float(np.mean(delta * delta) ** .5 / (np.mean(expected.astype(np.float64) ** 2) ** .5)),
                                    'array_sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
                arrays['feature_' + str(index) + '_' + layout] = array
            rows[str(index)] = {'input_sha256': digest(path), 'variants': variants}
    if strides != [layer.weight.stride() for layer in detector.layers]:
        raise RuntimeError('layout experiment changed parameter strides')
    np.savez(output / 'features.npz', **arrays)
    report = {'scope': __doc__, 'rows': rows, 'parameters_unchanged': True,
              'models_sha256': digest(models.__file__), 'weights_sha256': digest(args.weights),
              'worker_sha256': digest(__file__), 'original_live_report_sha256': digest(reference / 'report.private.json')}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
