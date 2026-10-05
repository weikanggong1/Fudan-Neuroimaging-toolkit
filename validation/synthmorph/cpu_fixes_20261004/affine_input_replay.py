"""Integrated CPU joint affine on identical saved actual original detector inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from fnit.synthmorph import models, _cpu_features


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
    network = models.AffineNetwork(args.weights).eval()
    strides = [layer.weight.stride() for layer in network.detector.layers]
    inputs = [reference / ('VxmAffineFeatureDetector_vxm_affine_feature_detector_input_' + str(i) + '.npy')
              for i in range(2)]
    images = [torch.from_numpy(np.load(path)).permute(0, 4, 1, 2, 3) for path in inputs]
    detector_calls = []
    hook = network.detector.register_forward_hook(lambda module, arguments, result: detector_calls.append(result))
    try:
        start = time.monotonic()
        with torch.inference_mode():
            enabled = (models._cpu_joint_inference_enabled(network, *images, True)
                       if hasattr(models, '_cpu_joint_inference_enabled') else None)
            result = network(*images, half_res=False, mid_space=True, return_features=True)
        call_seconds = time.monotonic() - start
    finally:
        hook.remove()
    arrays = {'affine0': result[0][None, :3, :].numpy(), 'affine1': result[1][None, :3, :].numpy(),
              'feature0': result[2].permute(0, 2, 3, 4, 1).numpy(),
              'feature1': result[3].permute(0, 2, 3, 4, 1).numpy()}
    names = {'affine0': 'VxmAffineFeatureDetector_vxm_affine_feature_detector_output_0.npy',
             'affine1': 'VxmAffineFeatureDetector_vxm_affine_feature_detector_output_1.npy',
             'feature0': 'Conv3D_conv3d_8_output_0.npy', 'feature1': 'Conv3D_conv3d_8_output_1.npy'}
    rows = {}
    for key, actual in arrays.items():
        expected_path = reference / names[key]
        expected = np.load(expected_path)
        if actual.shape != expected.shape:
            raise ValueError('stage shape differs')
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        rows[key] = {'values': int(actual.size), 'different_values': int(np.count_nonzero(actual != expected)),
                     'max_abs': float(np.abs(delta).max()), 'reference_file_sha256': digest(expected_path),
                     'actual_array_sha256': hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest()}
    if len(detector_calls) != 2 or any(detector_calls[i] is not result[2 + i] for i in range(2)):
        raise RuntimeError('detector module hooks changed')
    if strides != [layer.weight.stride() for layer in network.detector.layers]:
        raise RuntimeError('detector parameter strides changed')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / 'stages.npz', **arrays)
    report = {'scope': __doc__, 'rows': rows, 'call_seconds': call_seconds,
              'detector_forward_hooks_preserved': True, 'parameter_strides_unchanged': True,
              'CPU_inference_policy_enabled': enabled,
              'input_sha256': [digest(path) for path in inputs],
              'weights_sha256': digest(args.weights), 'worker_sha256': digest(__file__),
              'models_sha256': digest(models.__file__), 'cpu_features_sha256': digest(_cpu_features.__file__),
              'original_live_report_sha256': digest(reference / 'report.private.json')}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
