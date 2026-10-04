"""Replay CPU joint arithmetic against actual named original eager layers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from fnit.synthmorph import models


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--extent', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    reference = Path(args.reference)
    feature_files = [reference / ('Conv3D_conv3d_8_output_' + str(i) + '.npy') for i in range(2)]
    features = [torch.from_numpy(np.load(path)).permute(0, 4, 1, 2, 3) for path in feature_files]
    shape = (args.extent // 2,) * 3
    pairs = [models._cpu_joint_barycenter(feature, shape) for feature in features]
    normalized = [mass / models._cpu_inner_sum(mass).unsqueeze(-1) for center, mass in pairs]
    weights = normalized[0] * normalized[1]
    fits = [models._cpu_joint_fit_affine(pairs[i][0], pairs[1-i][0], weights) for i in range(2)]
    average = (fits[0] + models._cpu_joint_inverse(fits[1])) * .5
    inverse = models._cpu_joint_inverse(average)
    halves = [models._cpu_joint_matrix_sqrt(value) for value in (average, inverse)]
    centered = [models._cpu_joint_center_affine(value, shape) for value in halves]
    arrays = {'mass0': pairs[0][1], 'mass1': pairs[1][1], 'center0': pairs[0][0], 'center1': pairs[1][0],
              'weights': weights, 'fit0': fits[0][..., :3, :], 'fit1': fits[1][..., :3, :],
              'average': average[..., :3, :], 'inverse': inverse,
              'half0': halves[0], 'half1': halves[1],
              'centered0': centered[0][..., :3, :], 'centered1': centered[1][..., :3, :]}
    original = {
        'mass0': 'TFOpLambda_tf.math.reduce_sum_4_output_0.npy',
        'mass1': 'TFOpLambda_tf.math.reduce_sum_5_output_0.npy',
        'center0': 'TFOpLambda_tf.math.multiply_1_output_0.npy',
        'center1': 'TFOpLambda_tf.math.multiply_3_output_0.npy',
        'weights': 'TFOpLambda_tf.math.multiply_4_output_0.npy',
        'fit0': 'TFOpLambda_tf.linalg.matrix_transpose_1_output_0.npy',
        'fit1': 'TFOpLambda_tf.linalg.matrix_transpose_3_output_0.npy',
        'average': 'TFOpLambda_tf.math.multiply_7_output_0.npy',
        'inverse': 'TFOpLambda_tf.linalg.inv_3_output_0.npy',
        'half0': 'TFOpLambda_tf.linalg.sqrtm_output_0.npy',
        'half1': 'TFOpLambda_tf.linalg.sqrtm_1_output_0.npy',
        'centered0': 'VxmAffineFeatureDetector_vxm_affine_feature_detector_output_0.npy',
        'centered1': 'VxmAffineFeatureDetector_vxm_affine_feature_detector_output_1.npy',
    }
    rows = {}
    for name, tensor in arrays.items():
        actual = tensor.numpy()
        expected = np.load(reference / original[name])
        rows[name] = {'different_values': int(np.count_nonzero(actual != expected)),
                      'max_abs': float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))),
                      'reference_file': original[name], 'reference_file_sha256': digest(reference / original[name])}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / 'stages.npz', **{name: value.numpy() for name, value in arrays.items()})
    report = {'scope': __doc__, 'extent': args.extent, 'rows': rows,
              'features_sha256': [digest(path) for path in feature_files],
              'models_sha256': digest(models.__file__), 'worker_sha256': digest(__file__),
              'original_live_report_sha256': digest(reference / 'report.private.json')}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
