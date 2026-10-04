"""Isolated real-input affine stages; reference mode uses the original interpreter.

The saved arrays are private diagnostics. This script is not imported by FNIT.
--inputs replays a declared, hashed normalized input and isolates CNN/arithmetic
from image decoding and network-space preprocessing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('fnit', 'reference'), required=True)
    parser.add_argument('--moving', required=True)
    parser.add_argument('--fixed', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--inputs')
    parser.add_argument('--features', help='replay hashed frozen feature maps; no CNN')
    parser.add_argument('--extent', type=int, default=256)
    parser.add_argument('--reset-row', action='store_true', help='isolated CPU prototype only')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'scope': 'isolated real-input affine stages, not complete CLI timing',
              'role': args.role, 'extent': args.extent, 'inputs_replayed': bool(args.inputs),
              'features_replayed': bool(args.features),
              'prototype_reset_row': args.reset_row,
              'input_sha256': {name: digest(getattr(args, name)) for name in ('moving', 'fixed')},
              'weight_sha256': digest(args.weight), 'worker_sha256': digest(__file__)}
    if args.role == 'fnit':
        import nibabel as nib
        import torch
        from fnit.synthmorph import models, pipeline, spatial
        torch.set_num_threads(8)
        images = [nib.load(args.moving), nib.load(args.fixed)]
        tensors = []
        if args.inputs:
            replay = np.load(args.inputs)
            tensors = [torch.from_numpy(replay[name].copy()).permute(0, 4, 1, 2, 3)
                       for name in ('moving', 'fixed')]
            report['replayed_inputs_sha256'] = digest(args.inputs)
        else:
            for image in images:
                mapping, _ = pipeline.network_space(image, (args.extent,) * 3)
                tensor = spatial.transform(pipeline._tensor(image, 'cpu'), mapping,
                                           shape=(args.extent,) * 3)
                tensor -= tensor.min()
                tensors.append(tensor / tensor.max())
        if args.reset_row:
            # Diagnose the original 3x4 -> homogeneous-square boundary. This
            # does not install or mutate a production function.
            def reset(matrix):
                out = matrix.clone()
                out[..., 3, :] = matrix.new_tensor([0, 0, 0, 1])
                return out
        else:
            reset = lambda matrix: matrix
        with torch.inference_mode():
            if args.features:
                replay_features = np.load(args.features)
                features = [torch.from_numpy(replay_features[f'feature_{i}'].copy()).permute(0, 4, 1, 2, 3)
                            for i in range(2)]
                report['replayed_features_sha256'] = digest(args.features)
            else:
                network = models.AffineNetwork(args.weight)
                features = [network.detector(t[..., ::2, ::2, ::2]) for t in tensors]
            pairs = [models.barycenter(f, (args.extent,) * 3) for f in features]
            centers, mass = zip(*pairs)
            weights = (mass[0] / mass[0].sum(-1, keepdim=True)) * (mass[1] / mass[1].sum(-1, keepdim=True))
            fits = [models.fit_affine(centers[0], centers[1], weights),
                    models.fit_affine(centers[1], centers[0], weights)]
            average = reset((fits[0] + torch.linalg.inv(fits[1])) * 0.5)
            inverse = reset(torch.linalg.inv(average))
            halves = [reset(models.matrix_sqrt(m)) for m in (average, inverse)]
        convert = lambda t: t.detach().numpy()
        channel_last = lambda t: convert(t.permute(0, 2, 3, 4, 1))
        arrays = {f'input_{i}': channel_last(t) for i, t in enumerate(tensors)}
        arrays.update({f'feature_{i}': channel_last(t) for i, t in enumerate(features)})
        arrays.update({f'center_{i}': convert(t) for i, t in enumerate(centers)})
        arrays.update({f'mass_{i}': convert(t) for i, t in enumerate(mass)})
        arrays.update({f'fit_{i}': convert(t) for i, t in enumerate(fits)})
        arrays.update({f'half_{i}': convert(t) for i, t in enumerate(halves)})
        arrays.update(weights=convert(weights), average=convert(average), inverse=convert(inverse))
        report['source_sha256'] = {name: digest(module.__file__) for name, module in
                                   [('models', models), ('pipeline', pipeline), ('spatial', spatial)]}
    else:
        import tensorflow as tf
        import surfa as sf
        import voxelmorph as vxm
        import neurite as ne
        from synthmorph import registration
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(8)
        if args.inputs:
            replay = np.load(args.inputs)
            tensors = [tf.convert_to_tensor(replay[name]) for name in ('moving', 'fixed')]
            report['replayed_inputs_sha256'] = digest(args.inputs)
        else:
            images = [sf.load_volume(args.moving), sf.load_volume(args.fixed)]
            tensors = [registration.transform(image.data, registration.network_space(
                image, (args.extent,) * 3)[0], shape=(args.extent,) * 3, normalize=True,
                batch=True) for image in images]
        if args.features:
            replay_features = np.load(args.features)
            features = [tf.convert_to_tensor(replay_features[f'feature_{i}']) for i in range(2)]
            report['replayed_features_sha256'] = digest(args.features)
        else:
            network = vxm.networks.VxmAffineFeatureDetector(
                in_shape=(args.extent,) * 3, bidir=True, make_dense=False, return_feat=True)
            registration.load_weights(network, args.weight)
            prediction = network(tensors)
            features = prediction[-2:]
        centers = [ne.utils.barycenter(f, axes=range(1, 4), normalize=True,
                                      shift_center=True) * args.extent for f in features]
        mass = [tf.reduce_sum(f, axis=(1, 2, 3)) for f in features]
        weights = (mass[0] / tf.reduce_sum(mass[0], axis=-1, keepdims=True)) * (mass[1] / tf.reduce_sum(mass[1], axis=-1, keepdims=True))
        fits = [vxm.utils.fit_affine(centers[0], centers[1], weights=weights),
                vxm.utils.fit_affine(centers[1], centers[0], weights=weights)]
        average = (fits[0] + vxm.utils.invert_affine(fits[1])) * 0.5
        inverse = vxm.utils.invert_affine(average)
        halves = [tf.linalg.sqrtm(vxm.utils.make_square_affine(t)) for t in (average, inverse)]
        arrays = {f'input_{i}': t.numpy() for i, t in enumerate(tensors)}
        arrays.update({f'feature_{i}': t.numpy() for i, t in enumerate(features)})
        arrays.update({f'center_{i}': t.numpy() for i, t in enumerate(centers)})
        arrays.update({f'mass_{i}': t.numpy() for i, t in enumerate(mass)})
        arrays.update({f'fit_{i}': vxm.utils.make_square_affine(t).numpy() for i, t in enumerate(fits)})
        arrays.update({f'half_{i}': t.numpy() for i, t in enumerate(halves)})
        arrays.update(weights=weights.numpy(), average=vxm.utils.make_square_affine(average).numpy(),
                      inverse=vxm.utils.make_square_affine(inverse).numpy())
        report['source_sha256'] = {name: digest(module.__file__) for name, module in
                                   [('registration', registration), ('vxm_networks', vxm.networks),
                                    ('vxm_utils', vxm.utils), ('ne_utils', ne.utils)]}
    np.savez(output / 'normalized_inputs.npz', moving=arrays['input_0'], fixed=arrays['input_1'])
    del arrays['input_0'], arrays['input_1']
    np.savez_compressed(output / 'stages.npz', **arrays)
    report['diagnostic_seconds'] = time.perf_counter() - started
    report['stage_sha256'] = digest(output / 'stages.npz')
    report['status'] = 'complete'
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
