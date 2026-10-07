"""Finite original first-convolution oracle on an actual saved CLI input."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'weights', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    datasets = {}
    with h5py.File(args.weights) as handle:
        def visit(name, value):
            if isinstance(value, h5py.Dataset) and name.split('/')[-2] == 'conv3d':
                datasets[name.split('/')[-1].split(':')[0]] = value[...]
        handle.visititems(visit)
    if set(datasets) != {'kernel', 'bias'}:
        raise ValueError('ambiguous first affine convolution weights')
    values = np.load(args.input)
    layer = tf.keras.layers.Conv3D(256, kernel_size=3, padding='same')
    layer.build(values.shape)
    layer.set_weights([datasets['kernel'], datasets['bias']])
    convolution = layer(tf.constant(values))
    activation = tf.keras.layers.LeakyReLU(.2)(convolution)
    pooled = tf.keras.layers.MaxPool3D(dtype=tf.float32)(activation)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = {}
    for name, tensor in [('convolution', convolution), ('activation', activation), ('pool', pooled)]:
        array = tensor.numpy()
        np.save(output / (name + '.npy'), array)
        rows[name] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                      'file_sha256': digest(output / (name + '.npy')),
                      'array_sha256': hashlib.sha256(array.tobytes()).hexdigest()}
    # Separate preprocessing from convolution on the bound live192 source.
    # The resulting normalized input must equal the observed real CLI input.
    import voxelmorph as vxm
    live = Path(args.input).parent
    for index in range(2):
        source = np.load(live / ('source_data_' + str(index) + '.npy'))
        matrix = np.load(live / ('source_to_network_' + str(index) + '.npy'))
        expected = np.load(live / ('network_input_' + str(index) + '.npy'))
        raw = vxm.utils.transform(source[..., None], matrix, fill_value=0, shift_center=False,
                                  shape=expected.shape[1:4])
        shifted = raw - tf.reduce_min(raw)
        normalized = shifted / tf.reduce_max(shifted)
        np.save(output / ('resampled_' + str(index) + '.npy'), raw.numpy())
        np.save(output / ('normalized_' + str(index) + '.npy'), normalized.numpy())
        rows['preprocessing_' + str(index)] = {
            'normalized_equals_original_live_input': bool(np.array_equal(normalized.numpy()[None], expected)),
            'resampled_sha256': digest(output / ('resampled_' + str(index) + '.npy')),
            'normalized_sha256': digest(output / ('normalized_' + str(index) + '.npy')),
            'source_sha256': digest(live / ('source_data_' + str(index) + '.npy')),
            'matrix_sha256': digest(live / ('source_to_network_' + str(index) + '.npy'))}
    report = {'scope': 'finite first affine layer, original Keras Conv3D/LeakyReLU/MaxPool3D on exact live original192 affine moving input; isolated operators, no complete CNN',
              'input_sha256': digest(args.input), 'weights_sha256': digest(args.weights),
              'worker_sha256': digest(__file__), 'tensorflow_version': tf.__version__, 'rows': rows}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
