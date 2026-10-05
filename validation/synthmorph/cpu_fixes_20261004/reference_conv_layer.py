"""Finite original affine convolution oracle on a saved real intermediate."""
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
    parser.add_argument('--layer-index', type=int, required=True)
    args = parser.parse_args()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    layer_name = 'conv3d' + ('_' + str(args.layer_index) if args.layer_index else '')
    values = {}
    with h5py.File(args.weights) as handle:
        def visit(name, dataset):
            if isinstance(dataset, h5py.Dataset) and name.split('/')[-2] == layer_name:
                key = name.split('/')[-1].split(':')[0]
                if key in values:
                    raise ValueError('ambiguous convolution weights')
                values[key] = dataset[...]
        handle.visititems(visit)
    if set(values) != {'kernel', 'bias'}:
        raise ValueError('missing convolution weights')
    image = np.load(args.input)
    layer = tf.keras.layers.Conv3D(values['bias'].size, kernel_size=3, padding='same')
    layer.build(image.shape)
    layer.set_weights([values['kernel'], values['bias']])
    convolution = layer(tf.constant(image))
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
    report = {'scope': __doc__, 'layer_index': args.layer_index,
              'input_sha256': digest(args.input), 'weights_sha256': digest(args.weights),
              'worker_sha256': digest(__file__), 'tensorflow_version': tf.__version__, 'rows': rows}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
