"""Finite original affine convolution tail on a saved real second-layer pool."""
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
    values = {}
    with h5py.File(args.weights) as handle:
        def visit(name, dataset):
            if not isinstance(dataset, h5py.Dataset):
                return
            parent = name.split('/')[-2]
            if not parent.startswith('conv3d_'):
                return
            index = int(parent.split('_')[-1])
            if not 2 <= index <= 8:
                return
            group = values.setdefault(index, {})
            key = name.split('/')[-1].split(':')[0]
            if key in group:
                raise ValueError('ambiguous convolution weights')
            group[key] = dataset[...]
        handle.visititems(visit)
    if set(values) != set(range(2, 9)):
        raise ValueError('missing affine tail weights')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    image = tf.constant(np.load(args.input))
    rows = {}
    for index in range(2, 9):
        group = values[index]
        if set(group) != {'kernel', 'bias'}:
            raise ValueError('missing kernel or bias')
        layer = tf.keras.layers.Conv3D(group['bias'].size, kernel_size=3,
                                      padding='same', activation='relu' if index == 8 else None)
        layer.build(image.shape)
        layer.set_weights([group['kernel'], group['bias']])
        convolution = layer(image)
        stages = [('input', image), ('convolution', convolution)]
        if index == 8:
            image = convolution
        else:
            image = tf.keras.layers.LeakyReLU(.2)(convolution)
            stages.append(('activation', image))
            if index < 4:
                image = tf.keras.layers.MaxPool3D(dtype=tf.float32)(image)
                stages.append(('pool', image))
        rows[str(index)] = {}
        for name, tensor in stages:
            array = tensor.numpy()
            path = output / ('layer_' + str(index) + '_' + name + '.npy')
            np.save(path, array)
            rows[str(index)][name] = {
                'shape': list(array.shape), 'dtype': str(array.dtype),
                'file_sha256': digest(path),
                'array_sha256': hashlib.sha256(array.tobytes()).hexdigest()}
    report = {'scope': __doc__, 'input_sha256': digest(args.input),
              'weights_sha256': digest(args.weights), 'worker_sha256': digest(__file__),
              'tensorflow_version': tf.__version__, 'rows': rows,
              'final_convolution_includes_relu': True}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
