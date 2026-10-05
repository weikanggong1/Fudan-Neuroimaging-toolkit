"""Observe tensors in the original eager CLI without rebuilding its model."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import tensorflow as tf
    registration = importlib.import_module('synthmorph.registration')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    counter = {}

    def save(label, value):
        if hasattr(value, 'numpy'):
            value = value.numpy()
        elif not isinstance(value, np.ndarray):
            return
        if value.dtype.kind not in 'fc':
            return
        index = counter.get(label, 0)
        counter[label] = index + 1
        filename = label.replace('/', '_') + '_' + str(index) + '.npy'
        np.save(output / filename, value)
        rows.append({'label': label, 'index': index, 'file': filename,
                     'shape': list(value.shape), 'dtype': str(value.dtype),
                     'array_sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                     'file_sha256': digest(output / filename)})

    original_transform = registration.transform
    def transform(volume, matrix, *positional, **keywords):
        save('source_data', volume.data)
        save('source_vox2world', np.asarray(volume.geom.vox2world.matrix))
        save('source_to_network', np.asarray(matrix))
        result = original_transform(volume, matrix, *positional, **keywords)
        save('network_input', result)
        return result

    original_call = tf.keras.layers.Layer.__call__
    def layer_call(layer, *positional, **keywords):
        result = original_call(layer, *positional, **keywords)
        if not tf.executing_eagerly():
            return result
        values = tf.nest.flatten(result)
        real = [value for value in values if hasattr(value, 'numpy')]
        if not real:
            return result
        class_name = type(layer).__name__
        label = class_name + '_' + layer.name
        is_feature = class_name == 'Conv3D' and getattr(layer, 'filters', None) == 64
        is_model = class_name in ('VxmAffineFeatureDetector', 'HyperVxmJoint')
        if is_model:
            for value in tf.nest.flatten(positional):
                save(label + '_input', value)
        function = getattr(layer, 'function', None)
        function_name = getattr(function, '__name__', None)
        for value in real:
            shape = tuple(value.shape)
            small_matrix = len(shape) >= 2 and shape[-2:] in ((3, 4), (4, 4), (64, 3))
            small_weights = shape == (1, 64)
            dense_stage = class_name in ('ComposeTransform', 'AffineToDenseShift', 'VecInt')
            if is_feature or is_model or small_matrix or small_weights or dense_stage:
                save(label + '_output', value)
                rows[-1]['class'] = class_name
                rows[-1]['function'] = function_name
        return result

    registration.transform = transform
    tf.keras.layers.Layer.__call__ = layer_call
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.argv = [args.entry, *arguments]
    try:
        runpy.run_path(args.entry, run_name='__main__')
    except SystemExit as error:
        if error.code:
            raise
    finally:
        registration.transform = original_transform
        tf.keras.layers.Layer.__call__ = original_call
    import inspect
    import voxelmorph as vxm
    report = {'scope': 'actual original registration CLI source decoding, eager model inputs/features and live affine/dense stages; no extra model output, rebuild or predict graph',
              'versions': {'tensorflow': tf.__version__, 'numpy': np.__version__},
              'entry_sha256': digest(args.entry), 'worker_sha256': digest(__file__),
              'registration_sha256': digest(registration.__file__),
              'compose_source_sha256': digest(inspect.getsourcefile(vxm.utils.compose)),
              'rows': rows}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'saved_tensor_count': len(rows)}))


if __name__ == '__main__':
    main()
