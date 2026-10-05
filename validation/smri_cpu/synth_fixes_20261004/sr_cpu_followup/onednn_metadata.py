"""Validation-only TensorFlow CPU backend/op registration metadata; no benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.python.framework import op_def_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('preserve existing metadata')
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    names = ('_FusedConv3D', '_MklNativeFusedConv3D', '_FusedConv2D', '_MklNativeFusedConv2D')
    definitions = {name: str(op_def_registry.get(name)) for name in names}

    @tf.function
    def smoke(source, kernel, bias):
        return tf.nn.elu(tf.nn.bias_add(tf.nn.conv3d(source, kernel, strides=[1]*5, padding='SAME'), bias))

    # Tiny array only makes the runtime print its oneDNN version and selected ISA.
    # These values are never used for model numerical/performance acceptance.
    output = smoke(tf.ones((1,4,4,4,1), tf.float32), tf.ones((3,3,3,1,2), tf.float32), tf.ones((2,), tf.float32))
    output.numpy()
    report = {'scope': 'CPU runtime and registered fused op metadata; tiny smoke is not a benchmark',
              'tensorflow_version': tf.__version__, 'build_info': tf.sysconfig.get_build_info(),
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'registered_op_definitions': definitions, 'status': 'complete'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
