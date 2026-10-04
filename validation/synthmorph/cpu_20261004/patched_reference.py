"""Isolated diagnostic for the installed official initial-affine dtype bug.

Never imported by FNIT. Unmodified reference failures remain in the report.
Only mixed float32/float64 compose inputs are cast to the network float32
before the original compose; all original network calls and outputs remain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf
    import voxelmorph as vxm

    original = vxm.utils.compose
    patched_calls = []

    def compose(transforms, *positional, **keywords):
        dtypes = [np.asarray(value).dtype if not tf.is_tensor(value)
                  else value.dtype.as_numpy_dtype for value in transforms]
        if len({np.dtype(dtype).str for dtype in dtypes}) > 1:
            patched_calls.append([str(np.dtype(dtype)) for dtype in dtypes])
            transforms = [tf.cast(value, tf.float32) for value in transforms]
        return original(transforms, *positional, **keywords)

    vxm.utils.compose = compose
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.argv = [args.entry, *arguments]
    try:
        runpy.run_path(args.entry, run_name='__main__')
    finally:
        entry = Path(args.entry)
        Path(args.manifest).write_text(json.dumps({
            'scope': 'patched reference diagnostic; not an unmodified official run',
            'patch': 'cast mixed compose transform dtypes to float32',
            'patched_calls_input_dtypes': patched_calls,
            'original_entry_sha256': hashlib.sha256(entry.read_bytes()).hexdigest(),
            'original_compose_module': original.__module__,
            'tensorflow_version': tf.__version__,
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()
