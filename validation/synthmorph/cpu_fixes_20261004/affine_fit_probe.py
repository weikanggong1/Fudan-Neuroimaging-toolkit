"""Compare bound real CPU fit intermediates; original runtime is diagnostic only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, reference):
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return {'different_values': int(np.count_nonzero(actual != reference)),
            'max_abs': float(np.max(np.abs(difference)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stages', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    started = time.perf_counter()
    import tensorflow as tf
    import torch
    import voxelmorph as vxm
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    rows = {}; arrays = {}
    for filename in args.stages:
        stages = np.load(filename)
        extent = int(stages['feature_0'].shape[1] * 32)
        for scale in (1., .5):
            for direction in range(2):
                source = stages[f'center_{direction}'] * scale
                target = stages[f'center_{1-direction}'] * scale
                weights = stages['weights']
                prefix = f'extent{extent}_scale{scale}_direction{direction}'
                x = tf.concat((tf.convert_to_tensor(target), tf.ones_like(target[..., :1])), -1)
                xt = tf.linalg.matrix_transpose(x) * weights[..., None, :]
                gram = xt @ x
                inverse = tf.linalg.inv(gram)
                projection = inverse @ xt
                beta = projection @ source
                original = vxm.utils.fit_affine(source, target, weights=weights)
                tensors = {name: value.numpy() for name, value in
                           [('x', x), ('xt', xt), ('gram', gram), ('inverse', inverse),
                            ('projection', projection), ('beta', beta)]}
                for name, value in tensors.items(): arrays[prefix + '_' + name] = value
                tx = torch.from_numpy(tensors['x']); txt = torch.from_numpy(tensors['xt'])
                tsource = torch.from_numpy(source)
                tgram = txt @ tx
                tinverse = torch.linalg.inv(tgram)
                tprojection = tinverse @ txt
                tbeta = tprojection @ tsource
                rows[prefix] = {
                    'original_formula_equal': np.array_equal(beta.numpy().swapaxes(-1, -2), original.numpy()),
                    'stored_fit_equal': (np.array_equal(original.numpy(), stages[f'fit_{direction}'][..., :3, :]) if scale == 1. else None),
                    'native_torch': {name: metrics(value.numpy(), tensors[name]) for name, value in
                                     [('gram', tgram), ('inverse', tinverse), ('projection', tprojection), ('beta', tbeta)]},
                    'identical_reference_input': {
                        'inverse': metrics(torch.linalg.inv(torch.from_numpy(tensors['gram'])).numpy(), tensors['inverse']),
                        'projection': metrics((torch.from_numpy(tensors['inverse']) @ txt).numpy(), tensors['projection']),
                        'beta': metrics((torch.from_numpy(tensors['projection']) @ tsource).numpy(), tensors['beta'])},
                    'double_then_float': {
                        'gram': metrics((txt.double() @ tx.double()).float().numpy(), tensors['gram']),
                        'inverse': metrics(torch.linalg.inv(torch.from_numpy(tensors['gram']).double()).float().numpy(), tensors['inverse'])}}
    np.savez_compressed(output / 'fit_stages.npz', **arrays)
    report = {'scope': 'original fit expression replay on saved real centers/weights; no CNN or image resampling',
              'stages_sha256': {Path(name).parent.parent.name: digest(name) for name in args.stages},
              'rows': rows, 'diagnostic_seconds': time.perf_counter() - started,
              'worker_sha256': digest(__file__), 'original_utils_sha256': digest(vxm.utils.__file__),
              'fit_stages_sha256': digest(output / 'fit_stages.npz')}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
