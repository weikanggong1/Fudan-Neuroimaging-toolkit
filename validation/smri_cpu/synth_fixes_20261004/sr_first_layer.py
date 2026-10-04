"""Compare the first SynthSR layer on a captured, real full-volume input.

Reference TensorFlow is an isolated diagnostic process. Production FNIT never
imports it. Arrays remain in the private run directory; reports expose hashes
and scalar errors only. This is a stage diagnosis, not an inference benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def metrics(reference, actual):
    count = different = 0
    total = squared = 0.0
    maximum = 0.0
    for i in range(reference.shape[2]):
        a = np.asarray(reference[:, :, i], dtype=np.float64)
        b = np.asarray(actual[:, :, i], dtype=np.float64)
        d = np.abs(a - b)
        count += d.size
        different += np.count_nonzero(d)
        total += d.sum()
        squared += np.square(d).sum()
        maximum = max(maximum, float(d.max()))
    return {'count': count, 'different': int(different), 'mae': total/count,
            'rmse': float(np.sqrt(squared/count)), 'max_abs': maximum}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('reference', 'candidate'), required=True)
    p.add_argument('--network-input', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--reference-dir', type=Path)
    p.add_argument('--official-script', type=Path)
    a = p.parse_args()
    if a.output_dir.exists():
        p.error('preserve previous runs; choose a fresh output directory')
    a.output_dir.mkdir(parents=True)
    if a.weights.stat().st_size != 106163752 or digest(a.weights) != 'a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b':
        raise ValueError('default SynthSR weight size/SHA differs from fixed resource manifest')
    x = np.load(a.network_input)
    report = {'schema': 'fnit.synthsr.first_layer.real.v1',
              'arm': a.arm, 'input_shape': list(x.shape), 'input_dtype': str(x.dtype),
              'input_sha256': digest(a.network_input), 'weight_sha256': digest(a.weights),
              'driver_sha256': digest(__file__), 'timing_scope': 'first-layer diagnostic only', 'results': {}}
    if a.arm == 'reference':
        if digest(a.official_script) != '917b20e90b00c1b2807ffcc1e8e2db123d3a63d52c8d395b8ba5a7f078c8b098':
            raise ValueError('reference source changed')
        ns = runpy.run_path(str(a.official_script), run_name='_isolated_reference_')
        tf = ns['tf']
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)
        model = ns['build_model'](str(a.weights))
        layer = model.get_layer('unet_conv_downarm_0_0')
        first = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
        started = time.perf_counter()
        output = first.predict(x.transpose(0, 2, 3, 4, 1), verbose=0)
        report['results']['keras_conv_elu'] = {'seconds': time.perf_counter()-started}
        np.save(a.output_dir/'elu.npy', output.transpose(0, 4, 1, 2, 3))
        del output
        layer.activation = tf.keras.activations.linear
        raw_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
        output = raw_model.predict(x.transpose(0, 2, 3, 4, 1), verbose=0)
        np.save(a.output_dir/'raw.npy', output.transpose(0, 4, 1, 2, 3))
        report['tensorflow_version'] = tf.__version__
    else:
        import torch
        from torch.nn import functional as F
        from fnit.synthsr.model import SynthSRUNet, load_h5_weights
        torch.set_num_threads(8)
        model = load_h5_weights(SynthSRUNet(), a.weights).eval()
        report['torch_version'] = torch.__version__
        report['production_model_sha256'] = digest(__import__('fnit.synthsr.model', fromlist=['']).__file__)
        raw_ref = np.load(a.reference_dir/'raw.npy', mmap_mode='r')
        elu_ref = np.load(a.reference_dir/'elu.npy', mmap_mode='r')
        with torch.inference_mode():
            for mode in ('contiguous', 'channels_last', 'no_onednn', 'fp64_accumulation'):
                tensor = torch.from_numpy(x)
                conv = model.down[0][0]
                if mode == 'channels_last':
                    tensor = tensor.contiguous(memory_format=torch.channels_last_3d)
                if mode == 'fp64_accumulation':
                    conv = __import__('copy').deepcopy(conv).double()
                    tensor = tensor.double()
                old = torch.backends.mkldnn.enabled
                if mode == 'no_onednn':
                    torch.backends.mkldnn.enabled = False
                try:
                    started = time.perf_counter()
                    result = conv(tensor).float()
                    elapsed = time.perf_counter()-started
                finally:
                    torch.backends.mkldnn.enabled = old
                raw = result.numpy()
                row = {'seconds': elapsed, 'raw': metrics(raw_ref, raw),
                       'elu': metrics(elu_ref, F.elu(result).numpy())}
                report['results'][mode] = row
                del raw, result, tensor
    (a.output_dir/'report.public.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
