"""Third real-input preprocessing route; no CNN or registration output."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import nibabel as nib
import numpy as np
import torch
from fnit.synthmorph.pipeline import _tensor, network_space
from fnit.synthmorph.spatial import transform
from object_api_worker import array_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('moving', 'fixed', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    started = time.perf_counter()
    report = {'scope': 'real input ArrayProxy directly requested float32 vs path and decoded float64 object; preprocessing only, no CNN',
              'nibabel_version': nib.__version__, 'inputs': {}}
    for name in ('moving', 'fixed'):
        original = nib.load(getattr(args, name))
        native = np.array(original.dataobj, dtype=np.float32, copy=True)
        direct = type(original)(native, original.affine.copy(), header=original.header.copy())
        decoded64 = np.array(np.asanyarray(original.dataobj), copy=True)
        earlier = type(original)(decoded64, original.affine.copy(), header=original.header.copy())
        path_tensor, direct_tensor, earlier_tensor = (_tensor(image, 'cpu') for image in (original, direct, earlier))
        matrix, inverse = network_space(original, (256,) * 3)
        object_matrix, object_inverse = network_space(direct, (256,) * 3)
        normalized = []
        for tensor, mapping in ((path_tensor, matrix), (direct_tensor, object_matrix)):
            x = transform(tensor, mapping, shape=(256,) * 3)
            x -= x.min()
            normalized.append(x / x.max())
        difference = (path_tensor - earlier_tensor).abs()
        out = {
            'stored_dtype': str(original.get_data_dtype()),
            'ArrayProxy_slope': original.dataobj.slope,
            'ArrayProxy_inter': original.dataobj.inter,
            'default_decoded_dtype': str(decoded64.dtype),
            'direct_materialized_dtype': str(direct.dataobj.dtype),
            'direct_materialized_numpy_strides': list(direct.dataobj.strides),
            'path_tensor_strides': list(path_tensor.stride()),
            'direct_tensor_strides': list(direct_tensor.stride()),
            'direct_object_is_ndarray': isinstance(direct.dataobj, np.ndarray),
            'direct_object_filename_is_none': direct.get_filename() is None,
            'header_binary_equal': original.header.binaryblock == direct.header.binaryblock,
            'affine_equal': bool(np.array_equal(original.affine, direct.affine)),
            'network_to_image_equal': bool(np.array_equal(matrix, object_matrix)),
            'image_to_network_equal': bool(np.array_equal(inverse, object_inverse)),
            'path_tensor_sha256': array_digest(path_tensor.numpy()),
            'direct_tensor_sha256': array_digest(direct_tensor.numpy()),
            'path_direct_tensor_equal': bool(torch.equal(path_tensor, direct_tensor)),
            'path_direct_normalized_equal': bool(torch.equal(*normalized)),
            'normalized_path_sha256': array_digest(normalized[0].numpy()),
            'normalized_direct_sha256': array_digest(normalized[1].numpy()),
            'path_vs_earlier_float64_object_tensor_different_values': int(torch.count_nonzero(difference)),
            'path_vs_earlier_float64_object_tensor_max_abs': float(difference.max()),
        }
        if not all(out[key] for key in ('header_binary_equal', 'affine_equal', 'network_to_image_equal',
                                       'image_to_network_equal', 'path_direct_tensor_equal', 'path_direct_normalized_equal')):
            raise RuntimeError('direct float32 materialization failed real input preprocessing match')
        report['inputs'][name] = out
    report['preprocessing_control_seconds'] = time.perf_counter() - started
    report['status'] = 'complete'
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
