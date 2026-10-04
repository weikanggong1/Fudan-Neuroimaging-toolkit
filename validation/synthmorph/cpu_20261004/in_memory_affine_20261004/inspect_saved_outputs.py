"""Read saved outputs and real input preprocessing; never invoke a network."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import nibabel as nib
import numpy as np
import torch
from fnit._transforms import load_lta
from fnit.synthmorph.pipeline import _tensor, network_space
from fnit.synthmorph.spatial import transform
from object_api_worker import materialize, array_digest


def equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind in 'fc' and b.dtype.kind in 'fc' else bool(np.array_equal(a, b))


def geometry_pair(candidate, reference):
    a, b = nib.load(str(candidate)), nib.load(str(reference))
    extensions = lambda x: [(int(e.get_code()), hashlib.sha256(e._raw).hexdigest()) for e in x.header.extensions]
    return {
        'data_equal': equal(np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)),
        'data_array_sha256': {'candidate': array_digest(np.asanyarray(a.dataobj)), 'reference': array_digest(np.asanyarray(b.dataobj))},
        'affine_equal': equal(a.affine, b.affine),
        'qform_equal': equal(a.get_qform(), b.get_qform()),
        'sform_equal': equal(a.get_sform(), b.get_sform()),
        'shape_equal': a.shape == b.shape,
        'dtype_equal': a.get_data_dtype() == b.get_data_dtype(),
        'header_binary_equal': a.header.binaryblock == b.header.binaryblock,
        'different_header_fields': [k for k in a.header.keys() if not equal(a.header[k], b.header[k])],
        'extensions_equal': extensions(a) == extensions(b),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('moving', 'fixed', 'candidate', 'old-candidate', 'official', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    started = time.perf_counter()
    out = {'scope': 'posthoc actual saved outputs and real preprocessing only; no CNN/registration rerun',
           'torch_version': torch.__version__, 'numpy_version': np.__version__,
           'mkldnn_enabled': torch.backends.mkldnn.enabled, 'inputs': {}, 'saved_outputs': {}}
    for name in ('moving', 'fixed'):
        raw = nib.load(getattr(args, name))
        obj, _ = materialize(raw)
        original_tensor, object_tensor = _tensor(raw, 'cpu'), _tensor(obj, 'cpu')
        a, b = np.array(raw.dataobj, dtype=np.float32, copy=True), np.array(obj.dataobj, dtype=np.float32, copy=True)
        tensor_info = lambda t: {'strides': list(t.stride()), 'contiguous': t.is_contiguous(), 'array_sha256': array_digest(t.numpy())}
        matrix, inverse = network_space(raw, (256,) * 3)
        omatrix, oinverse = network_space(obj, (256,) * 3)
        normalized = []
        for tensor, mapping in ((original_tensor, matrix), (object_tensor, omatrix)):
            x = transform(tensor, mapping, shape=(256,) * 3)
            x -= x.min()
            x = x / x.max()
            normalized.append(x)
        out['inputs'][name] = {
            'raw_loaded_numpy_strides': list(np.asanyarray(raw.dataobj).strides),
            'materialized_numpy_strides': list(obj.dataobj.strides),
            'raw_tensor_copy_numpy_strides': list(a.strides),
            'object_tensor_copy_numpy_strides': list(b.strides),
            'raw_tensor': tensor_info(original_tensor), 'object_tensor': tensor_info(object_tensor),
            'tensor_values_equal': bool(torch.equal(original_tensor, object_tensor)),
            'network_to_image_equal': equal(matrix, omatrix),
            'image_to_network_equal': equal(inverse, oinverse),
            'normalized_path_tensor': tensor_info(normalized[0]),
            'normalized_object_tensor': tensor_info(normalized[1]),
            'normalized_network_values_equal': bool(torch.equal(*normalized)),
            'normalized_network_max_abs': float((normalized[0] - normalized[1]).abs().max()),
        }
    for arm, reference in (('same_candidate', Path(args.old_candidate)), ('official', Path(args.official))):
        rows = {}
        for name in ('moved', 'fixed_moved'):
            rows[name] = geometry_pair(Path(args.candidate) / (name + '.nii.gz'), reference / (name + '.nii.gz'))
        for name in ('forward', 'inverse'):
            a, b = load_lta(Path(args.candidate) / (name + '.lta')), load_lta(reference / (name + '.lta'))
            rows[name] = {
                'matrix_equal': equal(a.matrix, b.matrix),
                'matrix_max_abs': float(np.abs(a.matrix - b.matrix).max()),
                'source_shape_equal': a.source.shape == b.source.shape,
                'target_shape_equal': a.target.shape == b.target.shape,
                'source_affine_equal': equal(a.source.affine, b.source.affine),
                'target_affine_equal': equal(a.target.affine, b.target.affine),
            }
        out['saved_outputs'][arm] = rows
    out['posthoc_seconds'] = time.perf_counter() - started
    out['status'] = 'complete'
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
