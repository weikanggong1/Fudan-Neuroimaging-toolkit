"""Eight complete FAST outputs; run separately from measured inference."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


SUFFIXES = ('pve_0', 'pve_1', 'pve_2', 'seg', 'pveseg', 'mixeltype', 'bias', 'restore')


def metrics(first, second, selected):
    a = first[selected].astype(np.float64)
    b = second[selected].astype(np.float64)
    difference = a - b
    result = {'values': int(a.size), 'mae': float(np.abs(difference).mean()),
              'rmse': float(np.sqrt(np.mean(difference ** 2))),
              'maximum_absolute_error': float(np.max(np.abs(difference))),
              'p99_absolute_error': float(np.percentile(np.abs(difference), 99)),
              'changed_numerical_values': int(np.count_nonzero(difference))}
    result['pearson_r'] = (float(np.corrcoef(a, b)[0, 1])
                           if np.std(a) > 0 and np.std(b) > 0 else None)
    return result


def compare(first_prefix, second_prefix, included):
    rows = {}
    for suffix in SUFFIXES:
        first_path = Path(str(first_prefix) + '_' + suffix + '.nii.gz')
        second_path = Path(str(second_prefix) + '_' + suffix + '.nii.gz')
        first, second = nib.load(first_path), nib.load(second_path)
        a, b = np.asanyarray(first.dataobj), np.asanyarray(second.dataobj)
        row = {'first_sha256': hashlib.sha256(first_path.read_bytes()).hexdigest(),
               'second_sha256': hashlib.sha256(second_path.read_bytes()).hexdigest(),
               'shape_equal': a.shape == b.shape,
               'affine_equal': bool(np.array_equal(first.affine, second.affine)),
               'affine_maximum_absolute_error': float(np.max(np.abs(first.affine - second.affine))),
               'first_dtype': str(first.get_data_dtype()), 'second_dtype': str(second.get_data_dtype()),
               'finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
               'header_fields_equal': {key: bool(np.array_equal(first.header[key], second.header[key]))
                                       for key in ('dim', 'pixdim', 'datatype', 'xyzt_units',
                                                   'qform_code', 'sform_code', 'intent_code')}}
        if a.shape != b.shape or a.shape != included.shape or not row['affine_equal']:
            row['comparison_status'] = 'geometry_mismatch'
            rows[suffix] = row
            continue
        row['brain'] = metrics(a, b, included)
        row['whole_grid'] = metrics(a, b, np.ones(a.shape, dtype=np.bool_))
        row['values_exact_equal'] = bool(np.array_equal(a, b))
        row['dtype_equal'] = first.get_data_dtype() == second.get_data_dtype()
        if a.dtype == b.dtype:
            row['changed_bit_patterns'] = int(np.count_nonzero(
                np.ascontiguousarray(a).view('u' + str(a.dtype.itemsize))
                != np.ascontiguousarray(b).view('u' + str(b.dtype.itemsize))))
        voxel_volume = abs(float(np.linalg.det(first.affine[:3, :3])))
        if suffix.startswith('pve_'):
            first_volume = float(a[included].sum(dtype=np.float64) * voxel_volume)
            second_volume = float(b[included].sum(dtype=np.float64) * voxel_volume)
            row['soft_volume_mm3'] = {'first': first_volume, 'second': second_volume,
                                     'difference_percent': ((second_volume / first_volume - 1) * 100
                                                            if first_volume else None)}
            aa, bb = a[included] > .5, b[included] > .5
            total = int(aa.sum() + bb.sum())
            row['dice_0.5'] = float(2 * np.count_nonzero(aa & bb) / total) if total else 1.0
        elif suffix in ('seg', 'pveseg', 'mixeltype'):
            row['labels'] = {}
            for label in (range(6) if suffix == 'mixeltype' else range(1, 4)):
                aa, bb = a[included] == label, b[included] == label
                n1, n2 = int(aa.sum()), int(bb.sum())
                row['labels'][str(label)] = {
                    'dice': float(2 * np.count_nonzero(aa & bb) / (n1 + n2)) if n1 + n2 else 1.0,
                    'first_voxels': n1, 'second_voxels': n2,
                    'volume_difference_percent': ((n2 / n1 - 1) * 100) if n1 else None}
        rows[suffix] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-prefix', required=True)
    parser.add_argument('--second-prefix', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--image', required=True, help='Same FAST input; intersect positive brain mask')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    included = (np.asanyarray(nib.load(args.mask).dataobj) > 0
                ) & (np.asanyarray(nib.load(args.image).dataobj) > 0)
    if not np.any(included):
        raise ValueError('comparison brain region is empty')
    result = {'scope': 'Every voxel of all eight maps; brain and whole-grid metrics are separate.',
              'brain_voxels': int(included.sum()),
              'outputs': compare(args.first_prefix, args.second_prefix, included)}
    result['all_value_bits_equal'] = all(row.get('changed_bit_patterns') == 0
                                         for row in result['outputs'].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_value_bits_equal': result['all_value_bits_equal'],
                      'output_maps': len(result['outputs'])}))


if __name__ == '__main__':
    main()
