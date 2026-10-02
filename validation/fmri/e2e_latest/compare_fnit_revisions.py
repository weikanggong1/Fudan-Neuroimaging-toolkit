"""核对两个已完成 FNIT 连续运行的完整科学产物；不追加影像变换。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def numeric_pair(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return {'shape_and_dtype_equal': False, 'decoded_bits_equal': False}
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    unsigned = np.dtype('u' + str(left.dtype.itemsize))
    equal = left.view(unsigned) == right.view(unsigned)
    exact = bool(equal.all())
    return {'shape_and_dtype_equal': True, 'shape': list(left.shape), 'dtype': str(left.dtype),
            'values': int(left.size), 'decoded_bits_equal': exact,
            'different_bit_patterns': int(left.size - equal.sum()),
            'max_absolute_difference': 0.0 if exact else float(np.max(np.abs(
                left.astype(np.float64) - right.astype(np.float64))))}


def file_pair(left, right):
    left, right = Path(left), Path(right)
    row = {'before_sha256': sha256(left), 'after_sha256': sha256(right)}
    row['file_bytes_equal'] = row['before_sha256'] == row['after_sha256']
    if left.name.endswith('.npy'):
        row.update(numeric_pair(np.load(left), np.load(right)))
    elif left.name.endswith(('.mat', '.par')):
        row.update(numeric_pair(np.loadtxt(left), np.loadtxt(right)))
    else:
        a, b = nib.load(left), nib.load(right)
        if isinstance(a, nib.gifti.GiftiImage):
            row['darray_count_equal'] = len(a.darrays) == len(b.darrays)
            row['arrays'] = [numeric_pair(x.data, y.data) for x, y in zip(a.darrays, b.darrays)]
            row['decoded_bits_equal'] = row['darray_count_equal'] and all(
                value['decoded_bits_equal'] for value in row['arrays'])
        else:
            if isinstance(a, nib.Cifti2Image):
                row['scientific_axes_equal'] = all(a.header.get_axis(i) == b.header.get_axis(i)
                                                   for i in range(2))
            else:
                row['affine_equal'] = bool(np.array_equal(a.affine, b.affine))
                fields = ('dim', 'pixdim', 'qform_code', 'sform_code', 'quatern_b', 'quatern_c',
                          'quatern_d', 'qoffset_x', 'qoffset_y', 'qoffset_z', 'srow_x', 'srow_y',
                          'srow_z', 'xyzt_units', 'datatype', 'bitpix', 'scl_slope', 'scl_inter')
                row['scientific_header_equal'] = all(np.array_equal(a.header[k], b.header[k],
                                                                    equal_nan=True) for k in fields)
            # Identical complete files already prove decoded bits; other files
            # are compared after loading, without changing scale or coordinates.
            if row['file_bytes_equal']:
                row.update({'decoded_bits_equal': True, 'shape': list(a.shape),
                            'values': int(np.prod(a.shape)), 'different_bit_patterns': 0})
            else:
                row.update(numeric_pair(np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-report', type=Path, required=True)
    parser.add_argument('--after-report', type=Path, required=True)
    parser.add_argument('--before-files', type=Path, required=True)
    parser.add_argument('--after-files', type=Path, required=True)
    parser.add_argument('--report-out', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    runs = [json.loads(path.read_text()) for path in (args.before_report, args.after_report)]
    if not all(run['source_unchanged_during_run'] for run in runs):
        raise ValueError('each run must attest unchanged runtime sources')
    if runs[0]['input_sha256'] != runs[1]['input_sha256'] or runs[0]['configuration'] != runs[1]['configuration']:
        raise ValueError('compare the same inputs, resources and scientific parameters')
    files = [json.loads(path.read_text()) for path in (args.before_files, args.after_files)]
    pairs = {}
    for role, path in files[0]['volume'].items():
        left = Path(path)
        if role in files[1]['volume'] and left.is_file() and left.name.endswith(
                ('.nii.gz', '.npy', '.mat', '.par')):
            pairs['volume_' + role] = (left, Path(files[1]['volume'][role]))
    for role in ('left', 'right', 'dtseries'):
        pairs['surface_' + role] = tuple(Path(value['surface'][role]) for value in files)
    for index in range(2):
        pairs['surface_registered_sphere_' + ('L' if index == 0 else 'R')] = tuple(
            Path(value['surface']['registered_spheres'][index]) for value in files)
    results = {role: file_pair(*pair) for role, pair in pairs.items()}
    report = {'schema_version': 1, 'before_revision': runs[0]['source_revision'],
              'after_revision': runs[1]['source_revision'], 'same_inputs_resources_configuration': True,
              'execution_report_sha256': {'before': sha256(args.before_report), 'after': sha256(args.after_report)},
              'files': results, 'all_decoded_bits_equal': all(row['decoded_bits_equal'] for row in results.values()),
              'all_scientific_headers_and_axes_equal': all(row.get('affine_equal', True) and
                  row.get('scientific_header_equal', True) and row.get('scientific_axes_equal', True)
                  for row in results.values()),
              'comparison_script_sha256': sha256(__file__),
              'validation_seconds_excluded_from_pipeline': time.perf_counter() - started,
              'scope': 'Complete scientific outputs and captured numerical intermediates of two successful continuous runs; all frames, no resampling or fitting.',
              'privacy': 'Only anonymous role names, hashes and scalar comparisons; native geometry and arrays remain private.'}
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'files': len(results), 'all_decoded_bits_equal': report['all_decoded_bits_equal'],
                      'all_scientific_headers_and_axes_equal': report['all_scientific_headers_and_axes_equal']}), flush=True)


if __name__ == '__main__':
    main()
