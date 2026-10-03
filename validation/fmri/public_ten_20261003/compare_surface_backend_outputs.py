"""只读比较两个完整 surface 的实际保存数组；文件来源、矩阵与算法均不拟合或改写。"""
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


def metric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {'count': 0, 'mean': None, 'minimum': None, 'q05': None, 'median': None, 'q95': None}
    return {'count': int(values.size), 'mean': float(values.mean()), 'minimum': float(values.min()),
            'q05': float(np.quantile(values, .05)), 'median': float(np.median(values)), 'q95': float(np.quantile(values, .95))}


def pearson_along(first, second, axis):
    first = first - first.mean(axis=axis, keepdims=True)
    second = second - second.mean(axis=axis, keepdims=True)
    denominator = np.sqrt(np.sum(first * first, axis=axis) * np.sum(second * second, axis=axis))
    valid = denominator > 0
    result = np.sum(first * second, axis=axis)[valid] / denominator[valid]
    return metric_summary(np.clip(result, -1., 1.)), int(np.count_nonzero(~valid))


def data_metrics(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('saved output has mismatched shape or nonfinite values')
    exact = np.array_equal(reference, candidate)
    bits_equal = reference.dtype == candidate.dtype and reference.tobytes() == candidate.tobytes()
    reference, candidate = reference.astype(np.float64), candidate.astype(np.float64)
    difference = candidate - reference
    rmse = np.sqrt(np.mean(difference * difference))
    rms = np.sqrt(np.mean(reference * reference))
    temporal, temporal_undefined = pearson_along(reference, candidate, axis=0)
    spatial, spatial_undefined = pearson_along(reference, candidate, axis=1)
    bias = difference.mean(axis=0)
    return {'shape': list(reference.shape), 'maximum_absolute_difference': float(np.max(np.abs(difference))),
            'mean_absolute_difference': float(np.mean(np.abs(difference))), 'rmse': float(rmse),
            'reference_rms': float(rms), 'nrmse_by_reference_rms': float(rmse / rms) if rms > 0 else None,
            'all_array_values_exact': bool(exact), 'all_array_bits_exact': bool(bits_equal),
            'mean_tmean_bias': float(bias.mean()), 'mean_absolute_tmean_bias': float(np.abs(bias).mean()),
            'temporal_pearson_over_defined_grayordinates': temporal,
            'temporal_pearson_undefined_grayordinate_count': temporal_undefined,
            'spatial_pearson_over_defined_frames': spatial,
            'spatial_pearson_undefined_frame_count': spatial_undefined}


def read_files(path):
    value = json.loads(path.read_text())
    files = value['result'] if 'result' in value else value
    for key in ('left', 'right', 'dtseries'):
        if not Path(files[key]).is_file():
            raise FileNotFoundError(files[key])
    if not files.get('registered_spheres'):
        metadata = json.loads(Path(files['metadata']).read_text())
        signal = metadata['FNIT']['Signal']
        files['registered_spheres'] = [str(Path(files[key]).with_name(
            Path(files[key]).name.split('_space-fsLR_')[0] + f'_space-fsLR_desc-{signal}Reg_sphere.surf.gii')) for key in ('left', 'right')]
    return files


def compare(reference_file, candidate_file):
    source_sha = sha256(__file__)
    reference, candidate = read_files(reference_file), read_files(candidate_file)
    paths = {'reference_private_file_map': reference_file, 'candidate_private_file_map': candidate_file}
    for name, files in (('reference', reference), ('candidate', candidate)):
        for key in ('left', 'right', 'dtseries', 'metadata'):
            paths[name + '/' + key] = Path(files[key])
        for index, sphere in enumerate(files['registered_spheres']):
            paths[name + '/registered_sphere_' + str(index)] = Path(sphere)
    before = {name: sha256(path) for name, path in paths.items()}
    tick = time.perf_counter()
    result = {'status': 'passed', 'scope': 'Readonly full saved-output comparison, independently timed after both complete API calls; no registration, fitting, data normalization or producer output modification.',
              'comparison_source_sha256': source_sha, 'data_metrics': {}, 'sphere_metrics': {}}
    first = nib.load(reference['dtseries'])
    second = nib.load(candidate['dtseries'])
    if first.header.get_axis(0) != second.header.get_axis(0) or first.header.get_axis(1) != second.header.get_axis(1):
        raise ValueError('CIFTI time or brain model axes differ')
    ref = np.asarray(first.dataobj)
    cand = np.asarray(second.dataobj)
    result['data_metrics']['dtseries'] = data_metrics(ref, cand)
    result['data_metrics']['brain_models'] = {}
    for name, selection, _ in first.header.get_axis(1).iter_structures():
        result['data_metrics']['brain_models'][name] = data_metrics(ref[:, selection], cand[:, selection])
    for key in ('left', 'right'):
        a, b = nib.load(reference[key]), nib.load(candidate[key])
        if len(a.darrays) != len(b.darrays):
            raise ValueError('GIFTI frame counts differ')
        result['data_metrics'][key] = data_metrics(np.stack([d.data for d in a.darrays]), np.stack([d.data for d in b.darrays]))
    for hemi, a_path, b_path in zip(('L', 'R'), reference['registered_spheres'], candidate['registered_spheres']):
        a, b = nib.load(a_path), nib.load(b_path)
        if len(a.darrays) != len(b.darrays):
            raise ValueError('sphere GIFTI array counts differ')
        arrays = []
        for x, y in zip(a.darrays, b.darrays):
            if x.intent != y.intent or x.data.shape != y.data.shape:
                raise ValueError('sphere intents or geometry shapes differ')
            arrays.append({'intent': int(x.intent), 'shape': list(x.data.shape),
                           'values_exact': bool(np.array_equal(x.data, y.data)),
                           'maximum_absolute_difference': float(np.max(np.abs(x.data.astype(np.float64) - y.data.astype(np.float64))))})
        result['sphere_metrics'][hemi] = arrays
    after = {name: sha256(path) for name, path in paths.items()}
    if before != after or source_sha != sha256(__file__):
        raise ValueError('saved reference/candidate output or validator changed during comparison')
    result.update(comparison_seconds=time.perf_counter()-tick, input_sha256_before=before,
                  input_sha256_after=after, input_guards_equal=True, comparison_source_guard_equal=True,
                  CIFTI_time_axis_exact=True, CIFTI_brain_models_exact=True,
                  dtseries_file_bytes_equal=before['reference/dtseries'] == before['candidate/dtseries'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-files', required=True, type=Path)
    parser.add_argument('--candidate-files', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.reference_files, args.candidate_files)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('FULL_SURFACE_COMPARISON_SAVED', result['comparison_seconds'], flush=True)


if __name__ == '__main__':
    main()
