"""Compare two completed real GEMS stages after their measured executions."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def array_contract(first, second):
    shapes_equal = first.shape == second.shape
    types_equal = first.dtype == second.dtype
    equal = shapes_equal and types_equal and np.array_equal(first, second, equal_nan=True)
    result = {'shape': list(first.shape), 'dtype': str(first.dtype),
              'shape_equal': shapes_equal, 'dtype_equal': types_equal,
              'all_values_exact': bool(equal), 'baseline_sha256': array_sha256(first),
              'candidate_sha256': array_sha256(second),
              'all_values_finite': bool(np.isfinite(first).all() and np.isfinite(second).all())}
    if shapes_equal:
        diff = first.astype(np.float64) - second.astype(np.float64)
        result['different_values'] = int(np.count_nonzero(first != second))
        result['max_absolute_error'] = float(np.max(np.abs(diff), initial=0))
        result['rmse'] = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('use a new report path')
    images = {}
    for name in ['subregions_native.nii.gz', 'highres/brainstem.nii.gz',
                 'highres/brainstem_posterior.nii.gz']:
        first, second = (nib.load(str(directory / name)) for directory in (args.baseline, args.candidate))
        values = array_contract(np.asanyarray(first.dataobj), np.asanyarray(second.dataobj))
        if first.ndim == 4:
            values['channel_count'] = first.shape[-1]
        values['affine_exact'] = bool(np.array_equal(first.affine, second.affine))
        values['zooms_exact'] = first.header.get_zooms() == second.header.get_zooms()
        images[name] = values
    parameters = {}
    with np.load(args.baseline / 'fit_parameters.private.npz') as first, np.load(args.candidate / 'fit_parameters.private.npz') as second:
        if first.files != second.files:
            raise ValueError('fit parameter schema differs')
        for name in first.files:
            parameters[name] = array_contract(first[name], second[name])
    fits = [json.loads((directory / 'fit_contract.private.json').read_text()) for directory in (args.baseline, args.candidate)]
    fit_stats_exact = json.dumps(fits[0], sort_keys=True) == json.dumps(fits[1], sort_keys=True)
    tables = {name: (args.baseline / name).read_bytes() == (args.candidate / name).read_bytes() for name in ['labels.tsv', 'volumes.tsv']}
    workers = [json.loads((directory / 'worker.json').read_text()) for directory in (args.baseline, args.candidate)]
    contracts = [json.loads((directory / 'stage_contract.public.json').read_text()) for directory in (args.baseline, args.candidate)]
    precision_keys = ['tf32_matmul', 'tf32_cudnn', 'cudnn_deterministic', 'cudnn_benchmark']
    precision_policy_exact = all(contracts[0][key] == contracts[1][key] for key in precision_keys)
    gate = (all(row['all_values_exact'] and row['all_values_finite'] and row['affine_exact'] and row['zooms_exact'] for row in images.values())
            and all(row['all_values_exact'] and row['all_values_finite'] for row in parameters.values())
            and fit_stats_exact and all(tables.values()) and precision_policy_exact)
    output = {'schema': 'fnit.gems.stage.comparison.v1', 'pipeline_completed': True,
              'comparison_timed': False, 'device': workers[0]['device'],
              'threads': workers[0]['threads'], 'images': images, 'parameters': parameters,
              'fit_solver_and_min_jacobian_exact': fit_stats_exact, 'tables_exact': tables,
              'precision_policy_exact': precision_policy_exact,
              'baseline_api_seconds': workers[0]['api_total_seconds'],
              'candidate_api_seconds': workers[1]['api_total_seconds'],
              'baseline_timings': workers[0]['timings'], 'candidate_timings': workers[1]['timings'],
              'baseline_runtime': contracts[0], 'candidate_runtime': contracts[1],
              'gate': 'Final labels/posteriors/geometry/volumes/vertices/Gaussians/objectives/solver stats exactly equal; precision policy unchanged',
              'gate_passed': gate}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'gate_passed': gate, 'device': workers[0]['device'],
                      'baseline_api_seconds': workers[0]['api_total_seconds'],
                      'candidate_api_seconds': workers[1]['api_total_seconds']}))
    if not gate:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
