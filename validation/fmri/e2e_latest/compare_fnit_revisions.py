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


def gifti_header_pair(left, right):
    # Workbench records the output command and working directory separately
    # from the scientific geometry, time metadata and array interpretation.
    provenance_keys = {'ParentProvenance', 'Provenance', 'WorkingDirectory'}
    left_meta, right_meta = dict(left.meta), dict(right.meta)
    metadata_equal = ({key: value for key, value in left_meta.items() if key not in provenance_keys} ==
                      {key: value for key, value in right_meta.items() if key not in provenance_keys})
    array_headers_equal = len(left.darrays) == len(right.darrays)
    for a, b in zip(left.darrays, right.darrays):
        fields = ('intent', 'datatype', 'encoding', 'endian', 'ind_ord', 'ext_fname', 'ext_offset')
        array_headers_equal = array_headers_equal and all(getattr(a, key) == getattr(b, key) for key in fields)
        array_headers_equal = array_headers_equal and dict(a.meta) == dict(b.meta)
        array_headers_equal = array_headers_equal and (
            a.coordsys.dataspace == b.coordsys.dataspace and
            a.coordsys.xformspace == b.coordsys.xformspace and
            np.array_equal(a.coordsys.xform, b.coordsys.xform, equal_nan=True))
    return {'scientific_header_equal': bool(metadata_equal and array_headers_equal and
                                            left.version == right.version and
                                            left.labeltable.to_xml() == right.labeltable.to_xml()),
            'non_scientific_image_metadata_different_keys': sorted(key for key in provenance_keys
                if left_meta.get(key) != right_meta.get(key))}


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
            row.update(gifti_header_pair(a, b))
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


def require_completed_identity(runs, files):
    """同时核对真实源码快照、完整API检查和实际正式输出文件身份。"""
    input_roles = {'bold', 'sbref', 't1w', 'mni_template', 'mni_brain_mask', 'synthstrip_weights'}
    configuration_roles = {
        'anatomical_cache', 'aroma_mode', 'bandpass', 'batch_size', 'bbr_execution',
        'brain_extraction', 'cudnn_allow_tf32', 'device', 'fast_config', 'fnirt_config',
        'fnirt_execution', 'global_signal', 'highpass_cutoff_seconds', 'ica_max_iter',
        'ica_n_components', 'matmul_allow_tf32', 'mni_interpolation', 'motion_algorithm',
        'motion_iterations', 'motion_model', 'motion_output', 'n_splits',
        'preproc_coordinate_precision', 'preproc_interpolation', 'random_state',
        'registration_backend', 'regress_csf', 'regress_motion', 'regress_wm',
        'reuse_anatomical', 'slice_time_reference', 'slice_timing', 'surface_cpu_thread_budget',
        'surface_msm_execution', 'surface_parallel', 'surface_signal', 'weights'}
    for run, mapping in zip(runs, files):
        before, after = run.get('source_sha256_before'), run.get('source_sha256_after')
        if (run.get('source_unchanged_during_run') is not True or
                not isinstance(before, dict) or not before or before != after):
            raise ValueError('each run must attest unchanged nonempty actual runtime source snapshots')
        inputs, configuration = run.get('input_sha256'), run.get('configuration')
        if (not isinstance(inputs, dict) or not input_roles.issubset(inputs) or
                not isinstance(configuration, dict) or not configuration_roles.issubset(configuration)):
            raise ValueError('each run must record complete inputs and scientific configuration')
        if any(not isinstance(inputs[key], str) or len(inputs[key]) != 64 or
               any(character not in '0123456789abcdef' for character in inputs[key]) for key in input_roles):
            raise ValueError('each complete input must have a recorded SHA256 identity')
        frames = run['data']['bold_shape'][-1]
        if frames < 2 or run['algorithm']['volume'].get('ica_converged') is not True:
            raise ValueError('the complete pipeline success checks are absent')
        checks = run['checks']
        for role in ('clean_native', 'clean_mni', 'preproc_t1w', 'preproc_mni'):
            row = checks['volume'][role]
            if (row.get('nonfinite_values') != 0 or row.get('dtype') != 'float32' or
                    row.get('shape', [0])[-1] != frames or row.get('time_unit') != 'sec'):
                raise ValueError('a complete volume API output did not pass its saved checks')
            if sha256(mapping['volume'][role]) != row['sha256']:
                raise ValueError('a formal volume output does not match the completed API record')
        surface = checks['surface']
        if surface.get('all_finite') is not True or surface['cifti_shape'][0] != frames:
            raise ValueError('a complete surface API output did not pass its saved checks')
        expected = {'dtseries': surface['cifti_sha256'],
                    'left': surface['hemispheres']['L']['sha256'],
                    'right': surface['hemispheres']['R']['sha256']}
        for hemisphere in ('L', 'R'):
            row = surface['hemispheres'][hemisphere]
            if row.get('all_finite') is not True or row.get('frames') != frames:
                raise ValueError('a complete surface hemisphere has not passed its saved checks')
        for role, digest in expected.items():
            if sha256(mapping['surface'][role]) != digest:
                raise ValueError('a formal surface output does not match the completed API record')
    if runs[0]['input_sha256'] != runs[1]['input_sha256'] or runs[0]['configuration'] != runs[1]['configuration']:
        raise ValueError('compare the same inputs, resources and scientific parameters')


def scientific_pairs(files):
    selected = [{role: Path(path) for role, path in mapping['volume'].items()
                 if role != 'motion_matrices' and Path(path).name.endswith(
                     ('.nii.gz', '.npy', '.mat', '.par'))} for mapping in files]
    if selected[0].keys() != selected[1].keys() or not selected[0]:
        raise ValueError('both complete runs must contain the same scientific volume roles')
    pairs = {'volume_' + role: (path, selected[1][role]) for role, path in selected[0].items()}
    for role in ('left', 'right', 'dtseries'):
        pairs['surface_' + role] = tuple(Path(value['surface'][role]) for value in files)
    if any(len(value['surface']['registered_spheres']) != 2 for value in files):
        raise ValueError('both complete runs must contain two registered spheres')
    for index in range(2):
        pairs['surface_registered_sphere_' + ('L' if index == 0 else 'R')] = tuple(
            Path(value['surface']['registered_spheres'][index]) for value in files)
    if any(not path.is_file() for pair in pairs.values() for path in pair):
        raise FileNotFoundError('a complete scientific output or captured numerical intermediate is absent')
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-report', type=Path, required=True)
    parser.add_argument('--after-report', type=Path, required=True)
    parser.add_argument('--before-files', type=Path, required=True)
    parser.add_argument('--after-files', type=Path, required=True)
    parser.add_argument('--report-out', type=Path, required=True)
    parser.add_argument('--expected-scientific-file-count', type=int)
    args = parser.parse_args()
    started = time.perf_counter()
    runs = [json.loads(path.read_text()) for path in (args.before_report, args.after_report)]
    files = [json.loads(path.read_text()) for path in (args.before_files, args.after_files)]
    require_completed_identity(runs, files)
    pairs = scientific_pairs(files)
    if args.expected_scientific_file_count is not None and len(pairs) != args.expected_scientific_file_count:
        raise ValueError('the scientific output count differs from the declared acceptance scope')
    results = {role: file_pair(*pair) for role, pair in pairs.items()}
    report = {'schema_version': 1, 'before_revision': runs[0]['source_revision'],
              'after_revision': runs[1]['source_revision'], 'same_inputs_resources_configuration': True,
              'actual_runtime_source_snapshots_unchanged': True,
              'formal_output_files_match_completed_api_checks': True,
              'scientific_file_count': len(results),
              'runtime_source_inventory_sha256': {label: hashlib.sha256(json.dumps(run['source_sha256_before'],
                  sort_keys=True, separators=(',', ':')).encode()).hexdigest() for label, run in zip(('before', 'after'), runs)},
              'execution_report_sha256': {'before': sha256(args.before_report), 'after': sha256(args.after_report)},
              'files': results, 'all_decoded_bits_equal': all(row['decoded_bits_equal'] for row in results.values()),
              'all_scientific_headers_and_axes_equal': all(row.get('affine_equal', True) and
                  row.get('scientific_header_equal', True) and row.get('scientific_axes_equal', True)
                  for row in results.values()),
              'comparison_script_sha256': sha256(__file__),
              'validation_seconds_excluded_from_pipeline': time.perf_counter() - started,
              'scope': 'Complete scientific outputs and captured numerical intermediates of two successful continuous runs; all frames, no resampling or fitting.',
              'privacy': 'Only anonymous role names, hashes and scalar comparisons; native geometry and arrays remain private.'}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'files': len(results), 'all_decoded_bits_equal': report['all_decoded_bits_equal'],
                      'all_scientific_headers_and_axes_equal': report['all_scientific_headers_and_axes_equal']}), flush=True)


if __name__ == '__main__':
    main()
