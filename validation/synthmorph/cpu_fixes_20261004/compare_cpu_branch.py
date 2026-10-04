"""Compare unchanged CPU modes to accepted real CLI outputs, without inference."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from fnit._transforms import load_lta
from build_report import gate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_comparison(left_path, right_path):
    left, right = [nib.load(path) for path in (left_path, right_path)]
    arrays = [np.ascontiguousarray(np.asanyarray(image.dataobj)) for image in (left, right)]
    hashes = [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays]
    return {'array_sha256': hashes, 'array_equal': hashes[0] == hashes[1],
            'binary_header_equal': left.header.binaryblock == right.header.binaryblock,
            'extensions_equal': left.header.extensions == right.header.extensions,
            'affine_equal': bool(np.array_equal(left.affine, right.affine)),
            'shape_equal': left.shape == right.shape, 'dtype_equal': arrays[0].dtype == arrays[1].dtype}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    base = Path(args.root) / 'runs/smri_cpu_20261004/remaining_20261004/morph'
    report = {'scope': 'complete real CPU rigid/affine/deform CLI outputs from v29 versus accepted modes; exact preservation transfers their fixed independent-reference gates',
              'worker_sha256': digest(__file__), 'modes': {}}
    for model in ('rigid', 'affine', 'deform'):
        accepted = base / ('nodecw7_rigid_v4' if model == 'rigid' else 'nodecw7_v2') / model
        candidate = base / 'nodecw7_accept_v29' / model
        for directory in (accepted, candidate):
            record = json.loads((directory / 'record.json').read_text())
            if record['status'] != 'complete' or record['returncode'] != 0:
                raise RuntimeError('CPU mode has no completed output: ' + model)
        rows = {name: image_comparison(accepted / (name + '.nii.gz'), candidate / (name + '.nii.gz'))
                for name in ('moved', 'fixed_moved')}
        for name in ('forward', 'inverse'):
            if model == 'deform':
                rows[name] = image_comparison(accepted / (name + '.nii.gz'), candidate / (name + '.nii.gz'))
            else:
                paths = [directory / (name + '.lta') for directory in (accepted, candidate)]
                left, right = [load_lta(path) for path in paths]
                rows[name] = {'serialized_sha256': [digest(path) for path in paths],
                              'world_matrix_equal': bool(np.array_equal(left.matrix, right.matrix)),
                              'source_shape_equal': left.source.shape == right.source.shape,
                              'target_shape_equal': left.target.shape == right.target.shape,
                              'source_affine_equal': bool(np.array_equal(left.source.affine, right.source.affine)),
                              'target_affine_equal': bool(np.array_equal(left.target.affine, right.target.affine))}
        comparison = json.loads((accepted / 'nodecw7_reference_comparison.private.json').read_text())
        oracle = gate(comparison, model)
        same = all(all(value for key, value in row.items() if key.endswith('_equal')) for row in rows.values())
        report['modes'][model] = {'outputs': rows, 'accepted_reference_comparison_sha256': digest(accepted / 'nodecw7_reference_comparison.private.json'),
                                  'accepted_reference_comparison': comparison, 'fixed_reference_gate': oracle,
                                  'all_gates_passed': same and oracle['all_gates_passed']}
    report['all_gates_passed'] = all(row['all_gates_passed'] for row in report['modes'].values())
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_gates_passed']:
        raise RuntimeError('unchanged CPU branch regression failed')


if __name__ == '__main__':
    main()
