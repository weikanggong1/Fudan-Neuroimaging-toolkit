"""Reapply saved rigid/affine matrices with final source, without running a CNN."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from fnit.synthmorph import apply_transform
from fnit._transforms import load_lta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('moving', 'fixed', 'rigid', 'affine', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    rows = {}
    for mode in ('rigid', 'affine'):
        folder = Path(getattr(args, mode))
        rows[mode] = {}
        for image_path, matrix_name, result_name in [
            (args.moving, 'forward', 'moved'),
            (args.fixed, 'inverse', 'fixed_moved'),
        ]:
            matrix_path = folder / (matrix_name + '.lta')
            expected = nib.load(folder / (result_name + '.nii.gz'))
            actual = apply_transform(image_path, load_lta(matrix_path), device='cpu')
            # LTA serialization retains enough decimal places for the exact
            # float32 coordinate path of these measured registrations.
            rows[mode][result_name] = {
                'array_equal': bool(np.array_equal(actual.dataobj, np.asanyarray(expected.dataobj))),
                'header_equal': actual.header.binaryblock == expected.header.binaryblock,
                'affine_equal': bool(np.array_equal(actual.affine, expected.affine)),
                'extensions_equal': actual.header.extensions == expected.header.extensions,
                'saved_matrix_sha256': hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            }
    report = {'scope': 'final-source public CPU affine application of saved measured matrices; no CNN',
              'rows': rows, 'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report['all_gates_passed'] = all(
        all(value for key, value in item.items() if key.endswith('_equal'))
        for mode in rows.values() for item in mode.values())
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_gates_passed']:
        raise RuntimeError('final source saved-affine replay failed')


if __name__ == '__main__':
    main()
