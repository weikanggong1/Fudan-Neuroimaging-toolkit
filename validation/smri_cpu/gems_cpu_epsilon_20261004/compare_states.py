"""Retain changed CPU trajectories and verify final storage and geometry."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'helper', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh result path')
    spec = importlib.util.spec_from_file_location('existing_state_contract', args.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    images = {}
    names = ['subregions_native.nii.gz'] + [str(p.relative_to(args.baseline)) for p in sorted((args.baseline / 'highres').glob('*.nii.gz'))]
    for name in names:
        first, second = (nib.load(str(directory / name)) for directory in (args.baseline, args.candidate))
        values = helper.array_contract(np.asanyarray(first.dataobj), np.asanyarray(second.dataobj))
        values['affine_exact'] = bool(np.array_equal(first.affine, second.affine))
        values['zooms_exact'] = first.header.get_zooms() == second.header.get_zooms()
        images[name] = values
    parameters = {}
    with np.load(args.baseline / 'fit_parameters.private.npz') as first, np.load(args.candidate / 'fit_parameters.private.npz') as second:
        assert first.files == second.files
        for name in first.files:
            parameters[name] = helper.array_contract(first[name], second[name])
    output = {'schema': 'fnit.gems.cpu-epsilon.state.v1',
              'scope': 'CPU formulas and optimization trajectories may change; exact old values are recorded rather than required',
              'images': images, 'parameters': parameters,
              'trajectories': {}, 'geometry_dtype_finite_gate_passed': all(
                  row['shape_equal'] and row['dtype_equal'] and row['all_values_finite'] and
                  row['affine_exact'] and row['zooms_exact'] for row in images.values()),
              'helper_sha256': hashlib.sha256(args.helper.read_bytes()).hexdigest()}
    for tag, directory in [('baseline', args.baseline), ('candidate', args.candidate)]:
        worker = json.loads((directory / 'worker.json').read_text())
        report = json.loads((directory / 'report.json').read_text())
        fits = json.loads((directory / 'fit_contract.private.json').read_text())
        output['trajectories'][tag] = {'worker': worker, 'fits': fits,
                                     'timings': report['timings'], 'volumes': report['volumes'],
                                     'initialization': report['initialization'],
                                     'saved_output_sha256': {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in sorted(directory.rglob('*')) if path.is_file()}}
        record = directory.parent / 'record.json'
        if record.exists():
            output['trajectories'][tag]['process_record'] = json.loads(record.read_text())
    output['parameter_storage_finite_gate_passed'] = all(
        row['dtype_equal'] and row['all_values_finite'] for row in parameters.values())
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'geometry_dtype_finite': output['geometry_dtype_finite_gate_passed'],
                      'parameter_storage_finite': output['parameter_storage_finite_gate_passed'],
                      'images': {name: row['different_values'] for name, row in images.items()},
                      'parameters': {name: row.get('different_values') for name, row in parameters.items()}}))


if __name__ == '__main__':
    main()
