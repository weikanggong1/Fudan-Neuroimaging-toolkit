"""Read complete affine GPU arrays and headers from the frozen paired runs."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    baseline, candidate = Path(args.baseline), Path(args.candidate)
    records = [json.loads((folder / 'report.private.json').read_text())
               for folder in (baseline, candidate)]
    arrays = {name: {'baseline_sha256': digest(baseline / (name + '.npy')),
                     'candidate_sha256': digest(candidate / (name + '.npy'))}
              for name in ('forward', 'inverse', 'moved', 'fixed_moved')}
    for row in arrays.values():
        row['exact_equal'] = row['baseline_sha256'] == row['candidate_sha256']
    headers = {}
    for name in ('moved', 'fixed_moved'):
        left, right = [nib.load(folder / (name + '.nii.gz'))
                       for folder in (baseline, candidate)]
        headers[name] = {'binary_header_equal': left.header.binaryblock == right.header.binaryblock,
                         'extensions_equal': left.header.extensions == right.header.extensions,
                         'affine_equal': bool(np.array_equal(left.affine, right.affine)),
                         'shape_equal': left.shape == right.shape}
    report = {'scope': 'full real affine GPU old/new arrays and complete saved headers; shared GPU timing observations',
              'arrays': arrays, 'headers': headers,
              'same_input_sha256': records[0]['input_sha256'] == records[1]['input_sha256'],
              'source_sha256': {'baseline': records[0]['source_sha256'], 'candidate': records[1]['source_sha256']},
              'api_seconds': [row['api_seconds'] for row in records],
              'peak_cuda_reserved_bytes': [row['peak_cuda_reserved_bytes'] for row in records],
              'worker_sha256': digest(__file__)}
    report['all_gates_passed'] = (report['same_input_sha256'] and
        all(row['exact_equal'] for row in arrays.values()) and
        all(all(row.values()) for row in headers.values()) and
        all(value < 20_000_000_000 for value in report['peak_cuda_reserved_bytes']))
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_gates_passed']:
        raise RuntimeError('affine GPU regression failed')


if __name__ == '__main__':
    main()
