"""Verify saved real SynthMorph ABBA scalar bit patterns, including signed zero."""
import argparse
import json
from pathlib import Path
import hashlib
import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
report = {'declared_tolerance': {'scalar_bitwise_neq': 0, 'affine_bitwise_neq': 0}, 'comparisons': {}}
for filename in ('forward.nii.gz', 'tian_s1.nii.gz', 'tian_s4.nii.gz'):
    reference = nib.load(args.report_dir / '0_baseline' / filename)
    a = np.asarray(reference.dataobj)
    for index, arm in enumerate(('candidate', 'candidate', 'baseline'), 1):
        current = nib.load(args.report_dir / f'{index}_{arm}' / filename)
        b = np.asarray(current.dataobj)
        same = a.shape == b.shape and a.dtype == b.dtype
        words = np.dtype(f'u{a.dtype.itemsize}')
        scalar_neq = int(np.count_nonzero(a.view(words) != b.view(words))) if same else None
        affine_neq = int(np.count_nonzero(reference.affine.view(np.uint64) != current.affine.view(np.uint64)))
        header_equal = reference.header.binaryblock == current.header.binaryblock
        extensions_equal = [(e.get_code(), e.content) for e in reference.header.extensions] == [(e.get_code(), e.content) for e in current.header.extensions]
        report['comparisons'][f'baseline0_vs_{index}_{arm}/{filename}'] = {
            'shape_dtype_equal': same, 'scalar_values': int(a.size), 'scalar_bitwise_neq': scalar_neq,
            'affine_bitwise_neq': affine_neq, 'header_equal': header_equal, 'extensions_equal': extensions_equal,
            'passed': same and scalar_neq == 0 and affine_neq == 0 and header_equal and extensions_equal}
report['passed'] = all(row['passed'] for row in report['comparisons'].values())
report['benchmark_report_sha256'] = hashlib.sha256((args.report_dir / 'report.json').read_bytes()).hexdigest()
args.output.write_text(json.dumps(report, indent=2)+'\n')
if not report['passed']: raise SystemExit('bitwise output verification failed')
