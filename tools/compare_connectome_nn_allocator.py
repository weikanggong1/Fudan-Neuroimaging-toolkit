"""Compare real saved warp and labels across CUDA allocator profiles (exact tolerance)."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cached', type=Path, required=True)
parser.add_argument('--uncached', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
report = {'declared_tolerance': {'label_neq': 0, 'warp_max_abs': 0}, 'files': {}}
for filename in ('mni_to_t1.nii.gz', 'tian_s1_cpu.nii.gz', 'tian_s4_cpu.nii.gz'):
    left, right = (nib.load(folder / filename) for folder in (args.cached, args.uncached))
    a, b = np.asarray(left.dataobj), np.asarray(right.dataobj)
    geometry_equal = a.shape == b.shape and np.array_equal(left.affine, right.affine)
    dtype_equal = a.dtype == b.dtype
    result = {'shape': list(a.shape), 'dtype': str(a.dtype), 'geometry_equal': geometry_equal, 'dtype_equal': dtype_equal}
    if geometry_equal:
        diff = a.astype(np.float64) - b.astype(np.float64)
        absolute = np.abs(diff)
        result.update(neq=int(np.count_nonzero(a != b)), max_abs=float(absolute.max()),
                      p99_abs=float(np.percentile(absolute, 99)), rmse=float(np.sqrt(np.mean(diff * diff))))
        result['passed'] = dtype_equal and result['neq'] == 0
    else:
        result['passed'] = False
    report['files'][filename] = result
report['passed'] = all(x['passed'] for x in report['files'].values())
args.output.write_text(json.dumps(report, indent=2) + '\n')
if not report['passed']:
    raise SystemExit('allocator output comparison failed exact tolerance')
