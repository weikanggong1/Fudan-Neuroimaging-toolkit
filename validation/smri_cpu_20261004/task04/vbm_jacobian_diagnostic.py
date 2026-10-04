"""Separate analytic-vs-dense Jacobian differences on the original FNIRT warp."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np


def metrics(a, b, selected):
    x, y = a[selected].astype(np.float64), b[selected].astype(np.float64)
    error = x-y
    return {'values': int(x.size), 'mae': float(np.abs(error).mean()),
            'rmse': float(np.sqrt(np.mean(error**2))),
            'maximum_absolute_error': float(np.abs(error).max()),
            'pearson_r': float(np.corrcoef(x, y)[0, 1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--template', required=True)
    parser.add_argument('--reference-mask', required=True)
    parser.add_argument('--candidate-jacobian', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    residual_path = args.output_dir / 'original_nonaffine_residual.nii.gz'
    command = [str(Path(os.environ['FSLDIR'])/'bin/fnirtfileutils'),
               '--in='+str(args.reference/'gm_coeff.nii.gz'), '--ref='+args.template,
               '--out='+str(residual_path), '--outformat=field']
    before = time.perf_counter()
    with (args.output_dir/'fnirtfileutils.log').open('wb') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    residual = np.asanyarray(nib.load(residual_path).dataobj).astype(np.float64)
    fixed = nib.load(args.template)
    if residual.shape != (*fixed.shape, 3):
        raise ValueError('original residual field does not match template grid')
    scaled_mm = np.diag(fixed.header.get_zooms()[:3]).astype(np.float64)
    if np.linalg.det(fixed.affine[:3, :3]) > 0:
        scaled_mm[0, 0] *= -1
    field = np.moveaxis(residual, -1, 0)
    gradient_voxel = np.stack(np.gradient(field, axis=(1, 2, 3), edge_order=1), axis=1)
    derivative = np.einsum('caxyz,ab->cbxyz', gradient_voxel, np.linalg.inv(scaled_mm))
    dense_jacobian = np.linalg.det(np.moveaxis(derivative, (0, 1), (-2, -1))+np.eye(3))
    native_jacobian = np.asanyarray(nib.load(args.reference/'T1_GM_JAC_nl.nii.gz').dataobj)
    candidate = np.asanyarray(nib.load(args.candidate_jacobian).dataobj)
    mask = ((np.asanyarray(nib.load(args.reference_mask).dataobj)>0) &
            (np.asanyarray(fixed.dataobj)>0))
    regions = {'whole_grid': np.ones(fixed.shape, dtype=bool), 'template_brain': mask}
    report = {'scope': 'same original FNIRT spline field; analytic native --jout versus independent dense finite differences; no registration rerun',
              'fnirtfileutils_argv': command,
              'seconds': time.perf_counter()-before,
              'residual_intent_code': int(nib.load(residual_path).header['intent_code']),
              'original_dense_vs_original_analytic': {name: metrics(dense_jacobian, native_jacobian, selected)
                                                       for name, selected in regions.items()},
              'candidate_dense_vs_original_dense': {name: metrics(candidate, dense_jacobian, selected)
                                                      for name, selected in regions.items()}}
    (args.output_dir/'report.private.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
