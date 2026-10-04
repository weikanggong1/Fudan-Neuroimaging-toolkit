"""Whole-grid comparison and chunk checks for grouped real apply outputs."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from compare_registration import image_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    candidate = Path(args.candidate)
    reference = Path(args.reference)
    rows = []
    for case in plan['cases']:
        result = image_pair(candidate/(case['id']+'.nii.gz'),reference/(case['id']+'.nii.gz'))
        result['case'] = {key:value for key,value in case.items()
                          if key not in ('image','reference','affine','warp','output_mask','pre_affine_pull_ras')}
        result['acceptance'] = result['whole_grid']['exact_equal'] if case['method']=='nearest' else (
            result['whole_grid']['rmse']==0 or
            (result['whole_grid']['nrmse_reference_p99_minus_p1'] is not None and
             result['whole_grid']['nrmse_reference_p99_minus_p1']<=1e-3))
        rows.append(result)
    checks = {}
    if all((candidate/(name+'.nii.gz')).exists() for name in ('epi4d_chunk1','epi4d_chunk2','epi4d_auto')):
        baseline = nib.load(candidate/'epi4d_auto.nii.gz')
        data = np.asanyarray(baseline.dataobj)
        for name in ('epi4d_chunk1','epi4d_chunk2'):
            other = nib.load(candidate/(name+'.nii.gz'))
            checks[name] = {'array_equal':bool(np.array_equal(data,np.asanyarray(other.dataobj))),
                            'affine_equal':bool(np.array_equal(baseline.affine,other.affine)),
                            'shape_equal':baseline.shape==other.shape,
                            'tr_equal':float(baseline.header['pixdim'][4])==float(other.header['pixdim'][4])}
    Path(args.output).write_text(json.dumps({'scope':'same common transforms, actual full real input grid; 2 acquired DWI frames, not a full time-series throughput claim','rows':rows,'candidate_frame_chunk_checks':checks},indent=2)+'\n')


if __name__=='__main__':
    main()
