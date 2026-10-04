"""Compare already saved real WMH results without inference or resampling."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--candidate-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    if a.report.exists():p.error('preserve existing result')
    report={'schema':'fnit.wmh.saved_result_comparison.v1','results':{}}
    for name in ('seg','lesion'):
        first=nib.load(str(a.reference_dir/(name+'.nii.gz')))
        second=nib.load(str(a.candidate_dir/(name+'.nii.gz')))
        x,y=first.get_fdata(),second.get_fdata()
        if x.shape!=y.shape:raise ValueError('grids differ; no comparison resampling allowed')
        d=np.abs(x-y)
        report['results'][name]={'different':int(np.count_nonzero(d)),'max_abs':float(d.max()),
            'mae':float(d.mean()),'rmse':float(np.sqrt(np.square(d).mean())),
            'shape':list(x.shape),'affine_exact':bool(np.array_equal(first.affine,second.affine)),
            'header_exact':bool(first.header==second.header)}
        if name=='seg':report['results'][name]['label_dice']={str(int(k)):float(2*np.count_nonzero((x==k)&(y==k))/(np.count_nonzero(x==k)+np.count_nonzero(y==k))) for k in np.union1d(x,y)}
    import csv
    values=[]
    for path in (a.reference_dir/'volumes.csv',a.candidate_dir/'volumes.csv'):
        with path.open() as stream:rows=list(csv.reader(stream))
        values.append(np.asarray([float(v) for v in rows[1][1:]]))
    d=np.abs(values[0]-values[1]);report['results']['csv']={'columns':len(d),'different':int(np.count_nonzero(d)),'max_abs':float(d.max()),'signed_deltas':(values[1]-values[0]).tolist()}
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
