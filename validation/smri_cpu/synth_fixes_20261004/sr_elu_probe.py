"""Test CPU ELU formulas at the actual first-layer divergence."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sr_first_layer import metrics, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    if a.report.exists():
        p.error('preserve prior report')
    torch.set_num_threads(8)
    raw = np.load(a.reference_dir/'raw.npy', mmap_mode='r')
    expected = np.load(a.reference_dir/'elu.npy', mmap_mode='r')
    result = {'schema':'fnit.synthsr.elu.first_divergence.real.v1',
              'driver_sha256':digest(__file__),'raw_sha256':digest(a.reference_dir/'raw.npy'),
              'elu_sha256':digest(a.reference_dir/'elu.npy'),'results':{}}
    modes=('elu', 'exp_minus_one', 'expm1', 'numpy_exp_minus_one')
    for mode in modes:
        count=different=0; total=square=maximum=0.
        for i in range(raw.shape[2]):
            x=np.ascontiguousarray(raw[:,:,i:i+1])
            y=np.asarray(expected[:,:,i:i+1])
            t=torch.from_numpy(x)
            if mode=='elu': out=torch.nn.functional.elu(t).numpy()
            elif mode=='exp_minus_one': out=torch.where(t>0,t,torch.exp(t)-1).numpy()
            elif mode=='expm1': out=torch.where(t>0,t,torch.expm1(t)).numpy()
            else: out=np.where(x>0,x,np.exp(x)-np.float32(1))
            row=metrics(y,out)
            count+=row['count'];different+=row['different'];total+=row['mae']*row['count'];square+=row['rmse']**2*row['count'];maximum=max(maximum,row['max_abs'])
        result['results'][mode]={'count':count,'different':different,'mae':total/count,'rmse':float(np.sqrt(square/count)),'max_abs':maximum}
    a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
