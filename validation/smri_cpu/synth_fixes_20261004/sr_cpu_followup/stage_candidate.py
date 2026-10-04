"""Strict FP32 fused-ELU and actual-BN-factor stage gates on real saved tensors."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from numba import set_num_threads
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sr_first_layer import digest, metrics
from elu_numba import _elu_flat, elu_numpy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first-dir',type=Path,required=True)
    p.add_argument('--second-dir',type=Path,required=True)
    p.add_argument('--factors',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('keep previous attempt')
    a.output_dir.mkdir(parents=True)
    torch.set_num_threads(8);set_num_threads(8)
    report={'driver_sha256':digest(__file__),'elu_driver_sha256':digest(Path(__file__).with_name('elu_numba.py')),
            'first_raw_sha256':digest(a.first_dir/'raw.npy'),'first_elu_sha256':digest(a.first_dir/'elu.npy'),
            'bn_reference_sha256':digest(a.second_dir/'bn.npy'),'factor_sha256':digest(a.factors),
            'fastmath':False,'threads':8,'scope':'full real first ELU and first BN stage only; no complete CNN speed claim'}
    _elu_flat(np.array([-.5],dtype=np.float32),np.empty(1,dtype=np.float32))
    raw=np.load(a.first_dir/'raw.npy',mmap_mode='r');reference=np.load(a.first_dir/'elu.npy',mmap_mode='r')
    started=time.perf_counter();actual=elu_numpy(raw)
    report['first_elu_seconds']=time.perf_counter()-started
    report['first_elu']=metrics(reference,actual)
    (a.output_dir/'elu_gate.public.json').write_text(json.dumps(report,indent=2)+'\n')
    del raw,reference,actual
    factors=np.load(a.factors)
    mean=torch.from_numpy(factors['mean'].copy());beta=torch.from_numpy(factors['beta'].copy());gamma=torch.from_numpy(factors['gamma'].copy())
    variance=torch.from_numpy(factors['variance'].copy())
    inv=torch.from_numpy(factors['inverse'].copy())
    report['inverse_factors']={name:metrics(factors['inverse'].reshape(1,1,-1),value.numpy().reshape(1,1,-1)) for name,value in [('torch_rsqrt',torch.rsqrt(variance+np.float32(.001))),('one_over_sqrt',1/torch.sqrt(variance+np.float32(.001)))]}
    exact=np.load(a.second_dir/'elu.npy',mmap_mode='r');bn=np.load(a.second_dir/'bn.npy',mmap_mode='r')
    x=torch.from_numpy(np.ascontiguousarray(exact[:,:,exact.shape[2]//2:exact.shape[2]//2+1]))
    ref=bn[:,:,bn.shape[2]//2:bn.shape[2]//2+1]
    shape=(1,-1,1,1,1);mean=mean.view(shape);beta=beta.view(shape);gamma=gamma.view(shape);inv=inv.view(shape)
    scale=gamma*inv;offset=beta-mean*scale
    with torch.no_grad():
        operations={
          'subtract_inv_gamma':lambda v:((v-mean)*inv)*gamma+beta,
          'subtract_scale':lambda v:(v-mean)*scale+beta,
          'linear_separate':lambda v:v*scale+offset,
          'linear_output_fma':lambda v:torch.addcmul(offset,v,scale),
          'linear_offset_output_fma':lambda v:torch.addcmul(torch.addcmul(beta,-mean,scale),v,scale),
          'subtract_scale_fma':lambda v:torch.addcmul(beta,v-mean,scale)}
        report['bn_plane']={name:metrics(ref,operation(x).numpy()) for name,operation in operations.items()}
        passed=[name for name,row in report['bn_plane'].items() if row['different']==0]
        report['bn_full']={}
        for name in passed:
            actual=np.empty(exact.shape,dtype=np.float32)
            started=time.perf_counter()
            for plane in range(exact.shape[2]):
                v=torch.from_numpy(np.ascontiguousarray(exact[:,:,plane:plane+1]))
                actual[:,:,plane:plane+1]=operations[name](v).numpy()
            row=metrics(bn,actual);row['arithmetic_seconds']=time.perf_counter()-started
            report['bn_full'][name]=row
            del actual
    report['status']='complete'
    (a.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
