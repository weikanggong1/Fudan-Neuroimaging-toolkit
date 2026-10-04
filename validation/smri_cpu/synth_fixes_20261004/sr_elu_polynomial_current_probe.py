"""Probe the exponential polynomial in the installed TF 2.13 Eigen headers.

The input is a plane of the actual captured full-volume first convolution.
No original software is imported and no production operator is changed.
The constants and operation order are documented by Eigen pexp_float; this
file independently expresses the arithmetic with PyTorch primitives.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from sr_first_layer import digest, metrics


def polynomial_elu(value, mode):
    x=value.clamp(max=0)
    def ma(first,second,last):
        second=torch.as_tensor(second,dtype=torch.float32)
        last=torch.as_tensor(last,dtype=torch.float32)
        if mode=='fma':return (first.double()*second.double()+last.double()).float()
        if mode=='addcmul':return torch.addcmul(last,first,second)
        return first*second+last
    exponent=torch.floor(ma(x,1.44269504088896341,.5))
    remainder=ma(exponent,-.693359375,x)
    remainder=ma(exponent,2.12194440e-4,remainder)
    square=remainder*remainder
    even=ma(square,1.37449637986719608306884765625e-3,4.166965186595916748046875e-2)
    odd=ma(square,8.36894474923610687255859375e-3,.16666518151760101318359375)
    even=ma(square,even,.49999988079071044921875)
    low=remainder+1
    y=ma(remainder,odd,even)
    y=ma(square,y,low)
    scale=((exponent.int()+127)<<23).view(torch.float32)
    return torch.where(value>0,value,y*scale-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    if a.report.exists():p.error('preserve previous report')
    torch.set_num_threads(8)
    raw=np.load(a.reference_dir/'raw.npy',mmap_mode='r')
    expected=np.load(a.reference_dir/'elu.npy',mmap_mode='r')
    index=raw.shape[2]//2
    x=torch.from_numpy(np.ascontiguousarray(raw[:,:,index:index+1]))
    y=np.asarray(expected[:,:,index:index+1])
    report={'schema':'fnit.synthsr.elu.installed_polynomial.real.v1',
            'driver_sha256':digest(__file__),'subset':{'axis':2,'index':index,'shape':list(x.shape)},
            'input_range':[float(x.min()),float(x.max())],'results':{}}
    for mode in ('separate','fma','addcmul'):
        started=time.perf_counter()
        out=polynomial_elu(x,mode)
        report['results'][mode]={'metrics':metrics(y,out.numpy()),'seconds':time.perf_counter()-started}
    report['results']['float64_exp_rounded_before_subtract']={'metrics':metrics(y,torch.where(x>0,x,torch.exp(x.double()).float()-1).numpy())}
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
