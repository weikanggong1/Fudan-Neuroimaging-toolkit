"""Real first-layer probe of Eigen's published FP32 exponential polynomial.

Formula reference: Eigen GenericPacketMathFunctions.h, pexp_float.
This diagnostic evaluates separately rounded multiply/add and simulated FP32
FMA using FP64 intermediates. It does not modify production SynthSR.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sr_first_layer import digest, metrics


def polynomial_elu(value, fused):
    x=value.clamp(-88.3762626647949,0)
    def ma(first,second,last):
        if fused:return (first.double()*second.double()+torch.as_tensor(last,dtype=torch.float32).double()).float()
        return first*second+last
    exponent=torch.floor(ma(x,torch.full_like(x,1.44269504088896341),.5))
    if fused:
        remainder=ma(exponent,torch.full_like(x,-0.6931471805599453),x)
    else:
        remainder=(x-exponent*np.float32(.693359375))-exponent*np.float32(-2.12194440e-4)
    coefficients=(1.9875691500e-4,1.3981999507e-3,8.3334519073e-3,
                  4.1665795894e-2,1.6666665459e-1,5.0000001201e-1)
    y=torch.full_like(x,coefficients[0])
    for c in coefficients[1:]:y=ma(y,remainder,c)
    y=ma(y,remainder*remainder,remainder)+1
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
    report={'schema':'fnit.synthsr.elu.polynomial.real.v1','driver_sha256':digest(__file__),'results':{}}
    # A middle plane uses every channel and real brain/background intensities.
    # Full-volume gating follows only after this targeted first difference.
    index=raw.shape[2]//2
    x=torch.from_numpy(np.ascontiguousarray(raw[:,:,index:index+1]))
    y=np.asarray(expected[:,:,index:index+1])
    for fused in (False,True):
        report['results']['fma' if fused else 'separate_rounding']=metrics(y,polynomial_elu(x,fused).numpy())
    report['subset']={'axis':2,'index':index,'shape':list(x.shape),'scope':'one actual plane, all 24 channels; not complete network'}
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
