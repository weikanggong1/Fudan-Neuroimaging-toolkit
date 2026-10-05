"""Verify self-contained strict FP32 rsqrt/BN against actual reference factors."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from numba import set_num_threads
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sr_first_layer import digest,metrics
from bn_numba import inverse_factors,batch_norm_torch
from fnit.synthsr.model import SynthSRUNet,load_h5_weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for argument in ('weights','factors','second-dir','output-dir'):
        p.add_argument('--'+argument,type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('preserve prior output')
    a.output_dir.mkdir(parents=True)
    torch.set_num_threads(8);set_num_threads(8)
    assert a.weights.stat().st_size==106163752
    assert digest(a.weights)=='a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b'
    factor=np.load(a.factors)
    candidate=inverse_factors(factor['variance'],np.float32(.001))
    report={'driver_sha256':digest(__file__),'bn_driver_sha256':digest(Path(__file__).with_name('bn_numba.py')),
            'factor_sha256':digest(a.factors),'fastmath':False,
            'inverse':metrics(factor['inverse'].reshape(1,1,-1),candidate.reshape(1,1,-1))}
    if report['inverse']['different']:
        report['status']='inverse_gate_failed'
    else:
        model=load_h5_weights(SynthSRUNet(),a.weights).eval()
        source=np.load(a.second_dir/'elu.npy',mmap_mode='r')
        reference=np.load(a.second_dir/'bn.npy',mmap_mode='r')
        x=torch.from_numpy(np.ascontiguousarray(source))
        with torch.no_grad():
            start=time.perf_counter();actual=batch_norm_torch(x,model.down_bn[0])
            report['bn_seconds']=time.perf_counter()-start
        report['bn']=metrics(reference,actual.numpy())
        report['status']='complete' if report['bn']['different']==0 else 'bn_gate_failed'
    (a.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
