"""Gate the production CPU ELU against the whole captured real first layer."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from sr_cpu_elu_prototype import _cpu_reference_elu
from sr_first_layer import digest, metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    if a.report.exists():p.error('preserve previous report')
    torch.set_num_threads(8)
    raw=np.load(a.reference_dir/'raw.npy',mmap_mode='r')
    expected=np.load(a.reference_dir/'elu.npy',mmap_mode='r')
    values=[]
    elapsed=0
    with torch.inference_mode():
        for index in range(raw.shape[2]):
            x=torch.from_numpy(np.ascontiguousarray(raw[:,:,index:index+1]))
            started=time.perf_counter()
            y=_cpu_reference_elu(x).numpy()
            elapsed+=time.perf_counter()-started
            values.append(metrics(expected[:,:,index:index+1],y))
    count=sum(r['count'] for r in values)
    results={'count':count,'different':sum(r['different'] for r in values),
             'max_abs':max(r['max_abs'] for r in values),
             'mae':sum(r['mae']*r['count'] for r in values)/count,
             'rmse':float(np.sqrt(sum(r['rmse']**2*r['count'] for r in values)/count))}
    report={'schema':'fnit.synthsr.cpu_elu.real.whole_first_layer.v1',
            'driver_sha256':digest(__file__),
            'model_sha256':digest(__import__('fnit.synthsr.model',fromlist=['']).__file__),
            'input_shape':list(raw.shape),'raw_sha256':digest(a.reference_dir/'raw.npy'),
            'elu_sha256':digest(a.reference_dir/'elu.npy'),
            'metrics':results,'arithmetic_seconds':elapsed,
            'timing_scope':'plane-wise whole first ELU gate, excludes reference array IO/comparison; not whole CNN timing',
            'pass':results['different']==0}
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
