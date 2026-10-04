"""Complete real SynthSR CPU inference and the existing floating/byte gates.

All image arrays remain in the server run directory. Accuracy is compared on
the saved output grids without resampling or a changed tolerance. API/network
timings exclude saving and comparison; cold CLI timing is measured separately.
"""

import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.synthsr import SynthSR
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sr_first_layer import digest, metrics
from numba import set_num_threads
from cpu_inference_numba import cpu_inference
from elu_numba import _elu_flat
from bn_numba import inverse_factors,_affine_channels,_affine_interleaved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-revision',required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('preserve prior runs')
    a.output_dir.mkdir(parents=True)
    if a.weights.stat().st_size!=106163752 or digest(a.weights)!='a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b':
        raise ValueError('SynthSR checkpoint differs from fixed size/SHA manifest')
    torch.set_num_threads(8)
    report={'schema':'fnit.synthsr.real.cpu.elu_fix.v1','source_revision':a.source_revision,
            'driver_sha256':digest(__file__),'input_sha256':digest(a.input),
            'weight_sha256':digest(a.weights),'torch_version':torch.__version__,
            'model_sha256':digest(__import__('fnit.synthsr.model',fromlist=['']).__file__),
            'network_calls':[],'fixed_float_gate':{'rtol':1e-5,'atol':1e-3},
            'timing_scope':'constructor/API/save separately; networks nested in API; excludes comparison'}
    started=time.perf_counter();model=SynthSR(weights=a.weights,device='cpu',threads=8)
    report['constructor_seconds']=time.perf_counter()-started
    model.model.to(memory_format=torch.channels_last_3d)
    set_num_threads(8)
    warm=time.perf_counter()
    _elu_flat(np.array([-.5],dtype=np.float32),np.empty(1,dtype=np.float32))
    inverse_factors(np.ones(24,dtype=np.float32),np.float32(.001))
    _affine_channels(np.zeros((24,1),dtype=np.float32),np.empty((24,1),dtype=np.float32),np.ones(24,dtype=np.float32),np.zeros(24,dtype=np.float32))
    _affine_interleaved(np.zeros((1,24),dtype=np.float32),np.empty((1,24),dtype=np.float32),np.ones(24,dtype=np.float32),np.zeros(24,dtype=np.float32))
    report['jit_warmup_seconds']=time.perf_counter()-warm
    report['diagnostic_helpers_sha256']={name:digest(Path(__file__).with_name(name+'.py')) for name in ('elu_numba','bn_numba','cpu_inference_numba')}
    report['scope']='CPU-only diagnostic; no production change or CUDA execution'
    report['fastmath']=False
    original=lambda value:cpu_inference(model.model,value)
    def measured(value):
        started=time.perf_counter();out=original(value)
        report['network_calls'].append({'shape':list(value.shape),'dtype':str(value.dtype),
                                        'seconds':time.perf_counter()-started})
        return out
    model.model.forward=measured
    started=time.perf_counter();result=model(a.input)
    report['api_seconds']=time.perf_counter()-started
    started=time.perf_counter()
    result.image.save(a.output_dir/'image.nii.gz');result.image.save(a.output_dir/'image.npz')
    report['save_seconds']=time.perf_counter()-started
    first=np.load(a.reference_dir/'image.npz')['vol_data'];second=result.image.float_data
    report['float']=metrics(first[None,None],second[None,None])
    close=np.isclose(second,first,rtol=1e-5,atol=1e-3)
    report['fixed_float_gate'].update(passes=bool(close.all()),failed_values=int(np.count_nonzero(~close)))
    expected=nib.load(str(a.reference_dir/'image.nii.gz'))
    x=np.asarray(expected.dataobj,dtype=np.int16);y=result.image.data.astype(np.int16)
    d=np.abs(x-y)
    report['quantized']={'count':d.size,'different':int(np.count_nonzero(d)),
                         'max_abs':int(d.max()),'mae':float(d.mean()),
                         'affine_exact':bool(np.array_equal(expected.affine,nib.load(str(a.output_dir/'image.nii.gz')).affine))}
    report['output_sha256']={name:digest(a.output_dir/name) for name in ('image.nii.gz','image.npz')}
    report['status']='complete'
    (a.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
