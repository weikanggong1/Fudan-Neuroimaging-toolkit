"""Complete real CPU diagnostic with fixed whole-output gates and RSS sampling.

No production imports this script. The frozen original full-graph oracle stays
unchanged. API memory includes retained input/output tensor views for hashing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import nibabel as nib
import numpy as np
import torch
from fnit.synthsr import SynthSR
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr_first_layer import digest, metrics


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'weights', 'reference-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--implementation', choices=('original','production','prototype'), default='prototype')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('preserve prior runs')
    args.output_dir.mkdir(parents=True)
    assert args.weights.stat().st_size == 106163752
    assert digest(args.weights) == 'a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b'
    torch.set_num_threads(8)
    if args.implementation != 'original':
        from numba import set_num_threads
        set_num_threads(8)
        if args.implementation == 'production':
            from fnit.synthsr._cpu_math import _elu_flat, inverse_factors, _affine_channels, _affine_interleaved
        else:
            from cpu_inference_numba import cpu_inference
            from elu_onednn_numba import elu_torch, _elu_flat
            from bn_numba import inverse_factors, _affine_channels, _affine_interleaved
    report = {'scope': __doc__, 'source_revision': args.source_revision, 'driver_sha256': digest(__file__),
              'input_sha256': digest(args.input), 'weight_sha256': digest(args.weights),
              'torch_version': torch.__version__, 'model_sha256': digest(sys.modules['fnit.synthsr.model'].__file__),
              'implementation':args.implementation, 'helpers_sha256': {},
              'network_calls': [], 'rss_sample_interval_seconds': .05, 'rss_phase_peak_bytes': {},
              'fixed_float_gate': {'rtol':1e-5, 'atol':1e-3}, 'cpu_elu_explicit_fma':args.implementation != 'original',
              'timing_scope':'constructor/JIT/API/save/compare separately; CNN timers nested in API'}
    stop = threading.Event(); phase = ['constructor']; page_bytes=os.sysconf('SC_PAGE_SIZE')
    def sample():
        while not stop.is_set():
            name=phase[0]; rss=int(Path('/proc/self/statm').read_text().split()[1])*page_bytes
            report['rss_phase_peak_bytes'][name]=max(report['rss_phase_peak_bytes'].get(name,0),rss)
            stop.wait(.05)
    sampler=threading.Thread(target=sample, daemon=True); sampler.start()
    captured=[]
    try:
        started=time.perf_counter(); model=SynthSR(weights=args.weights,device='cpu',threads=8)
        if args.implementation == 'prototype':
            model.model.to(memory_format=torch.channels_last_3d)
        report['constructor_seconds']=time.perf_counter()-started
        phase[0]='jit_warmup'; started=time.perf_counter()
        if args.implementation != 'original':
            _elu_flat(np.array([-.5],dtype=np.float32),np.empty(1,dtype=np.float32))
            inverse_factors(np.ones(24,dtype=np.float32),np.float32(.001))
            _affine_channels(np.zeros((24,1),dtype=np.float32),np.empty((24,1),dtype=np.float32),np.ones(24,dtype=np.float32),np.zeros(24,dtype=np.float32))
            _affine_interleaved(np.zeros((1,24),dtype=np.float32),np.empty((1,24),dtype=np.float32),np.ones(24,dtype=np.float32),np.zeros(24,dtype=np.float32))
            if args.implementation == 'production':
                report['helpers_sha256']={name:digest(sys.modules['fnit.synthsr.'+name].__file__) for name in ('_cpu_math',)}
            else:
                report['helpers_sha256']={name:digest(Path(__file__).with_name(name+'.py')) for name in ('elu_onednn_numba','elu_numba','bn_numba','cpu_inference_numba')}
        report['jit_warmup_seconds']=time.perf_counter()-started
        forward = (lambda value:cpu_inference(model.model,value,elu_function=elu_torch)) if args.implementation == 'prototype' else model.model.forward
        def measured(value):
            started=time.perf_counter(); output=forward(value)
            report['network_calls'].append({'seconds':time.perf_counter()-started,'shape':list(value.shape),'dtype':str(value.dtype)})
            captured.append((value.numpy(),output.numpy()))
            return output
        model.model.forward=measured
        phase[0]='api'; started=time.perf_counter(); result=model(args.input)
        report['api_seconds']=time.perf_counter()-started
        if args.implementation == 'production':
            report['helpers_sha256']['_cpu_inference']=digest(sys.modules['fnit.synthsr._cpu_inference'].__file__)
        report['retained_network_view_bytes']=sum(v.nbytes for pair in captured for v in pair)
        phase[0]='save'; started=time.perf_counter()
        result.image.save(args.output_dir/'image.nii.gz'); result.image.save(args.output_dir/'image.npz')
        report['save_seconds']=time.perf_counter()-started
        phase[0]='compare'; started=time.perf_counter()
        for index,(input_value,output) in enumerate(captured):
            expected_input=np.load(args.reference_dir/f'network_{index}_input.npy',mmap_mode='r')
            expected_output=np.load(args.reference_dir/f'network_{index}_output.npy',mmap_mode='r')
            row=report['network_calls'][index]
            row['input_array_sha256']=array_sha(input_value)
            row['input_exact']=row['input_array_sha256']==array_sha(expected_input)
            row['output_array_sha256']=array_sha(output)
            row['frozen_output_file_sha256']=digest(args.reference_dir/f'network_{index}_output.npy')
            row['output_exact']=row['output_array_sha256']==array_sha(expected_output)
            row['metrics']=metrics(expected_output,output)
        first=np.load(args.reference_dir/'image.npz')['vol_data']; second=result.image.float_data
        report['float']=metrics(first[None,None],second[None,None])
        close=np.isclose(second,first,rtol=1e-5,atol=1e-3)
        report['fixed_float_gate'].update(passes=bool(close.all()),failed_values=int(np.count_nonzero(~close)))
        expected=nib.load(str(args.reference_dir/'image.nii.gz'))
        x=np.asarray(expected.dataobj,dtype=np.int16); y=result.image.data.astype(np.int16); difference=np.abs(x-y)
        report['quantized']={'count':difference.size,'different':int(np.count_nonzero(difference)),
                             'max_abs':int(difference.max()),'mae':float(difference.mean()),
                             'affine_exact':bool(np.array_equal(expected.affine,nib.load(str(args.output_dir/'image.nii.gz')).affine))}
        report['output_sha256']={name:digest(args.output_dir/name) for name in ('image.nii.gz','image.npz')}
        report['compare_seconds']=time.perf_counter()-started
    finally:
        stop.set(); sampler.join()
    report['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['status']='complete'
    (args.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
