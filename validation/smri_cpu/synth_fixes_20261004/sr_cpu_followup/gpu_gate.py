"""Complete original/new CUDA SR comparison with the unchanged 20 GB cap.

Full real arrays remain private on the server. API timing includes numerical
trace copies; saving/comparison are separate. GNU wall includes cold imports,
CUDA initialization, loading, API, saving and the posthoc comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import nibabel as nib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sr_first_layer import digest,metrics


def state():
    result=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
                           '--format=csv,noheader,nounits'],text=True,capture_output=True)
    return result.stdout.strip().splitlines()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','weights','reference-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-revision',required=True)
    parser.add_argument('--reference-kind',choices=('official-cpu','old-gpu'),required=True)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('preserve prior results')
    args.output_dir.mkdir(parents=True)
    assert args.weights.stat().st_size==106163752
    assert digest(args.weights)=='a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b'
    report={'scope':__doc__,'source_revision':args.source_revision,'reference_kind':args.reference_kind,
            'input_sha256':digest(args.input),'weight_sha256':digest(args.weights),
            'driver_sha256':digest(__file__),'allocator_budget_bytes':20_000_000_000,
            'external_gpu_before':state(),'network_calls':[]}
    import torch
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    print('checkpoint: torch imported; starting explicit CUDA initialization',flush=True)
    torch.cuda.device_count();torch.cuda.init()
    report['cuda_free_total_after_init_bytes']=list(torch.cuda.mem_get_info(0))
    properties=torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(20_000_000_000/properties.total_memory,0)
    probe=torch.arange(12,device='cuda',dtype=torch.float32);probe=probe*2;torch.cuda.synchronize();del probe
    print('checkpoint: CUDA init and allocator fraction succeeded; importing FNIT',flush=True)
    from fnit.synthsr import SynthSR
    import fnit.synthsr.model as source
    report.update(torch_version=torch.__version__,cudnn_version=torch.backends.cudnn.version(),
                  model_sha256=digest(source.__file__),cuda_total_memory_bytes=properties.total_memory)
    torch.cuda.reset_peak_memory_stats(0)
    started=time.perf_counter();model=SynthSR(weights=args.weights,device='cuda',threads=8)
    torch.cuda.synchronize();report['constructor_seconds']=time.perf_counter()-started
    report['conv_weight_strides']=[list(module.weight.stride()) for module in model.model.modules() if isinstance(module,torch.nn.Conv3d)]
    report['cuda_policy']={'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                           'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                           'benchmark':torch.backends.cudnn.benchmark,'deterministic':torch.backends.cudnn.deterministic}
    forward=model.model.forward;captured=[]
    def measured(value):
        torch.cuda.synchronize();started=time.perf_counter();output=forward(value);torch.cuda.synchronize()
        report['network_calls'].append({'seconds':time.perf_counter()-started,'shape':list(value.shape),'dtype':str(value.dtype)})
        # Both arms use these same trace transfers; no GPU arrays are retained.
        captured.append((value.detach().cpu().numpy(),output.detach().cpu().numpy()))
        return output
    model.model.forward=measured
    torch.cuda.synchronize();started=time.perf_counter();result=model(args.input);torch.cuda.synchronize()
    report['api_seconds']=time.perf_counter()-started
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved(0)
    report['cpu_math_imported']='fnit.synthsr._cpu_math' in sys.modules
    report['cpu_dispatch_imported']='fnit.synthsr._cpu_inference' in sys.modules
    assert not report['cpu_math_imported'] and not report['cpu_dispatch_imported']
    report['external_gpu_after_api']=state()
    started=time.perf_counter()
    result.image.save(args.output_dir/'image.nii.gz');result.image.save(args.output_dir/'image.npz')
    for index,(input_value,output) in enumerate(captured):
        np.save(args.output_dir/f'network_{index}_input.npy',input_value)
        np.save(args.output_dir/f'network_{index}_output.npy',output)
    report['save_seconds']=time.perf_counter()-started
    started=time.perf_counter()
    for index,(input_value,output) in enumerate(captured):
        expected_input=np.load(args.reference_dir/f'network_{index}_input.npy',mmap_mode='r')
        expected_output=np.load(args.reference_dir/f'network_{index}_output.npy',mmap_mode='r')
        row=report['network_calls'][index]
        row.update(input_array_sha256=array_sha(input_value),input_exact=array_sha(input_value)==array_sha(expected_input),
                   output_array_sha256=array_sha(output),output_exact=array_sha(output)==array_sha(expected_output),
                   metrics=metrics(expected_output,output))
    expected_float=np.load(args.reference_dir/'image.npz')['vol_data']
    report['float']=metrics(expected_float[None,None],result.image.float_data[None,None])
    expected=nib.load(str(args.reference_dir/'image.nii.gz'));actual=nib.load(str(args.output_dir/'image.nii.gz'))
    delta=np.abs(np.asarray(expected.dataobj,dtype=np.int16)-result.image.data.astype(np.int16))
    report['quantized']={'different':int(np.count_nonzero(delta)),'max_abs':int(delta.max()),
                         'affine_exact':bool(np.array_equal(expected.affine,actual.affine))}
    report['header_binary_sha256']=hashlib.sha256(actual.header.binaryblock).hexdigest()
    report['header_exact']=expected.header.binaryblock==actual.header.binaryblock
    report['output_file_sha256']={name:digest(args.output_dir/name) for name in ('image.nii.gz','image.npz')}
    report['compare_seconds']=time.perf_counter()-started
    report['status']='complete'
    (args.output_dir/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
