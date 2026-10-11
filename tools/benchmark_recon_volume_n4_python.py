"""完整同输入 N4 冷/热诊断；公开 JSON 仅含 typed 指标和哈希。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import subprocess
import threading

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--backend', choices=('simpleitk1','simpleitk4','torch'), required=True)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--repeat', type=int, default=2)
args = parser.parse_args()
started = time.perf_counter()
import nibabel as nib
import numpy as np
import torch
from fnit.recon_all.n4_simpleitk import correct_volume as correct_sitk
from fnit.recon_all.n4_itk_torch_experimental import correct_volume as correct_torch

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

if args.repeat < 1 or args.repeat > 4:
    raise ValueError('repeat must be 1..4')
args.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
original=nib.load(str(args.input))
report={'scope':'same_input_complete_N4_only_not_whole', 'backend':args.backend,
        'input_sha256':sha(args.input), 'input_shape':[int(v) for v in original.shape],
        'input_affine':original.affine.tolist(),'input_dtype':str(original.get_data_dtype()),
        'source_sha256':{},'runs':[], 'strict_tolerance_declared':0,
        'overall_equivalence':'not_assessed', 'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        'target_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'nvml_process_attribution':'not established in container; no device-total or zero treated as own memory'}
source=Path(__file__).resolve().parents[1]/'src/fnit/recon_all'
for name in ('n4_simpleitk.py','n4_itk_torch_experimental.py','n4_bspline_torch.py','n4_itk.py'):
    report['source_sha256'][name]=sha(source/name)
report['gpu_device_samples']=[]
stop=threading.Event()
def monitor():
    while not stop.is_set():
        tick=time.time()
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            matching=[v.split(',') for v in raw.splitlines() if os.environ.get('CUDA_VISIBLE_DEVICES') in (v.split(',')[0].strip(),v.split(',')[1].strip())]
            for fields in matching:
                report['gpu_device_samples'].append({'time':tick,'physical_index':int(fields[0]),'uuid':fields[1].strip(),'device_total_bytes':int(fields[2])*1048576,'utilization_percent':int(fields[3]),'own_memory_bytes':None})
        except Exception as error:
            report['gpu_device_samples'].append({'time':tick,'error_type':type(error).__name__})
        stop.wait(1)
watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
for index in range(args.repeat):
    output=args.output/f'nu0-{index}.mgz'
    if args.backend=='torch':
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    tick=time.perf_counter()
    if args.backend=='torch':
        api=correct_torch(input_path=args.input,output_path=output,device=args.device,profile=True)
        torch.cuda.synchronize(args.device)
        api={k:v for k,v in api.items() if k not in ('input_path','output_path')}
    else:
        api=correct_sitk(input_file=args.input,output_file=output,reconstruction_threads=4,
                         fitting_threads=1 if args.backend=='simpleitk1' else 4)
        api={k:v for k,v in api.items() if k!='output'}
    wall=time.perf_counter()-tick
    out=nib.load(str(output));array=np.asarray(out.dataobj)
    row={'index':index,'cache_scope':'first call in new process' if index==0 else 'same-process repeated call',
         'wall_including_load_transfer_JIT_IO_seconds':wall,'api':api,
         'output_sha256':sha(output),'dtype':str(out.get_data_dtype()),
         'affine':out.affine.tolist(),'shape':[int(v) for v in out.shape],
         'finite':bool(np.isfinite(array).all()),
         'torch_precision':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'autocast':False}}
    if args.backend=='torch':
        row['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
        row['torch_peak_reserved_bytes']=torch.cuda.max_memory_reserved(args.device)
    if index:
        first=np.asarray(nib.load(str(args.output/'nu0-0.mgz')).dataobj)
        delta=np.abs(array.astype(np.float64)-first.astype(np.float64))
        row['repeat_vs_first']={'different_elements':int(np.count_nonzero(array!=first)), 'max_error':float(delta.max()),'p99_error':float(np.quantile(delta,.99))}
    report['runs'].append(row)
    report['outer_seconds_after_argument_parse_including_imports']=time.perf_counter()-started
    (args.output/'metrics.public.json').write_text(json.dumps(report,indent=2)+'\n')
stop.set();watcher.join()
report['gpu_sample_interval_requested_seconds']=1
report['execution_complete']=True
(args.output/'metrics.public.json').write_text(json.dumps(report,indent=2)+'\n')
