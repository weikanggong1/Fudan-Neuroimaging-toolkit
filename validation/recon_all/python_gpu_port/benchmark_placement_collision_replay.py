"""回放已保存的实际有序状态面对；部分捕获不能当作完整迭代验收。"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import numpy as np
import torch


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-directory',type=Path,required=True)
    parser.add_argument('--candidate-directory',type=Path,required=True)
    parser.add_argument('--dependency-directory',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--code-commit',required=True)
    args=parser.parse_args()
    import fnit.recon_all
    for path in reversed(args.dependency_directory):fnit.recon_all.__path__.insert(0,str(path))
    fnit.recon_all.__path__.insert(0,str(args.candidate_directory))
    from fnit.recon_all.place_surface_collision_torch import triangle_pairs_intersect_torch,_source_pairs
    from fnit.recon_all.place_surface_collision import triangles_intersect
    device=torch.device(args.device)
    if device.type=='cuda' and device.index is None:raise ValueError('explicit CUDA target required')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    report=dict(scope='partial_real_ordered_dynamic_triangle_pairs_replay_only',
        hostname=platform.node(),code_commit=args.code_commit,device=str(device),threads=args.threads,
        python_executable=sys.executable,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        tf32_matmul=True,tf32_cudnn=True,autocast=False,
        precision_exception='source float64 geometry from float32 mesh',
        reference='same-input Numba placement Moller predicate',
        predicate_differences_allowed_before_run=0,
        limitations=['partial captured pairs, not complete first trial or complete pial',
                     'does not validate live GPU vertex acceptance',
                     'official full-stage repeat and comparison reported separately'],chunks=[])
    report['source_sha256']={Path(inspect.getfile(inspect.unwrap(getattr(function,'py_func',function)))).name:
        sha256(inspect.getfile(inspect.unwrap(getattr(function,'py_func',function))))
        for function in (triangle_pairs_intersect_torch,triangles_intersect)}
    cpuinfo=Path('/proc/cpuinfo')
    if cpuinfo.exists():
        report['cpu_model']=next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines()
                                  if line.startswith('model name')),None)
    report['affinity_cpu_count']=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def persist(status):
        report['execution_status']=status
        temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        temporary.replace(args.output)
    persist('started')
    paths=sorted(args.input_directory.glob('live_pairs_*.npz'))
    if not paths:raise FileNotFoundError('no captured real triangle pairs')
    for path in paths:
        with np.load(path) as data:
            first,second=data['first'],data['second']
            tick=time.perf_counter();reference=_source_pairs(first,second);cold_source=time.perf_counter()-tick
            sync();tick=time.perf_counter()
            actual,diagnostic=triangle_pairs_intersect_torch(first,second,device=str(device))
            actual=actual.cpu().numpy();sync();cold_torch=time.perf_counter()-tick
            paired=[]
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            for backend in ('source','torch','torch','source'):
                sync();tick=time.perf_counter()
                result=_source_pairs(first,second) if backend=='source' else triangle_pairs_intersect_torch(first,second,device=str(device))[0].cpu().numpy()
                sync();paired.append(dict(backend=backend,seconds=time.perf_counter()-tick,
                    boolean_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
            medians={name:statistics.median(row['seconds'] for row in paired if row['backend']==name) for name in ('source','torch')}
            report['chunks'].append(dict(input_sha256=sha256(path),pairs=len(first),
                different_predicates=int(np.count_nonzero(actual!=reference)),source_hits=int(reference.sum()),
                cold_source_seconds=cold_source,cold_torch_seconds=cold_torch,paired=paired,
                median_seconds=medians,speed_ratio_source_over_torch=medians['source']/medians['torch'],
                diagnostic=diagnostic,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None))
            persist('replay_in_progress')
    report['total_pairs']=sum(row['pairs'] for row in report['chunks'])
    report['total_different_predicates']=sum(row['different_predicates'] for row in report['chunks'])
    if device.type=='cuda':report['gpu']=torch.cuda.get_device_name(device)
    report['cuda_initialized']=torch.cuda.is_initialized()
    report['whole_process_gpu_memory']='not measured; allocator only if CUDA target'
    persist('complete');print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
