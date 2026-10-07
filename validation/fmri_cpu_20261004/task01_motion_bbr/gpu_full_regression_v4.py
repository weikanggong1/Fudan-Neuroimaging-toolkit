"""完整 MCFLIRT/BBR GPU 回归；由协调者以 gpu0 锁串行启动。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def publish(path,record):
    temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)


def main():
    own=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    own.add_argument('--gpu-lock',type=Path,required=True)
    own.add_argument('--maximum-gpu-bytes',type=int,default=20_000_000_000)
    custom,remaining=own.parse_known_args()
    sys.path.insert(0,str(Path(__file__).parent))
    import benchmark_baseline as baseline
    args=baseline.arguments(remaining)
    if args.backend!='fnit' or not str(args.device).startswith('cuda'):raise ValueError('Full GPU regression requires FNIT CUDA API')
    if args.estimate_only or args.stages!=3 or tuple(args.stage_iterations)!=(1,1,1):raise ValueError('Preserve complete default algorithm and normal output contract')
    if args.output_dir.exists():raise FileExistsError('Use a new GPU regression output')
    cpus=baseline.parse_cpus(args.cpu_list);os.sched_setaffinity(0,cpus)
    if os.sched_getaffinity(0)!=cpus:raise RuntimeError('Actual GPU host CPU affinity differs from manifest')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]=str(args.threads)
    args.output_dir.mkdir(parents=True);args.cpu_lock.parent.mkdir(parents=True,exist_ok=True);custom.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
    publish(args.output_dir/'status.safe.json',{'status':'waiting_for_cpu_gpu_leases','pid':os.getpid()})
    # 两个不同的锁：前置import/输入hash/API都在CPU锁内；GPU allocation也受全局GPU0锁保护。
    if args.cpu_lock.resolve()==custom.gpu_lock.resolve():raise ValueError('CPU and GPU locks must be distinct')
    with args.cpu_lock.open('a+') as cpu_lock:
        fcntl.flock(cpu_lock,fcntl.LOCK_EX)
        with custom.gpu_lock.open('a+') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            import torch
            device=torch.device(args.device)
            torch.cuda.set_device(device)
            # Both frozen sources run under the user-requested default TF32
            # policy, including standalone BBR with a fixed initialization.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            total=torch.cuda.get_device_properties(device).total_memory
            if custom.maximum_gpu_bytes<=0:raise ValueError('GPU memory cap must be positive')
            torch.cuda.set_per_process_memory_fraction(min(1.,custom.maximum_gpu_bytes/total),device)
            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            configuration=json.loads(args.case_json.read_text());case=configuration[args.case]
            info={'schema_version':1,'status':'running','function':args.function,'case':args.case,'backend':'fnit_gpu',
                  'source_revision':args.source_revision,'worker_sha256':digest(__file__),'baseline_adapter_sha256':digest(baseline.__file__),
                  'cpu_affinity':sorted(cpus),'cpu_threads':args.threads,'gpu_name':torch.cuda.get_device_name(device),'gpu_total_memory_bytes':total,
                  'maximum_gpu_bytes':custom.maximum_gpu_bytes,'complete_real_frames_required':True,'default_slice_timing':False,'records':[]}
            publish(args.output_dir/'report.safe.json',info)
            for repeat in range(args.warm_repeats+1):
                out=args.output_dir/f'repeat_{repeat}';out.mkdir()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                row=baseline._execute(args,case,configuration,out/'result')
                torch.cuda.synchronize(device)
                row.update(repeat=repeat,call_type='first_complete_gpu_api' if repeat==0 else 'warm_same_process_complete_gpu_api',
                           cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                           matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
                if row['cuda_peak_allocated_bytes']>custom.maximum_gpu_bytes or row['cuda_peak_reserved_bytes']>custom.maximum_gpu_bytes:raise RuntimeError('Full GPU regression exceeded declared 20GB allocator cap')
                info['records'].append(row);publish(args.output_dir/'report.safe.json',info)
            info['status']='complete';publish(args.output_dir/'report.safe.json',info)
            publish(args.output_dir/'status.safe.json',{'status':'complete','records':len(info['records']),'gpu_work_finished':True})
            print(json.dumps({'status':'complete','function':args.function,'case':args.case,'records':len(info['records'])}))

if __name__=='__main__':main()
