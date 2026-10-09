"""同冻结网格测试CUDA缓存关闭策略，不把不可用张量统计记为零显存。

--source-root 固定候选源码；--data为四网格公开manifest；--pair为已完成
同输入native参考目录；--profiling-module为固定FNIT采样模块。每次API
包含校验、拓扑、传输、积分和写出，GPU前后同步；读参考只发生在候选
全部输出之后。仅诊断独立进程，不更改生产环境或全局低显存策略。
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import threading
import time

import torch
from benchmark_complete import compare
from fnit.recon_all.inflate_standard_run import run_standard_inflate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profiling-module',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if json.loads((args.pair/'summary.json').read_text())['status']!='complete_stage_pair':
        raise ValueError('same input paired reference must be complete')
    if os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')!='1':
        raise ValueError('cache-disabled test requires a fresh process with flag=1')
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    from numba import set_num_threads
    set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    from fnit.recon_all import inflate_standard_run,inflate_torch,inflate_topology,place_surface_normals,mris_register_average_numba
    modules=[Path(__file__),args.profiling_module]+[Path(x.__file__) for x in
        (inflate_standard_run,inflate_torch,inflate_topology,place_surface_normals,mris_register_average_numba)]
    report={'scope':'complete_inflation_cache_disabled_same_input_not_recon_all',
        'host':socket.gethostname(),'device':args.device,'threads':args.threads,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'allocator_environment':{key:os.environ.get(key) for key in
            ('PYTORCH_NO_CUDA_MEMORY_CACHING','PYTORCH_CUDA_ALLOC_CONF','CUDA_VISIBLE_DEVICES')},
        'source_sha256':{str(p):sha(p) for p in modules},'pair_summary_sha256':sha(args.pair/'summary.json'),
        'tensor_counters_available':False,'peak_allocated_bytes':None,'peak_reserved_bytes':None,
        'tensor_counter_note':'caching allocator disabled; zero counters do not establish zero GPU memory',
        'production_default_changed':False,'status':'running','rows':[],
        'matmul_tf32':True,'cudnn_tf32':True,'half_precision':False,
        'cuda_initialized_before_benchmark':torch.cuda.is_initialized()}
    def save():
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('fixed_fnit_sampler',args.profiling_module)
    profiling=importlib.util.module_from_spec(spec);spec.loader.exec_module(profiling)
    sampler=profiling.ProcessTreeDeviceSampler(device=args.device,parent_pid=os.getpid(),interval=.5)
    stop=threading.Event()
    def sample():
        while not stop.is_set():
            sampler.sample_if_due(force=True);stop.wait(.5)
    background=threading.Thread(target=sample,daemon=True);background.start()
    try:
        begin=time.perf_counter();torch.empty(1,device=args.device);torch.cuda.synchronize(args.device)
        report['cuda_initialization_seconds_excluded_from_each_api']=time.perf_counter()-begin
        report['gpu']=torch.cuda.get_device_name(args.device)
        save()
        data=json.loads((args.data/'manifest.json').read_text())
        for entry in data['cases']:
            input_surface=args.data/entry['surface']
            if sha(input_surface)!=entry['sha256']:raise ValueError('input hash changed')
            hemi=entry['hemisphere'];directory=args.output/entry['case']/hemi;directory.mkdir(parents=True)
            torch.cuda.synchronize(args.device);begin=time.perf_counter()
            api=run_standard_inflate(input_surface=input_surface,inflated_output=directory/(hemi+'.inflated'),
                sulc_output=directory/(hemi+'.sulc'),backend='torch',device=args.device,profile=False)
            torch.cuda.synchronize(args.device)
            row={'case':entry['case'],'hemisphere':hemi,'input_sha256':entry['sha256'],
                 'wall_seconds':time.perf_counter()-begin,'api':api}
            row['inflated_sha256']=sha(directory/(hemi+'.inflated'));row['sulc_sha256']=sha(directory/(hemi+'.sulc'))
            report['rows'].append(row);save();print('DONE',entry['case'],hemi,row['wall_seconds'],flush=True)
        for row in report['rows']:
            directory=args.output/row['case']/row['hemisphere']
            row['to_native']=compare(directory,args.pair/row['case']/row['hemisphere']/'native_1',row['hemisphere'])
            row['to_cached_torch']=compare(directory,args.pair/row['case']/row['hemisphere']/'torch_2',row['hemisphere'])
        report['status']='complete_cache_disabled_pair'
    except Exception as error:
        report['status']='failed';report['error']=repr(error);raise
    finally:
        stop.set();background.join(timeout=10);sampler.sample_if_due(force=True)
        report['gpu_process_sampler']=sampler.report();save()


if __name__=='__main__':main()
