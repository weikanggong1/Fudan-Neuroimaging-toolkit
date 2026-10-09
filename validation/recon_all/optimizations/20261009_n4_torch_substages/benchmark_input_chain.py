"""两例原始公开T1→nu的新空目录连续配对，不以冻结orig冒充原始链。

--config为case/input/public_source_url/sha256数组；--weights/--assets为
声明资源，--native为固定独立Conda N4，--output须不存在。各backend均
从原始T1重算前段；跨输入参考只用于完成后的诊断。--profiling-module为
固定FNIT采样器。包含全部加载/传输/IO，GPU计时前后同步；模型FP32例外
由既有函数执行并报告，不改变其数学、随机状态或cache低显存策略。
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys
import threading
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import label
import torch

from fnit.recon_all.input_n4_chain import run_input_n4_chain


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(candidate,reference,brain):
    a,b=nib.load(str(candidate)),nib.load(str(reference))
    if a.shape!=b.shape:return {'same_shape':False,'same_index_comparison':'invalid'}
    values=np.asarray(a.dataobj);truth=np.asarray(b.dataobj)
    delta=values.astype(np.float64)-truth.astype(np.float64);absolute=np.abs(delta)
    different=delta!=0
    components,count=label(different)
    sizes=np.bincount(components.reshape(-1))[1:]
    def region(mask):
        d=delta[mask];s=np.abs(d)
        return {'voxels':int(mask.sum()),'different_voxels':int(np.count_nonzero(d)),
            'max_abs':float(s.max(initial=0)),'p99_abs':float(np.percentile(s,99)) if len(s) else None,
            'mean_signed':float(d.mean()) if len(d) else None,
            'rmse':float(np.sqrt(np.mean(d*d))) if len(d) else None,
            'negative_differences':int((d<0).sum()),'positive_differences':int((d>0).sum())}
    return {'same_shape':True,'same_affine':bool(np.array_equal(a.affine,b.affine)),
        'same_zooms':bool(np.array_equal(a.header.get_zooms(),b.header.get_zooms())),
        'same_dtype':bool(a.get_data_dtype()==b.get_data_dtype()),
        'candidate_dtype':str(a.get_data_dtype()),'reference_dtype':str(b.get_data_dtype()),
        'brain_definition':'self-produced native-run SynthStrip image>0, not atlas labels',
        'whole':region(np.ones(a.shape,bool)),'brain':region(brain),'outside_brain':region(~brain),
        'different_component_count_6_connectivity':int(count),
        'largest_different_component_voxels':int(sizes.max(initial=0))}


def lta_matrix_comparison(candidate,reference):
    """文件名随空目录不同；另比较变换矩阵，不把路径文本差异当注册误差。"""
    def read(path):
        lines=Path(path).read_text().splitlines();start=lines.index('1 4 4')+1
        return np.asarray([[float(x) for x in row.split()] for row in lines[start:start+4]],np.float64)
    delta=read(candidate)-read(reference)
    return {'different_matrix_elements':int(np.count_nonzero(delta)),
            'max_abs_matrix_element':float(np.abs(delta).max()),
            'raw_text_identity_includes_different_subject_filenames':Path(candidate).read_bytes()==Path(reference).read_bytes()}


def main():
    experiment_started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profiling-module',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    from numba import set_num_threads
    set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    spec=importlib.util.spec_from_file_location('fixed_n4_chain_sampler',args.profiling_module)
    profiling=importlib.util.module_from_spec(spec);spec.loader.exec_module(profiling)
    sampler=profiling.ProcessTreeDeviceSampler(device=args.device,parent_pid=os.getpid(),interval=.5)
    stop=threading.Event()
    def sample():
        while not stop.is_set():sampler.sample_if_due(force=True);stop.wait(.5)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    report={'scope':'two_public_raw_T1_empty_directory_continuous_to_nu_not_recon_all',
        'host':socket.gethostname(),'device':args.device,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'threads':args.threads,'seed':args.seed,'profile':args.profile,'status':'running','rows':[],
        'allocator_environment':{key:os.environ.get(key) for key in
            ('PYTORCH_NO_CUDA_MEMORY_CACHING','PYTORCH_CUDA_ALLOC_CONF','CUDA_VISIBLE_DEVICES')},
        'matmul_tf32_default':True,'cudnn_tf32_default':True,'half_precision':False,
        'precision_exceptions':'existing SynthStrip cuDNN FP32 and Talairach affine FP32; actual forward fields retained',
        'source_sha256':{str(Path(__file__)):sha(__file__),str(args.profiling_module):sha(args.profiling_module)},
        'resource_sha256':{str(p):sha(p) for p in
            (args.native,args.weights/'synthstrip.1.pt',args.weights/'synthmorph.affine.2.h5',args.assets/'average/mni305.cor.stripped.mgz')},
        'whole_recon_all_speedup':'not measured','whole_metrics_equivalence':'not assessed',
        'production_default_changed':False,'cuda_initialized_before_benchmark':torch.cuda.is_initialized()}
    def save():
        p=args.output/'summary.json';q=args.output/'summary.tmp';q.write_text(json.dumps(report,indent=2)+'\n');q.replace(p)
    try:
        for index,entry in enumerate(json.loads(args.config.read_text())):
            input_file=Path(entry['input'])
            if sha(input_file)!=entry['sha256']:raise ValueError('raw input hash changed')
            row={**entry,'runs':[]};report['rows'].append(row);save()
            # Alternating order limits fixed-first backend bias. Every directory starts empty.
            order=('native','torch') if index%2==0 else ('torch','native')
            for backend in order:
                random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
                directory=args.output/entry['case']/backend/'subject'
                if directory.exists():raise FileExistsError(directory)
                if torch.device(args.device).type=='cuda':torch.cuda.synchronize(args.device)
                begin=time.perf_counter()
                result=run_input_n4_chain(t1=input_file,subject_dir=directory,weights_dir=args.weights,
                    assets_dir=args.assets,n4_binary=args.native,device=args.device,threads=args.threads,
                    n4_backend=backend,profile=args.profile)
                if torch.device(args.device).type=='cuda':torch.cuda.synchronize(args.device)
                item={'backend':backend,'wall_seconds_including_load_transfer_and_io':time.perf_counter()-begin,'report':result,
                    'output_sha256':{str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}}
                if torch.device(args.device).type!='cuda' or os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')=='1':
                    item['peak_allocated_bytes']=None;item['peak_reserved_bytes']=None
                else:
                    item['process_cumulative_peak_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
                    item['process_cumulative_peak_reserved_bytes']=torch.cuda.max_memory_reserved(args.device)
                row['runs'].append(item);save();print('DONE',entry['case'],backend,item['wall_seconds_including_load_transfer_and_io'],flush=True)
            native=args.output/entry['case']/'native/subject';candidate=args.output/entry['case']/'torch/subject'
            brain=np.asarray(nib.load(str(native/'mri/synthstrip.mgz')).dataobj)>0
            row['comparisons']={f:metrics(candidate/f,native/f,brain) for f in
                ('mri/orig.mgz','mri/synthstrip.mgz','mri/tmp/nu0.mgz','mri/nu.mgz')}
            row['talairach_xfm_text_identical']=(native/'mri/transforms/talairach.xfm').read_bytes()==(candidate/'mri/transforms/talairach.xfm').read_bytes()
            row['talairach_lta_text_identical']=(native/'mri/transforms/synthmorph.mni305/aff.lta').read_bytes()==(candidate/'mri/transforms/synthmorph.mni305/aff.lta').read_bytes()
            row['talairach_lta_matrix']=lta_matrix_comparison(candidate/'mri/transforms/synthmorph.mni305/aff.lta',native/'mri/transforms/synthmorph.mni305/aff.lta')
            save()
        report['status']='complete_raw_input_to_nu_pair'
    except Exception as error:
        report['status']='failed';report['error']=repr(error);raise
    finally:
        stop.set();thread.join(timeout=10);sampler.sample_if_due(force=True)
        report['gpu_process_sampler']=sampler.report()
        report['experiment_wall_seconds_including_validation_context_load_transfer_io']=time.perf_counter()-experiment_started
        report['each_api_timing_boundary']='initialized parent CUDA due explicit timing synchronization; Python import excluded; experiment wall includes validation and context setup'
        for name,module in tuple(sys.modules.items()):
            p=getattr(module,'__file__',None)
            if name.startswith('fnit.') and p and str(p).endswith('.py'):
                report['source_sha256'][str(p)]=sha(p)
        save()


if __name__=='__main__':main()
