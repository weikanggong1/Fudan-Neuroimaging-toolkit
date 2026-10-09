"""完整N4的独立exec：复用现有算法，仅在子进程启用CUDA分配缓存。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from . import n4_bspline_torch, n4_itk_torch_experimental
from .profiling import autocast_state, configure_cuda_allocator
from .thread_budget import native_thread_environment


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def _validate(*,input_path,output_path,report_path,device,threads):
    target=torch.device(device)
    if target.type!='cuda' or target.index is None:
        raise ValueError('N4 worker requires an explicit logical CUDA index')
    if isinstance(threads,bool) or not isinstance(threads,int) or threads<1:
        raise ValueError('threads must be a positive integer')
    paths=[Path(p).resolve() for p in (input_path,output_path,report_path)]
    if len(set(paths))!=3:raise ValueError('input/output/report paths must differ')
    if not Path(input_path).is_file():raise FileNotFoundError(input_path)
    if Path(output_path).exists() or Path(report_path).exists():
        raise FileExistsError('N4 worker output/report must be new paths')


def run_isolated_n4(*,input_path:str|Path,output_path:str|Path,report_path:str|Path,
                    device:str,threads:int=4,profile:bool=False,
                    code_version:str='FNIT-source-hashes')->dict:
    """父API通过新exec运行完整缓存N4，保持父allocator与TF32设置。

    input_path为自产3D非负NIfTI/MGH，output_path为原xyz网格uint8；两者
    affine单位mm不变。report_path为新JSON。device必须cuda:N，保留
    CUDA_VISIBLE_DEVICES；threads默认4。profile默认False，不额外逐段
    同步。code_version是用户绑定标签，实际source SHA另记。父CUDA可
    已初始化，exec前不执行子CUDA计算；不fork计算、不参考复制、不调用
    C++。返回完整报告及isolated_cli_wall_seconds，含exec/导入/校验/
    初始化/读写/哈希/200轮/子退出。失败传播，不原生/CPU自动回退。
    """
    started=time.perf_counter()
    input_path,output_path,report_path=map(Path,(input_path,output_path,report_path))
    _validate(input_path=input_path,output_path=output_path,report_path=report_path,
              device=device,threads=threads)
    initialized=torch.cuda.is_initialized()
    environment,_=native_thread_environment(threads=threads)
    environment.pop('PYTORCH_NO_CUDA_MEMORY_CACHING',None)
    bootstrap=("import sys,runpy,fnit.recon_all; "
               "fnit.recon_all.__path__.insert(0,sys.argv[1]); "
               "sys.argv=sys.argv[2:]; "
               "runpy.run_module('fnit.recon_all.n4_torch_worker',run_name='__main__')")
    command=[sys.executable,'-c',bootstrap,str(Path(__file__).parent),
             'fnit.recon_all.n4_torch_worker','--input',str(input_path),'--output',str(output_path),
             '--report',str(report_path),'--device',device,'--threads',str(threads),
             '--matmul-tf32',str(int(torch.backends.cuda.matmul.allow_tf32)),
             '--cudnn-tf32',str(int(torch.backends.cudnn.allow_tf32)),
             '--code-version',code_version]
    if profile:command.append('--profile')
    subprocess.run(command,env=environment,check=True)
    result=json.loads(report_path.read_text())
    if result['input_sha256']!=_sha(input_path) or result['output_sha256']!=_sha(output_path):
        raise ValueError('N4 worker input/output hash mismatch')
    result['parent_cuda_initialized_before_exec']=initialized
    result['isolated_cli_wall_seconds']=time.perf_counter()-started
    result['isolation']='fresh exec; cached child; parent allocator and precision preserved'
    pending=report_path.with_suffix('.pending');pending.write_text(json.dumps(result,indent=2)+'\n');pending.replace(report_path)
    return result


def run_worker(*,input_path:Path,output_path:Path,report_path:Path,device:str='cuda:0',
               threads:int=4,profile:bool=False,matmul_tf32:bool=True,
               cudnn_tf32:bool=True,code_version:str='FNIT-source-hashes')->dict:
    """新进程复用correct_volume；相同完整recipe，uint8同网格写出。

    参数空间/路径同run_isolated_n4。matmul_tf32/cudnn_tf32默认True，
    父API传实际设置；不启用半精度。CUDA已初始化时拒绝改变allocator。
    返回输入/输出及算法SHA、完整N4详情、实际线程/精度/allocator、
    指定GPU张量峰值和API墙钟；张量峰值不是父子同期显存。同期采样由
    调用者复用FNIT sampler，不将未知值记0。不更改生产N4默认。
    """
    started=time.perf_counter()
    if torch.cuda.is_initialized():raise ValueError('N4 worker must start before CUDA initialization')
    _validate(input_path=input_path,output_path=output_path,report_path=report_path,
              device=device,threads=threads)
    allocator=configure_cuda_allocator(device=device,policy='enabled')
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=matmul_tf32;torch.backends.cudnn.allow_tf32=cudnn_tf32
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    api=n4_itk_torch_experimental.correct_volume(input_path=input_path,output_path=output_path,
                                                device=device,profile=profile)
    report={'scope':'same-input complete N4 cached exec; not raw-T1 whole recon-all',
        'input_sha256':_sha(input_path),'output_sha256':_sha(output_path),'code_version':code_version,
        'source_sha256':{'worker':_sha(__file__),'full_n4':_sha(n4_itk_torch_experimental.__file__),
                         'bspline_fit':_sha(n4_bspline_torch.__file__)},
        'worker_pid':os.getpid(),'cuda_allocator':allocator,'logical_device':device,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'precision':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                     'autocast':autocast_state('cuda'),'half_requested':False},
        'threads':{'torch':torch.get_num_threads(),'requested':threads,'cpu_affinity':sorted(os.sched_getaffinity(0))},
        'gpu_name':torch.cuda.get_device_name(device),'torch_version':torch.__version__,'api':api,
        'allocated_peak_bytes':torch.cuda.max_memory_allocated(device),'reserved_peak_bytes':torch.cuda.max_memory_reserved(device),
        'worker_api_wall_seconds_including_validation_init_read_transfer_write':time.perf_counter()-started,
        'whole_metrics_equivalence':'not assessed','production_default_changed':False}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    pending=report_path.with_suffix('.pending');pending.write_text(json.dumps(report,indent=2)+'\n');pending.replace(report_path)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','output','report'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--matmul-tf32',choices=('0','1'),default='1')
    parser.add_argument('--cudnn-tf32',choices=('0','1'),default='1')
    parser.add_argument('--code-version',default='FNIT-source-hashes')
    a=parser.parse_args()
    run_worker(input_path=a.input,output_path=a.output,report_path=a.report,device=a.device,
               threads=a.threads,profile=a.profile,matmul_tf32=a.matmul_tf32=='1',
               cudnn_tf32=a.cudnn_tf32=='1',code_version=a.code_version)


if __name__=='__main__':main()
