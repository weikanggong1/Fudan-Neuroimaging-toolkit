"""原始链完成后，用同自产orig验证缓存N4 exec：冷CLI与已初始化父API。

--raw-pair必须完整；--output新目录；--profiling-module固定同期采样器。
参考只用于计算后的比较。API父持有真实orig FP32 GPU张量，不用模拟
数组冒充实际输入；父cache-off，子exec独立cache-enabled，无全局更改。
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import nibabel as nib
import numpy as np
import torch

from benchmark_input_chain import metrics,lta_matrix_comparison
from fnit.recon_all.n4_torch_worker import run_isolated_n4


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def executable_ast(path):
    tree=ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-pair',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--previous-source',type=Path,required=True)
    p.add_argument('--profiling-module',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    raw=json.loads((a.raw_pair/'summary.json').read_text())
    if raw['status']!='complete_raw_input_to_nu_pair':raise ValueError('raw pair must be complete')
    for case in raw['rows']:
        orig=case['comparisons']['mri/orig.mgz']
        if orig['whole']['different_voxels'] or not orig['same_affine'] or not orig['same_dtype']:
            raise ValueError('cache policy comparison requires identical self-produced orig voxels and geometry')
    a.output.mkdir(parents=True)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    report={'scope':'two_real_self_produced_orig_cached_exec_not_raw_T1_whole_recon_all',
        'status':'running','rows':[],'raw_pair_sha256':sha(a.raw_pair/'summary.json'),
        'source_sha256':{str(Path(__file__)):sha(__file__),str(a.profiling_module):sha(a.profiling_module)},
        'parent_allocator_environment':{x:os.environ.get(x) for x in ('PYTORCH_NO_CUDA_MEMORY_CACHING','CUDA_VISIBLE_DEVICES')},
        'device':a.device,'threads':a.threads,'production_default_changed':False,
        'whole_metrics_equivalence':'not assessed','whole_recon_all_acceleration':'not measured',
        'parent_cuda_initialized_before_benchmark':torch.cuda.is_initialized()}
    old=a.previous_source/'src/fnit/recon_all/input_n4_chain.py';new=a.source_root/'src/fnit/recon_all/input_n4_chain.py'
    report['chain_doc_only_bridge']={'tested_module_sha256':sha(old),'current_module_sha256':sha(new),
        'tested_executable_ast_sha256':executable_ast(old),'current_executable_ast_sha256':executable_ast(new),
        'note':'only function docstring corrected from NIfTI/MGH to 3D NIfTI-1; raw run frozen module unchanged'}
    assert executable_ast(old)==executable_ast(new)
    spec=importlib.util.spec_from_file_location('n4_worker_fixed_sampler',a.profiling_module)
    profiling=importlib.util.module_from_spec(spec);spec.loader.exec_module(profiling)
    sampler=profiling.ProcessTreeDeviceSampler(device=a.device,parent_pid=os.getpid(),interval=.5)
    stop=threading.Event()
    def sample():
        while not stop.is_set():sampler.sample_if_due(force=True);stop.wait(.5)
    background=threading.Thread(target=sample,daemon=True);background.start()
    def save():
        (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        # First both cold CLI calls, keeping this parent entirely uninitialised.
        for case in raw['rows']:
            directory=a.output/case['case']/'cold_cli';directory.mkdir(parents=True)
            source=a.raw_pair/case['case']/'native/subject/mri/orig.mgz'
            output=directory/'nu0.mgz';child_report=directory/'child.json'
            command=[sys.executable,'-m','fnit.recon_all.n4_torch_worker','--input',str(source),
                '--output',str(output),'--report',str(child_report),'--device',a.device,'--threads',str(a.threads),
                '--code-version','803aec50+frozen-N4-worker-v2']
            begin=time.perf_counter()
            with (directory/'cli.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            report['rows'].append({'case':case['case'],'mode':'cold_cli','input_sha256':sha(source),
                'parent_cuda_initialized':torch.cuda.is_initialized(),'wall_seconds_including_exec_load_transfer_io':time.perf_counter()-begin,
                'worker':json.loads(child_report.read_text())});save()
        assert not torch.cuda.is_initialized()
        for case in raw['rows']:
            source=a.raw_pair/case['case']/'native/subject/mri/orig.mgz'
            retained=torch.from_numpy(np.asarray(nib.load(str(source)).dataobj,dtype=np.float32).copy()).to(a.device)
            torch.cuda.synchronize(a.device)
            directory=a.output/case['case']/'initialized_api';directory.mkdir(parents=True)
            begin=time.perf_counter()
            result=run_isolated_n4(input_path=source,output_path=directory/'nu0.mgz',report_path=directory/'child.json',
                device=a.device,threads=a.threads,profile=False,code_version='803aec50+frozen-N4-worker-v2')
            torch.cuda.synchronize(a.device)
            report['rows'].append({'case':case['case'],'mode':'initialized_api','input_sha256':sha(source),
                'parent_cuda_initialized':torch.cuda.is_initialized(),'parent_retained_real_input_gpu_bytes':retained.numel()*retained.element_size(),
                'wall_seconds_including_exec_load_transfer_io':time.perf_counter()-begin,'worker':result})
            del retained;save()
        for row in report['rows']:
            native=a.raw_pair/row['case']/'native/subject';direct=a.raw_pair/row['case']/'torch/subject'
            output=a.output/row['case']/row['mode']/'nu0.mgz'
            brain=np.asarray(nib.load(str(native/'mri/synthstrip.mgz')).dataobj)>0
            row['to_native']=metrics(output,native/'mri/tmp/nu0.mgz',brain)
            row['to_cache_disabled_complete_n4']=metrics(output,direct/'mri/tmp/nu0.mgz',brain)
        report['additional_lta_diagnostics']={case['case']:lta_matrix_comparison(
            a.raw_pair/case['case']/'torch/subject/mri/transforms/synthmorph.mni305/aff.lta',
            a.raw_pair/case['case']/'native/subject/mri/transforms/synthmorph.mni305/aff.lta') for case in raw['rows']}
        report['status']='complete_cached_exec_two_modes'
    except Exception as error:report['status']='failed';report['error']=repr(error);raise
    finally:
        stop.set();background.join(timeout=10);sampler.sample_if_due(force=True)
        report['gpu_process_sampler']=sampler.report();save()


if __name__=='__main__':main()
