"""以已验有序邻接/状态装配为控制，完整冻结候选与最终状态ABBA。

复用上一完整工具worker，不重复候选/评分/恢复算法。两个显式source
覆盖分别用于基线与候选，输入/线程/原冻结依赖完全相同；参考after
仅在原工具保存计算输出后比较。包含导入、JIT、校验与完整读写。
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

import benchmark_recon_topology_candidate_hotspots as original

SOURCES=('smooth_surface_python','topology_candidate_cycle','topology_candidate_geometry')
PATH_ARGUMENTS=('finalization_workspace','cycle_workspace','baseline_overlay_source',
                'overlay_source',*original.INPUTS)


def worker(args):
    """复用完整旧worker，仅选择已声明source及记录真实入射构造耗时。"""
    start=time.perf_counter()
    sys.path[:0]=[str(args.finalization_workspace/'tools'),str(args.cycle_workspace/'tools'),
                  str(args.cycle_workspace/'src')]
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(args.baseline_overlay_source/'fnit/recon_all'))
    if args.worker=='candidate':
        fnit.recon_all.__path__.insert(0,str(args.overlay_source/'fnit/recon_all'))
    fnit.recon_all.__path__.insert(0,str(args.finalization_workspace/'src/fnit/recon_all'))
    from fnit.recon_all import topology_candidate_geometry as geometry
    measured=[];compute=geometry.candidate_incidents
    def observed(*a,**kw):
        tick=time.perf_counter()
        try:return compute(*a,**kw)
        finally:measured.append(dict(function='candidate_incidents',seconds=time.perf_counter()-tick))
    geometry.candidate_incidents=observed
    # Load the selected cycle before delegating. Its imported pipeline then
    # references this same selected geometry; no function result is modified.
    from fnit.recon_all import topology_candidate_cycle
    receipt=original.worker(args)
    for name in SOURCES:
        if args.worker=='candidate':expected=args.overlay_source/f'fnit/recon_all/{name}.py'
        elif name=='topology_candidate_geometry':expected=args.cycle_workspace/f'src/fnit/recon_all/{name}.py'
        else:expected=args.baseline_overlay_source/f'fnit/recon_all/{name}.py'
        actual=receipt['actual_fnit_loaded_source_sha256']['fnit.recon_all.'+name]
        if original.sha(expected)!=actual:raise RuntimeError('wrong actual selected source: '+name)
    receipt.update(instrumented_function_calls=receipt['instrumented_function_calls']+measured,
        child_seconds_including_imports=time.perf_counter()-start,
        delegate_driver_sha256=original.sha(original.__file__),driver_sha256=original.sha(__file__),
        baseline_overlay_source_sha256={name:original.sha(args.baseline_overlay_source/f'fnit/recon_all/{name}.py')
            for name in SOURCES if name!='topology_candidate_geometry'})
    original.write(args.output/'receipt.json',receipt)
    return receipt


def compare_fields(reference,candidate):
    """全部完整容量/数值字段和控制记录逐字节比较，不删失败项。"""
    groups={}
    for label in ('A','B'):
        for ending in ('final.private.npz','state.private.npz'):
            name=label+'.'+ending
            with np.load(reference/name,allow_pickle=False)as old,np.load(candidate/name,allow_pickle=False)as new:
                fields={key:dict(shape_equal=old[key].shape==new[key].shape,
                    dtype_equal=old[key].dtype==new[key].dtype,byte_equal=old[key].tobytes()==new[key].tobytes())
                    for key in old.files if key in new.files}
                groups[name]=dict(keys_equal=set(old.files)==set(new.files),fields=fields)
        name=label+'.control.private.json'
        groups[name]=dict(equal=json.loads((reference/name).read_text())==json.loads((candidate/name).read_text()))
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in (*PATH_ARGUMENTS,'output'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--worker',choices=('control','candidate'))
    args=parser.parse_args()
    if args.worker:worker(args);return
    start=time.perf_counter();args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    records=[]
    for index,backend in enumerate(('control','candidate','candidate','control')):
        folder=args.output/f'{index}-{backend}';command=[sys.executable,str(Path(__file__).resolve())]
        for name in PATH_ARGUMENTS:command.extend(('--'+name.replace('_','-'),str(getattr(args,name))))
        command.extend(('--output',str(folder),'--threads',str(args.threads),'--worker',backend))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NUMBA_CACHE_DIR=str(args.output/f'cache-{backend}'))
        tick=time.perf_counter()
        with(args.output/f'{index}-{backend}.log').open('wb')as log:
            process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        wall=time.perf_counter()-tick
        if process.returncode:
            original.write(args.output/'FAILED.json',dict(index=index,backend=backend,exit_code=process.returncode))
            raise RuntimeError('complete stage failed; original diagnostics retained')
        record=json.loads((folder/'receipt.json').read_text());record.update(index=index,fresh_cli_seconds=wall)
        records.append(record)
        print(json.dumps(dict(index=index,backend=backend,child=0,wall=wall)),flush=True)
    checks=[dict(index=row['index'],backend=row['backend'],groups=compare_fields(
        args.output/'0-control/candidate',args.output/f"{row['index']}-{row['backend']}"/'candidate'))for row in records]
    equal=all(all(group.get('equal',True)and group.get('keys_equal',True)
        and all(field['shape_equal']and field['dtype_equal']and field['byte_equal']
        for field in group.get('fields',{}).values())for group in row['groups'].values())for row in checks)
    original.write(args.output/'report.private.json',dict(status='passed'if equal else'failed',records=records,
        comparisons=checks,complete_driver_seconds=time.perf_counter()-start,
        order=['control','candidate','candidate','control'],driver_sha256=original.sha(__file__),
        delegate_driver_sha256=original.sha(original.__file__),scope='full frozen candidate and final stage; baseline ce96 modules, no full GA/raw T1',
        reference_after_used_to_generate=False,complete_ga=False,final_mesh_quality='not_assessed',
        overall_metric_equivalence='not_assessed'))
    original.write(args.output/'EXIT.json',dict(exit_code=0 if equal else 1))
    if not equal:raise RuntimeError('complete state/control gate failed')


if __name__=='__main__':main()
