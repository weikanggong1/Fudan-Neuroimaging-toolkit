"""冻结完整候选/末段序列：原装配与显式基底状态缓存 ABBA。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import benchmark_recon_topology_candidate_hotspots as original
from benchmark_recon_topology_candidate_incident_reuse import compare_fields

SOURCES=('smooth_surface_python','topology_candidate_cycle','topology_candidate_geometry',
         'topology_defect_smooth','topology_defect_mri_match')
PATH_ARGUMENTS=('finalization_workspace','cycle_workspace','baseline_overlay_source',
                'overlay_source',*original.INPUTS)


def worker(args):
    """完整成熟诊断 worker；缓存构造/JIT/全域校验均计入 API/CLI。"""
    start=time.perf_counter()
    sys.path[:0]=[str(args.finalization_workspace/'tools'),str(args.cycle_workspace/'tools'),
                  str(args.cycle_workspace/'src')]
    import fnit.recon_all
    source=args.overlay_source if args.worker=='candidate' else args.baseline_overlay_source
    fnit.recon_all.__path__.insert(0,str(source/'fnit/recon_all'))
    fnit.recon_all.__path__.insert(0,str(args.finalization_workspace/'src/fnit/recon_all'))
    from fnit.recon_all import topology_candidate_cycle as cycle
    measured=[];contexts={}
    previous=cycle._candidate_state
    if args.worker=='candidate':
        from fnit.recon_all.topology_candidate_state_cache import prepare_candidate_state_context,materialize_candidate_state_cached
        def materialize(*,initial,result):
            tick=time.perf_counter()
            nfaces=int(initial['nfaces'])
            key=hashlib.sha256(np.asarray(initial['vertex_topology']).tobytes()
                +np.asarray(initial['faces'])[:nfaces].tobytes()).hexdigest()
            if key not in contexts:
                context=prepare_candidate_state_context(surface_state=initial)
                contexts[key]=context
                measured.append(dict(function='context_preparation',seconds=context.preparation_seconds))
            value=materialize_candidate_state_cached(initial=initial,result=result,context=contexts[key])
            measured.append(dict(function='complete_state_assembly',seconds=time.perf_counter()-tick))
            return value
    else:
        def materialize(*,initial,result):
            tick=time.perf_counter()
            try:return previous(initial=initial,result=result)
            finally:measured.append(dict(function='complete_state_assembly',seconds=time.perf_counter()-tick))
    cycle._candidate_state=materialize
    receipt=original.worker(args)
    for name in SOURCES:
        if original.sha(source/f'fnit/recon_all/{name}.py')!=receipt['actual_fnit_loaded_source_sha256']['fnit.recon_all.'+name]:
            raise RuntimeError('actual selected source differs from declared '+name)
    receipt.update(instrumented_function_calls=receipt['instrumented_function_calls']+measured,
        explicit_state_context_used=args.worker=='candidate',context_count=len(contexts),
        child_seconds_including_imports=time.perf_counter()-start,driver_sha256=original.sha(__file__),
        delegate_driver_sha256=original.sha(original.__file__),
        baseline_source_sha256={name:original.sha(args.baseline_overlay_source/f'fnit/recon_all/{name}.py')for name in SOURCES})
    original.write(args.output/'receipt.json',receipt)
    return receipt


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
            raise RuntimeError('complete stage failed; original failure retained')
        record=json.loads((folder/'receipt.json').read_text());record.update(index=index,fresh_cli_seconds=wall)
        records.append(record)
        print(json.dumps(dict(index=index,backend=backend,exit_code=0,seconds=wall)),flush=True)
    checks=[dict(index=row['index'],backend=row['backend'],groups=compare_fields(
        args.output/'0-control/candidate',args.output/f"{row['index']}-{row['backend']}"/'candidate'))for row in records]
    equal=all(all(group.get('equal',True)and group.get('keys_equal',True)
        and all(field['shape_equal']and field['dtype_equal']and field['byte_equal']for field in group.get('fields',{}).values())
        for group in row['groups'].values())for row in checks)
    original.write(args.output/'report.private.json',dict(status='passed'if equal else'failed',records=records,
        comparisons=checks,complete_driver_seconds=time.perf_counter()-start,
        order=['control','candidate','candidate','control'],driver_sha256=original.sha(__file__),
        scope='complete frozen candidate/final sequence; fc78 control, explicit state context; not full GA/raw T1',
        reference_after_used_to_generate=False,complete_ga=False,final_mesh_quality='not_assessed',
        overall_metric_equivalence='not_assessed'))
    original.write(args.output/'EXIT.json',dict(exit_code=0 if equal else 1))
    if not equal:raise RuntimeError('complete numeric/state/control regression failed')


if __name__=='__main__':main()
