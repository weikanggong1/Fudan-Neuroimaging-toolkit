"""冻结完整候选/末态序列的独立旧新 ABBA；不重放原 C++ 生产步骤。

旧 finalization/cycle workspace及输入由显式参数指定，读取后核对固定
PATCH_SOURCE。新source只覆盖本轮两个模块，其他依赖保持冻结旧版。
每child完整加载/校准→两条候选评分/DVS→最终重连/匹配/写出；参考
after仅由成熟工具最后读取。计时含首次JIT/传输外的CPU读写，不使用
cProfile结果作为旧时钟。不代表完整GA或原始T1整例，不启GPU。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')


def worker(args):
    """完整已验证run接口，仅附透明函数计时；不省略原步骤或比较。"""
    start=time.perf_counter();out=args.output
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    frozen=args.finalization_workspace
    manifest=json.loads((frozen/'PATCH_SOURCE.json').read_text())
    for name,record in manifest['payload'].items():
        if sha(frozen/name)!=record['sha256'] or (frozen/name).stat().st_size!=record['size']:
            raise RuntimeError('frozen payload changed: '+name)
    sys.path[:0]=[str(frozen/'tools'),str(args.cycle_workspace/'tools'),
                  str(args.cycle_workspace/'src')]
    import fnit.recon_all
    if args.worker=='candidate':
        fnit.recon_all.__path__.insert(0,str(args.overlay_source/'fnit/recon_all'))
    fnit.recon_all.__path__.insert(0,str(frozen/'src/fnit/recon_all'))
    from fnit.recon_all import smooth_surface_python as smoothing
    measured=[]
    def timed(name, function):
        def observed(*a,**kw):
            tick=time.perf_counter()
            try:return function(*a,**kw)
            finally:measured.append(dict(function=name,seconds=time.perf_counter()-tick))
        return observed
    smoothing.ordered_neighbors=timed('ordered_neighbors',smoothing.ordered_neighbors)
    program=frozen/'tools/benchmark_recon_topology_ga_finalize.py'
    namespace=runpy.run_path(str(program),run_name='fnit_hotspot_frozen_target')
    from fnit.recon_all import topology_candidate_cycle as cycle
    cycle._candidate_state=timed('_candidate_state',cycle._candidate_state)
    arguments={name:getattr(args,name) for name in INPUTS}
    arguments.update(output=out/'candidate',threads=args.threads)
    tick=time.perf_counter();result=namespace['run'](**arguments)
    complete_api=time.perf_counter()-tick
    loaded={name:sha(module.__file__) for name,module in sorted(sys.modules.items())
            if name.startswith('fnit.') and getattr(module,'__file__',None)
            and str(module.__file__).endswith('.py')}
    if args.worker=='candidate':
        for name in ('smooth_surface_python','topology_candidate_cycle'):
            expected=sha(args.overlay_source/f'fnit/recon_all/{name}.py')
            if loaded[f'fnit.recon_all.{name}']!=expected:
                raise RuntimeError('candidate overlay was not the actual loaded module')
    receipt=dict(status='complete',backend=args.worker,
        complete_api_seconds=complete_api,child_seconds_including_imports=time.perf_counter()-start,
        instrumented_function_calls=measured,records=result['records'],
        calibration_seconds=result['raw_calibration_seconds'],
        first_candidate_source_gate=result['initial_candidate_strict_source_gate'],
        final_repeat_gate=result['status'],
        input_sha256={name:sha(getattr(args,name)) for name in INPUTS},
        target_program_sha256=sha(program),frozen_patch_manifest_sha256=sha(frozen/'PATCH_SOURCE.json'),
        actual_fnit_loaded_source_sha256=loaded,driver_sha256=sha(__file__),
        thread_environment={name:os.environ.get(name) for name in (
            'OMP_NUM_THREADS','NUMBA_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        cpu_affinity=sorted(os.sched_getaffinity(0)),target_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),
        process_gpu_memory_bytes=None,complete_ga=False,final_mesh_quality='not_assessed')
    write(out/'receipt.json',receipt)
    write(out/'EXIT.json',dict(exit_code=0))
    return receipt


INPUTS=('before_retess','initial_attributes','initial_global_priors','input_surface',
        'input_qsphere','brain','after_statistics_reference','after_volume_reference','fitness_reference')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('finalization_workspace','cycle_workspace','overlay_source','output',*INPUTS):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--worker',choices=('control','candidate'))
    args=parser.parse_args()
    if args.worker:
        worker(args);return
    start=time.perf_counter();args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    records=[]
    for index,backend in enumerate(('control','candidate','candidate','control')):
        folder=args.output/f'{index}-{backend}'
        command=[sys.executable,str(Path(__file__).resolve())]
        for name in ('finalization_workspace','cycle_workspace','overlay_source',*INPUTS):
            command.extend(('--'+name.replace('_','-'),str(getattr(args,name))))
        command.extend(('--output',str(folder),'--threads',str(args.threads),'--worker',backend))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NUMBA_CACHE_DIR=str(args.output/f'cache-{backend}'))
        tick=time.perf_counter()
        with (args.output/f'{index}-{backend}.log').open('wb') as log:
            child=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        wall=time.perf_counter()-tick
        if child.returncode:
            write(args.output/'FAILED.json',dict(index=index,backend=backend,exit_code=child.returncode))
            raise RuntimeError('full frozen sequence failed; all diagnostics preserved')
        record=json.loads((folder/'receipt.json').read_text())
        record.update(index=index,fresh_cli_seconds=wall)
        records.append(record)
        print(json.dumps(dict(index=index,backend=backend,child=0,wall=wall)),flush=True)
    import numpy as np
    checks=[]
    reference=args.output/'0-control/candidate'
    for item in records:
        candidate=args.output/f"{item['index']}-{item['backend']}"/'candidate'
        groups={}
        for label in ('A','B'):
            for ending in ('final.private.npz','state.private.npz'):
                name=label+'.'+ending
                with np.load(reference/name,allow_pickle=False) as old,np.load(candidate/name,allow_pickle=False) as new:
                    keys_equal=set(old.files)==set(new.files)
                    entries={key:dict(shape_equal=old[key].shape==new[key].shape,
                        dtype_equal=old[key].dtype==new[key].dtype,
                        byte_equal=old[key].tobytes()==new[key].tobytes())
                        for key in old.files if key in new.files}
                    groups[name]=dict(keys_equal=keys_equal,fields=entries)
            name=label+'.control.private.json'
            groups[name]=dict(equal=json.loads((reference/name).read_text())==json.loads((candidate/name).read_text()))
        checks.append(dict(index=item['index'],backend=item['backend'],groups=groups))
    passed=all(all(group.get('equal',True) and group.get('keys_equal',True)
        and all(field['shape_equal'] and field['dtype_equal'] and field['byte_equal']
                for field in group.get('fields',{}).values()) for group in row['groups'].values()) for row in checks)
    report=dict(status='passed' if passed else 'failed',records=records,comparisons=checks,
        complete_driver_seconds=time.perf_counter()-start,order=['control','candidate','candidate','control'],
        scope='complete frozen same-input candidate and final materialization ABBA; no full GA or raw-T1 whole run',
        reference_after_used_to_generate=False,complete_ga=False,final_mesh_quality='not_assessed',
        overall_metric_equivalence='not_assessed',driver_sha256=sha(__file__))
    write(args.output/'report.private.json',report)
    write(args.output/'EXIT.json',dict(exit_code=0 if passed else 1))
    if not passed:raise RuntimeError('same-input complete numeric/state/control gate failed')


if __name__=='__main__':
    main()
