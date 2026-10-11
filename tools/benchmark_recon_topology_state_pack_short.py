"""当前自产三候选ABBA：无缓存完整状态装配，含随机排序/连续反馈/IO。"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import struct
import sys
import time
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    start=time.perf_counter();parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input-surface','input-qsphere','brain','manifest','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=args.manifest.parent;manifest=json.loads(args.manifest.read_text())
    for n,v in manifest['payload'].items():
        p=source/n
        if p.stat().st_size!=v['size']or sha(p)!=v['sha256']:raise ValueError('frozen source differs: '+n)
    from benchmark_recon_topology_genetic_distance_wiring import validate_loaded_source
    from benchmark_recon_topology_gpu_first_ga import prepare
    from benchmark_recon_topology_prepared_cache import candidate_payload
    from benchmark_recon_topology_candidate_state import dump_state_checkpoint
    from benchmark_recon_topology_patch_fitness import json_safe
    from recon_topology_state_pack_experiment import state_pack_scope
    from fnit.recon_all.topology_genetic_search import prepare_genetic_defect_evaluator,TopologyRandomStream
    from fnit.recon_all.topology_ga_population import generate_initial_ordering
    from fnit.recon_all.thread_budget import thread_budget
    from fnit.recon_all import topology_candidate_cycle as cycle
    import torch
    validate_loaded_source(source,manifest)
    for name in ['benchmark_recon_topology_state_pack_short.py','recon_topology_state_pack_experiment.py']:
        path=source/'tools'/name
        if name=='benchmark_recon_topology_state_pack_short.py'and Path(__file__).resolve()!=path.resolve():raise ValueError('actual entry path differs')
        if sha(path)!=manifest['payload']['tools/'+name]['sha256']:raise ValueError('actual experiment differs')
    import recon_topology_state_pack_experiment as experiment
    if Path(experiment.__file__).resolve()!=(source/'tools/recon_topology_state_pack_experiment.py').resolve():raise ValueError('wrong actual experiment imported')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=True
    import numba,nibabel
    CPU_model=next(line.split(':',1)[1].strip()for line in Path('/proc/cpuinfo').read_text().splitlines()if line.startswith('model name'))
    runtime_environment=dict(python=platform.python_version(),NumPy=np.__version__,Numba=numba.__version__,Torch=torch.__version__,nibabel=nibabel.__version__,CPU_model=CPU_model,OS=platform.system(),kernel=platform.release(),machine=platform.machine(),record_kind='within_actual_benchmark_process_before_stages',environment_fully_locked=False)
    stages=[]
    with thread_budget(threads=args.threads)as budget:
        tick=time.perf_counter();state,edges,candidate,upstream=prepare(args);upstream_seconds=time.perf_counter()-tick
        for index,backend in enumerate(['baseline','packed','packed','baseline']):
            target=args.output/(str(index)+'-'+backend);target.mkdir();stage_start=time.perf_counter();timings=[];calls=[]
            tick=time.perf_counter();prepared=prepare_genetic_defect_evaluator(surface_state=state,defect_inputs=edges,candidate_inputs=candidate,face_detection_cache=True,blocker_scan_backend='numba',distance_backend='serial');preparation=time.perf_counter()-tick
            evaluator=prepared['evaluator'];r=evaluator.retessellation_inputs;rng=TopologyRandomStream(seed=1234);rng_sha=hashlib.sha256();observed=0
            def draw(low,high):
                nonlocal observed
                value=rng(low,high);rng_sha.update(struct.pack('<Qddd',observed,float(low),float(high),float(value)));observed+=1;return value
            feedback=dict(vertex_rip=np.asarray(state['vertex_rip'],np.int32),use_counts=np.zeros(len(r['inside_vertices']),np.int32),vertex_fitness=np.zeros(len(r['inside_vertices']),np.float32))
            with state_pack_scope(backend=='packed'):
                original=cycle._candidate_state_validated
                def measured(**kw):
                    tick=time.perf_counter()
                    try:return original(**kw)
                    finally:calls.append(time.perf_counter()-tick)
                cycle._candidate_state_validated=measured
                try:
                    for population_index in range(3):
                        case_start=time.perf_counter();tick=time.perf_counter()
                        order=generate_initial_ordering(edge_count=len(edges['edge_vertices']),population_index=population_index,initial_selection=True,segment_edge_numbers=prepared['segment_edge_numbers'],segment_rows=prepared['segment_rows'],**prepared['overlap_arguments'],draw_random_number=draw)
                        ordering_seconds=time.perf_counter()-tick;draws_before=observed;tick=time.perf_counter()
                        feedback=evaluator(ordering=order['ordering'],discarded_defect_vertices=r['inside_discarded'],**{k:feedback[k]for k in ['vertex_rip','use_counts','vertex_fitness']});cycle_seconds=time.perf_counter()-tick
                        if observed!=draws_before:raise RuntimeError('materialization/score unexpectedly consumes RNG')
                        result=evaluator.last_result;tick=time.perf_counter();payload,control=candidate_payload(result['candidate']);payload.update({'feedback_'+k:np.asarray(v)for k,v in feedback.items()})
                        np.savez_compressed(target/(str(population_index)+'.candidate.private.npz'),**payload);dump_state_checkpoint(target/(str(population_index)+'.state.private.npz'),result['surface_state']);write(target/(str(population_index)+'.candidate_control.private.json'),json_safe(control));np.savez_compressed(target/(str(population_index)+'.ordering.private.npz'),**{k:order[k]for k in ['ordering','segment_order','random_draws']})
                        order_control={k:v for k,v in order.items()if k not in ['ordering','segment_order','random_draws']};write(target/(str(population_index)+'.ordering_control.private.json'),json_safe(order_control));IO_seconds=time.perf_counter()-tick
                        row=dict(population_index=population_index,ordering_seconds=ordering_seconds,complete_candidate_cycle_seconds=cycle_seconds,state_materialization_seconds=calls[-1],serialization_IO_seconds=IO_seconds,complete_case_API_with_IO_seconds=time.perf_counter()-case_start,actual_rng_draws_after_case=observed,candidate_seconds=result['candidate']['seconds'],geometry_stage_seconds=result['candidate']['geometry']['stage_seconds']);timings.append(row);print(json.dumps({'stage':index,'backend':backend,**{k:row[k]for k in ['population_index','complete_candidate_cycle_seconds','state_materialization_seconds','actual_rng_draws_after_case']}}),flush=True)
                finally:cycle._candidate_state_validated=original
            assert rng.draws==observed and observed>0
            row=dict(stage=index,backend=backend,evaluator_prepare_seconds=preparation,complete_three_candidate_stage_API_with_alloc_validation_IO_seconds=time.perf_counter()-stage_start,timings=timings,random_draws=observed,random_sequence_sha256=rng_sha.hexdigest());stages.append(row);write(target/'timings.private.json',json_safe(row))
    # Reference artefacts are read only after every side has written all new outputs.
    reference=args.output/'0-baseline';checks=[]
    for stage in stages:
        target=args.output/(str(stage['stage'])+'-'+stage['backend']);groups={}
        for case in range(3):
            for suffix in ['candidate.private.npz','state.private.npz','ordering.private.npz']:
                name=str(case)+'.'+suffix
                with np.load(reference/name,allow_pickle=False)as a,np.load(target/name,allow_pickle=False)as b:
                    assert set(a.files)==set(b.files),name
                    fields={}
                    for k in sorted(a.files):
                        equal=a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes();assert equal,(stage['stage'],name,k)
                        fields[k]=dict(dtype=str(a[k].dtype),shape=list(a[k].shape),bytes_equal=equal,array_sha256=hashlib.sha256(b[k].tobytes()).hexdigest())
                    groups[name]=dict(exact_expected_fields=sorted(a.files),fields=fields,all_equal=True)
            for suffix in ['candidate_control.private.json','ordering_control.private.json']:
                name=str(case)+'.'+suffix;a=json.loads((reference/name).read_text());b=json.loads((target/name).read_text());assert a==b,name;groups[name]=dict(expected_keys=sorted(a),equal=True,file_sha256=sha(target/name))
        assert stage['random_sequence_sha256']==stages[0]['random_sequence_sha256']and stage['random_draws']==stages[0]['random_draws']
        checks.append(dict(stage=stage['stage'],backend=stage['backend'],groups=groups,all_state_controls_ordering_RNG_equal=True))
    loaded=validate_loaded_source(source,manifest);loaded['recon_topology_state_pack_experiment']=sha(experiment.__file__)
    report=dict(status='complete',scope='three consecutive candidates identity/random/random, same-process ABBA; not complete GA',source_parent_commit=manifest['source_parent_commit'],prior_delivery_commit=manifest['prior_delivery_commit'],source_manifest_sha256=sha(args.manifest),driver_sha256=sha(__file__),actual_loaded_module_sha256=loaded,runtime_environment=runtime_environment,input_sha256={k:sha(getattr(args,v))for k,v in [('nofix','input_surface'),('qsphere','input_qsphere'),('brain','brain')]},upstream_preparation_seconds=upstream_seconds,upstream_stages=upstream,stages=stages,comparisons=checks,thread_budget=budget,CPU_affinity=sorted(os.sched_getaffinity(0)),threads=args.threads,production_default_changed=False,whole_GA='not_assessed',whole_recon_all='not_assessed',initial_compilation_in_stage_timings=True,independent_cold_children=False,main_entry_seconds=time.perf_counter()-start)
    write(args.output/'report.private.json',json_safe(report));write(args.output/'COMPLETE.json',dict(status='complete',report_sha256=sha(args.output/'report.private.json')))


if __name__=='__main__':main()
