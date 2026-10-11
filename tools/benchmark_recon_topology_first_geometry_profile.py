"""当前自产首候选短链：普通冷/热与独立cProfile，完整状态合同。

复用成熟prepare和首序列，只观察时间；不启动搜索、不接GPU、不改默认。
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input-surface','input-qsphere','brain','manifest','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=args.manifest.parent;manifest=json.loads(args.manifest.read_text())
    for name,record in manifest['payload'].items():
        p=source/name
        if p.stat().st_size!=record['size']or sha(p)!=record['sha256']:raise ValueError('frozen source mismatch: '+name)
    from benchmark_recon_topology_genetic_distance_wiring import validate_loaded_source
    from benchmark_recon_topology_gpu_first_ga import prepare
    from benchmark_recon_topology_candidate_state import dump_state_checkpoint
    from benchmark_recon_topology_prepared_cache import candidate_payload
    from benchmark_recon_topology_patch_fitness import json_safe
    import torch
    import fnit.recon_all.topology_candidate_geometry as geometry
    import fnit.recon_all.topology_candidate_cycle as cycle
    import fnit.recon_all.topology_candidate_pipeline as pipeline
    import fnit.recon_all.topology_retessellation_dynamic as retess
    import fnit.recon_all.place_surface_normals as normals
    from fnit.recon_all.topology_genetic_search import prepare_genetic_defect_evaluator
    from fnit.recon_all.topology_ga_population import generate_initial_ordering
    from fnit.recon_all.thread_budget import thread_budget
    actual=validate_loaded_source(source,manifest)
    if sha(__file__)!=manifest['payload']['tools/'+Path(__file__).name]['sha256']:raise ValueError('undeclared actual driver')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    calls=defaultdict(list);patches=[]
    def observe(module,name,label):
        original=getattr(module,name)
        def observed(*a,**kw):
            tick=time.perf_counter()
            try:return original(*a,**kw)
            finally:calls[label].append(time.perf_counter()-tick)
        setattr(module,name,observed);patches.append((module,name,original))
    for module,name,label in [
        (pipeline,'candidate_geometry','complete_geometry'),
        (geometry,'retessellate_and_detect_candidate','retess_and_detection'),
        (retess,'retessellate_ordered_edges','retess_ordered_edges'),
        (geometry,'defect_smooth_type2','smooth25'),
        (geometry,'defect_mri_match','MRI_match40'),
        (geometry,'candidate_incidents','candidate_incidents'),
        (geometry,'candidate_face_normals','complete_face_normals'),
        (geometry,'candidate_vertex_normals','selected_vertex_normals'),
        (normals,'ordered_face_csr','face_CSR_stable_sort'),
        (normals.FaceNormalTopology,'__init__','face_topology_build_and_copy'),
        (normals.FaceNormalTopology,'validate','same_candidate_topology_validation'),
        (cycle,'_one_ring_state','state_row_validation'),
        (cycle,'_candidate_state_validated','candidate_state_materialization'),
        (cycle,'restore_candidate_state','DVS_restore'),
        (pipeline,'candidate_second_fundamental_form','candidate_SFF'),
    ]:observe(module,name,label)
    records=[];golden=None;input_hash={k:sha(getattr(args,n))for k,n in [('nofix','input_surface'),('qsphere','input_qsphere'),('brain','brain')]}
    try:
        tick=time.perf_counter()
        with thread_budget(threads=args.threads)as budget:
            thread_enter=time.perf_counter()-tick
            tick=time.perf_counter();state,edges,candidate,preparation_stages=prepare(args);raw_prepare=time.perf_counter()-tick
            for label in ['ordinary_first_process_call','ordinary_warm','profile_warm']:
                calls.clear();profile=cProfile.Profile()if label=='profile_warm'else None
                if profile:profile.enable()
                tick=time.perf_counter();prepared=prepare_genetic_defect_evaluator(surface_state=state,defect_inputs=edges,candidate_inputs=candidate,face_detection_cache=True,blocker_scan_backend='numba',distance_backend='serial');evaluator_prepare=time.perf_counter()-tick
                evaluator=prepared['evaluator'];r=evaluator.retessellation_inputs
                def forbidden(*a):raise RuntimeError('identity first candidate must not consume RNG')
                order=generate_initial_ordering(edge_count=len(edges['edge_vertices']),population_index=0,initial_selection=True,segment_edge_numbers=prepared['segment_edge_numbers'],segment_rows=prepared['segment_rows'],**prepared['overlap_arguments'],draw_random_number=forbidden)
                tick=time.perf_counter();feedback=evaluator(ordering=order['ordering'],discarded_defect_vertices=r['inside_discarded'],vertex_rip=np.asarray(state['vertex_rip'],np.int32),use_counts=np.zeros(len(r['inside_vertices']),np.int32),vertex_fitness=np.zeros(len(r['inside_vertices']),np.float32));cycle_seconds=time.perf_counter()-tick
                if profile:profile.disable();profile.dump_stats(str(args.output/'candidate.cprofile.private'))
                result=evaluator.last_result;timers={k:list(v)for k,v in calls.items()}
                payload,control=candidate_payload(result['candidate']);payload.update({'feedback_'+k:np.asarray(v)for k,v in feedback.items()})
                target=args.output/(label+'.candidate.private.npz');np.savez_compressed(target,**payload);checkpoint=args.output/(label+'.state.private.npz');dump_state_checkpoint(checkpoint,result['surface_state']);control=json_safe(control);write(args.output/(label+'.control.private.json'),control)
                if golden is None:golden=(payload,control,checkpoint)
                candidate_equal=set(payload)==set(golden[0])and all(payload[k].shape==golden[0][k].shape and payload[k].dtype==golden[0][k].dtype and payload[k].tobytes()==golden[0][k].tobytes()for k in payload)
                with np.load(golden[2],allow_pickle=False)as a,np.load(checkpoint,allow_pickle=False)as b:
                    state_equal=set(a.files)==set(b.files)and all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes()for k in a.files)
                    state_fields=len(a.files)
                if not candidate_equal or not state_equal or control!=golden[1]:raise AssertionError('ordinary/profile complete first candidate differs')
                record=dict(label=label,evaluator_preparation_seconds=evaluator_prepare,complete_cycle_seconds=cycle_seconds,observed_nested_calls=timers,candidate_seconds=result['candidate']['seconds'],geometry_stage_seconds=result['candidate']['geometry']['stage_seconds'],fitness_seconds=result['candidate']['fitness']['seconds'],cycle_subseconds=result['seconds'],all_candidate_feedback_arrays_equal=candidate_equal,all_persistent_arrays_equal=state_equal,control_equal=control==golden[1],candidate_feedback_fields=len(payload),persistent_state_fields=state_fields,candidate_sha256=sha(target),state_sha256=sha(checkpoint))
                records.append(record);write(args.output/(label+'.timing.private.json'),json_safe(record));print(json.dumps({'label':label,'cycle_seconds':cycle_seconds,'evaluator_prepare_seconds':evaluator_prepare}),flush=True)
            exit_tick=time.perf_counter()
        thread_exit_seconds=time.perf_counter()-exit_tick
    finally:
        for module,name,original in reversed(patches):setattr(module,name,original)
    # cProfile is a separate observation with profiler overhead; never combine with ordinary timing.
    rows=[]
    for (filename,line,name),(primitive,count,self_seconds,cumulative,_)in pstats.Stats(str(args.output/'candidate.cprofile.private')).stats.items():
        rows.append(dict(module=Path(filename).name,line=line,function=name,calls=count,self_seconds=self_seconds,cumulative_seconds=cumulative))
    actual=validate_loaded_source(source,manifest)
    report=dict(status='complete',scope='same first identity candidate: independent evaluator rebuilt each trial; raw preparation once; no search/whole GA',source_parent_commit=manifest['source_parent_commit'],source_manifest_sha256=sha(args.manifest),input_sha256=input_hash,actual_loaded_module_sha256=actual,driver_sha256=sha(__file__),thread_budget=budget,thread_enter_seconds=thread_enter,thread_exit_seconds=thread_exit_seconds,raw_preparation_seconds=raw_prepare,raw_preparation_stages=preparation_stages,records=records,profile_top_cumulative=sorted(rows,key=lambda x:x['cumulative_seconds'],reverse=True)[:70],profile_top_self=sorted(rows,key=lambda x:x['self_seconds'],reverse=True)[:70],actual_rng_draws=0,cpu_affinity=sorted(os.sched_getaffinity(0)),threads=args.threads,CUDA_execution=False,cold_label='first call in fresh process and new cache directory; Python imports/source hash validation outside candidate timers; no independent empty OS cache claim',production_default_changed=False,whole_GA='not_assessed',whole_recon_all='not_assessed',complete_driver_seconds=time.perf_counter()-started)
    write(args.output/'report.private.json',json_safe(report));write(args.output/'COMPLETE.json',dict(status='complete',report_sha256=sha(args.output/'report.private.json')))


if __name__=='__main__':main()
