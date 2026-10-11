"""同一真实nofix/qsphere/MRI自产首缺陷；显式距离hook，完整RNG/末态。"""
import argparse,hashlib,json,os,struct,sys,time
from pathlib import Path
from functools import partial
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def prepare(args):
    import nibabel as nib
    import nibabel.freesurfer.io as fsio
    from fnit.recon_all.topology_genetic_base_prepare import prepare_nofix_genetic_base
    from fnit.recon_all.topology_global_priors import global_topology_priors
    from fnit.recon_all.topology_candidate_inputs import prepare_candidate_input_context,prepare_defect_candidate_inputs
    from fnit.recon_all.topology_genetic_defect_inputs import ordered_genetic_defect_inputs
    from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
    raw,faces,metadata=fsio.read_geometry(str(args.input_surface),read_metadata=True)
    sphere,sfaces=fsio.read_geometry(str(args.input_qsphere))
    raw,faces,sphere=np.asarray(raw,np.float32),np.asarray(faces,np.int32),np.asarray(sphere,np.float32)
    if raw.shape!=sphere.shape or not np.array_equal(faces,sfaces):raise ValueError('same ordered own nofix/qsphere required')
    image=nib.load(args.brain);volume=np.asarray(image.dataobj);matrix=source_topology_ras_to_voxel(volume_header=image.header)
    ticks={};start=time.perf_counter()
    base=prepare_nofix_genetic_base(vertices=raw,faces=faces,qsphere_vertices=sphere,volume=volume,ras_to_voxel=matrix,source_metadata=dict(use_real_ras=0,hemisphere=0,status=0,origxyz_status=0,volume_geometry=dict(metadata)))
    ticks['raw_base']=time.perf_counter()-start;start=time.perf_counter()
    state=base['surface_state'];metric=base['metric'];source=base['prepared']['base_arguments']
    priors=global_topology_priors(principal_curvature_fields=base['sff1'],smoothed_original_vertices=source['source_vertex_fields']['original_vertices'],faces=faces,neighbor_offsets=metric['one_ring_offsets'],neighbors=metric['one_ring_neighbors'],vertex_rip=metric['sff_arguments']['vertex_rip'],surface_status='sphere')
    context=prepare_candidate_input_context(vertices=raw,faces=faces,defect_vertices=base['merge']['labels']!=0,vertex_translation=base['vertex_translation'],volume=volume,ras_to_voxel=matrix,global_priors=priors)
    ticks['global_prior_and_input_context']=time.perf_counter()-start;start=time.perf_counter();p=context.preparation
    edges=ordered_genetic_defect_inputs(canonical_vertices=source['source_vertex_fields']['canonical_vertices'],original_vertices=p['original_vertices'],original_normals=p['original_normals'],faces=faces,defect_labels=base['merge']['labels'],ordered_regions=base['merge']['regions'],retention_status=base['retention_status'],vertex_translation=base['vertex_translation'],white_values=p['white_values'],gray_values=p['gray_values'],volume=volume,ras_to_voxel=matrix,defect_number=0,use_overlap=False)
    ticks['ordered_edges']=time.perf_counter()-start;start=time.perf_counter()
    candidate=prepare_defect_candidate_inputs(context=context,original_defect_hull=edges['original_hull_vertices'],candidate_edges=edges['edge_vertices'],current_corrected_original_vertices=state['original_vertices'])
    ticks['local_prior']=time.perf_counter()-start
    return state,edges,candidate['candidate_kwargs'],ticks


def main():
    p=argparse.ArgumentParser()
    for k in ['input-surface','input-qsphere','brain','output','manifest']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--mode',choices=['first','full'],required=True);p.add_argument('--backend',choices=['cpu','gpu'],default='gpu');p.add_argument('--device',required=True);p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    frozen=json.loads(args.manifest.read_text());root=args.manifest.parent
    for name,record in frozen['payload'].items():
        if sha(root/name)!=record['sha256']:raise ValueError('frozen payload changed: '+name)
    import torch
    from fnit.recon_all.thread_budget import thread_budget
    from fnit.recon_all.topology_genetic_search import prepare_genetic_defect_evaluator,search_and_finalize_defect,TopologyRandomStream
    from fnit.recon_all.topology_ga_population import generate_initial_ordering
    from fnit.recon_all.topology_gpu_distance import gpu_defect_signed_distance
    from benchmark_recon_topology_candidate_state import dump_state_checkpoint
    from benchmark_recon_topology_prepared_cache import candidate_payload
    from benchmark_recon_topology_patch_fitness import json_safe
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.cuda.set_device(args.device);torch.cuda.reset_peak_memory_stats(args.device)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    device_uuid=str(getattr(torch.cuda.get_device_properties(args.device),'uuid','unavailable'))
    if device_uuid=='unavailable':raise RuntimeError('actual device UUID required for benchmark attribution')
    write(args.output/'DEVICE.private.json',dict(cuda_device=args.device,gpu_uuid=device_uuid if device_uuid.startswith(('GPU-','MIG-'))else'GPU-'+device_uuid))
    started=time.perf_counter();timings=[];rng_sha=hashlib.sha256();observed=0
    with thread_budget(threads=args.threads):
        state,edges,candidate,stages=prepare(args)
        hook=partial(gpu_defect_signed_distance,device=args.device)
        if args.mode=='first':
            golden=None;comparisons=[]
            for trial in range(4):
                backend=['cpu','gpu','gpu','cpu'][trial];values=dict(candidate)
                if backend=='gpu':values['distance_evaluator']=hook
                tick=time.perf_counter();prepared=prepare_genetic_defect_evaluator(surface_state=state,defect_inputs=edges,candidate_inputs=values,face_detection_cache=True,blocker_scan_backend='numba')
                preparation=time.perf_counter()-tick;evaluator=prepared['evaluator'];r=evaluator.retessellation_inputs
                def forbidden(*a):raise RuntimeError('first identity candidate must not consume RNG')
                order=generate_initial_ordering(edge_count=len(edges['edge_vertices']),population_index=0,initial_selection=True,segment_edge_numbers=prepared['segment_edge_numbers'],segment_rows=prepared['segment_rows'],**prepared['overlap_arguments'],draw_random_number=forbidden)
                tick=time.perf_counter();feedback=evaluator(ordering=order['ordering'],discarded_defect_vertices=r['inside_discarded'],vertex_rip=np.asarray(state['vertex_rip'],np.int32),use_counts=np.zeros(len(r['inside_vertices']),np.int32),vertex_fitness=np.zeros(len(r['inside_vertices']),np.float32));torch.cuda.synchronize(args.device)
                elapsed=time.perf_counter()-tick;result=evaluator.last_result
                arrays,control=candidate_payload(result['candidate']);arrays.update({k:np.asarray(v)for k,v in feedback.items()})
                name=backend+'-'+str(trial);np.savez_compressed(args.output/(name+'.candidate.private.npz'),**arrays);dump_state_checkpoint(args.output/(name+'.state.private.npz'),result['surface_state'])
                if golden is None:golden=(arrays,control,args.output/(name+'.state.private.npz'))
                with np.load(golden[2],allow_pickle=False)as a,np.load(args.output/(name+'.state.private.npz'),allow_pickle=False)as b:
                    state_equal=set(a.files)==set(b.files)and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes()for k in a.files)
                exact=set(arrays)==set(golden[0])and all(arrays[k].dtype==golden[0][k].dtype and arrays[k].shape==golden[0][k].shape and arrays[k].tobytes()==golden[0][k].tobytes()for k in arrays)
                comparisons.append(dict(trial=trial,backend=backend,all_candidate_fields_byte_equal=exact,all_persistent_state_fields_byte_equal=state_equal,control_equal=control==golden[1]))
                timings.append(dict(trial=trial,backend=backend,evaluator_preparation_seconds=preparation,complete_candidate_cycle_seconds=elapsed,subseconds=result['candidate']['seconds'],fitness_seconds=result['candidate']['fitness']['seconds']))
            passed=all(all(row[k]for k in ['all_candidate_fields_byte_equal','all_persistent_state_fields_byte_equal','control_equal'])for row in comparisons)
            extra=dict(comparisons=comparisons,actual_rng_draws=0)
        else:
            if args.backend=='gpu':candidate=dict(candidate,distance_evaluator=hook)
            rng=TopologyRandomStream(seed=1234)
            def draw(low,high):
                nonlocal observed
                value=rng(low,high);rng_sha.update(struct.pack('<Qddd',observed,float(low),float(high),value));observed+=1;return value
            # Observation records wall times only; inputs, ordering and results remain unchanged.
            import fnit.recon_all.topology_candidate_cycle as cycle
            original=cycle.evaluate_candidate
            def observe(**kw):
                tick=time.perf_counter();result=original(**kw);row=dict(candidate=len(timings),fitness=result['fitness']['fitness'],seconds=time.perf_counter()-tick,subseconds=result['seconds'],fitness_seconds=result['fitness']['seconds']);timings.append(row)
                with (args.output/'progress.private.jsonl').open('a')as f:f.write(json.dumps(json_safe(row))+'\n')
                return result
            cycle.evaluate_candidate=observe
            try:result=search_and_finalize_defect(surface_state=state,defect_inputs=edges,candidate_inputs=candidate,draw_random_number=draw,population_size=10,max_unchanged=10,niters=-1,initial_selection=True,vertex_eliminate=True,deletion_count=1,elimination_threshold=np.float32(4),check_surface_intersection=False,intersection_checker=None,face_detection_cache=True,blocker_scan_backend='numba')
            finally:cycle.evaluate_candidate=original
            search=result['search'];final=result['finalization'];arrays,control=candidate_payload(final['final_evaluation']['candidate']);np.savez_compressed(args.output/'final_candidate.private.npz',**arrays);write(args.output/'candidate_control.private.json',json_safe(control));dump_state_checkpoint(args.output/'final_state.private.npz',result['surface_state'])
            np.savez_compressed(args.output/'search.private.npz',**{k:search[k]for k in ['best_ordering','best_discarded','population_orderings','population_fitness','population_ranks','vertex_rip','use_counts','vertex_fitness']})
            np.savez_compressed(args.output/'all_scores.private.npz',fitness=np.asarray([row['fitness']for row in search['evaluation_trace']]+[final['fitness']],np.float64))
            scientific={k:search[k]for k in ['counts','evaluation_trace','generation_trace','termination','generations','best_fitness','deletion_count','elimination_threshold']};scientific.update(actual_policy=result['actual_policy'],random_draws=rng.draws,observed_random_draws=observed,random_sequence_sha256=rng_sha.hexdigest(),final_fitness=final['fitness'],fitness_matches_best=final['fitness_matches_best'],additional_discarded=final['cleanup']['additional_discarded_count']);write(args.output/'search_control.private.json',json_safe(scientific))
            passed=search['counts']['evaluated']==495 and len(timings)==496 and len(search['generation_trace'])==29 and search['generations']==28 and rng.draws==observed
            extra=dict(search_evaluations=search['counts']['evaluated'],scores=len(timings),random_draws=rng.draws,random_sequence_sha256=rng_sha.hexdigest(),complete_first_defect=True)
    report=dict(status='passed'if passed else'failed',mode=args.mode,backend=args.backend,source_manifest_sha256=sha(args.manifest),input_sha256={k:sha(getattr(args,v))for k,v in [('nofix','input_surface'),('qsphere','input_qsphere'),('brain','brain')]},stages=stages,timings=timings,complete_tool_seconds=time.perf_counter()-started,cpu_affinity=sorted(os.sched_getaffinity(0)),threads=args.threads,device_uuid=str(getattr(torch.cuda.get_device_properties(args.device),'uuid','unavailable')),torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),actual_module_sha256={k:sha(v.__file__)for k,v in sys.modules.copy().items()if k.startswith('fnit.')and getattr(v,'__file__','').endswith('.py')},whole_GA='not_assessed',whole_recon_all='not_assessed',final_mesh_quality='not_assessed',production_default_changed=False,**extra)
    write(args.output/'report.private.json',json_safe(report))
    if not passed:raise RuntimeError('real first defect gate failed; retained outputs')
    write(args.output/'COMPLETE.json',dict(status='complete',report_sha256=sha(args.output/'report.private.json')))
    print(json.dumps({'status':'complete','mode':args.mode,'backend':args.backend,'seconds':report['complete_tool_seconds']}),flush=True)
if __name__=='__main__':main()
