"""显式库距离后端：原始nofix→五次初始评分+最终评分/几何写回同输入；不并行候选。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def validate_source(source):
    manifest=Path(source)/'SOURCE_MANIFEST.private.json'
    value=json.loads(manifest.read_text())
    for name,record in value['payload'].items():
        path=Path(source)/name
        if path.stat().st_size!=record['size'] or sha(path)!=record['sha256']:
            raise ValueError('immutable stage source changed: '+name)
    tool=Path(__file__).resolve()
    expected=Path(source).resolve()/'tools'/tool.name
    if tool!=expected or sha(tool)!=value['payload']['tools/'+tool.name]['sha256']:
        raise ValueError('actual diagnostic entry must be the declared frozen tool')
    return value,sha(manifest)


def validate_loaded_source(source,manifest):
    """已加载FNIT和具名benchmark/export helpers只能来自声明冻结文件。"""
    source=Path(source).resolve()
    loaded={}
    for name,module in sorted(sys.modules.copy().items()):
        if not (name=='fnit' or name.startswith('fnit.') or name.startswith(('benchmark_recon_','export_recon_'))):
            continue
        filename=getattr(module,'__file__',None)
        if filename is None:continue
        path=Path(filename).resolve()
        if not path.is_relative_to(source):
            raise ValueError('installed or undeclared helper used: '+name)
        key=str(path.relative_to(source))
        if key not in manifest['payload'] or sha(path)!=manifest['payload'][key]['sha256']:
            raise ValueError('loaded source payload not declared: '+name)
        loaded[name]=sha(path)
    return loaded


def worker(*,frozen_source,input_surface,input_qsphere,brain,hemisphere,output,threads,seed,
        distance_backend,complete_default_ga=False):
    """自有原始三输入→新显式search API，保存后态；不打开旧候选或参考。

    Path参数具名必填，output不存在。面和坐标来自同域nofix/qsphere，
    brain为同链uint8 MRI；读取/单位与完整首缺陷工具相同。固定阶段
    population_size=5,niters=0,max_unchanged=10，保留初始5次完整评分
    及最终1次评分/完整几何写回。默认短模式不是495轮或全9缺陷验证；仅 complete_default_ga=True
    选择默认10/-1，并完整要求495+final/29条trace，不借旧winner。
    distance_backend为serial/parallel，直接传search_and_finalize_defect；两侧保留面缓存True及Numba阻挡扫描。
    没有猴补prepare/search/工厂/评分函数。RNG包装只调用成熟流一次，
    记录全部实际区间和值的字节SHA，prime仍由成熟构造器处理。

    保存完整最终candidate/state/search、六评分、全部控制JSON与实际
    源码/输入SHA和各段秒数。所有字段/旧新输出只在两worker结束后比较。
    包括源验证、读写、JIT和准备；未运行整例/最终网格质量检查。
    """
    if type(complete_default_ga) is not bool:raise ValueError('complete_default_ga must be bool')
    population_size,niters=(10,-1)if complete_default_ga else(5,0)
    expected_evaluations,expected_scores=(495,496)if complete_default_ga else(5,6)
    started=time.perf_counter();out=Path(output)
    if threads!=2:raise ValueError('this short protocol requires exactly two caller threads')
    if not isinstance(distance_backend,str) or distance_backend not in ('serial','parallel'):
        raise ValueError('explicit serial/parallel distance_backend required')
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    source,manifest_sha=validate_source(frozen_source)
    sys.path[:0]=[str(Path(frozen_source)/'src'),str(Path(frozen_source)/'tools')]
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(Path(frozen_source)/'src/fnit/recon_all'))
    import numpy as np
    import nibabel as nib
    import nibabel.freesurfer.io as fsio
    from fnit.recon_all.topology_genetic_base_prepare import prepare_nofix_genetic_base
    from fnit.recon_all.topology_global_priors import global_topology_priors
    from fnit.recon_all.topology_candidate_inputs import prepare_candidate_input_context,prepare_defect_candidate_inputs
    from fnit.recon_all.topology_genetic_defect_inputs import ordered_genetic_defect_inputs
    from fnit.recon_all.topology_genetic_search import search_and_finalize_defect,TopologyRandomStream
    from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
    from fnit.recon_all.thread_budget import thread_budget
    from benchmark_recon_topology_prepared_cache import candidate_payload
    from benchmark_recon_topology_candidate_state import dump_state_checkpoint
    from benchmark_recon_topology_patch_fitness import json_safe
    loaded={}
    for name,expected in source['expected_module_sha256'].items():
        module=importlib.import_module(name);actual=sha(module.__file__)
        if actual!=expected or not Path(module.__file__).resolve().is_relative_to((Path(frozen_source)/'src').resolve()):
            raise ValueError('actual frozen module SHA differs: '+name)
        loaded[name]=actual
    validate_loaded_source(frozen_source,source)
    stages={'import_and_source_validation':time.perf_counter()-started}
    tick=time.perf_counter()
    raw,faces,geometry=fsio.read_geometry(str(input_surface),read_metadata=True)
    canonical,canonical_faces=fsio.read_geometry(str(input_qsphere))
    raw,faces,canonical=np.asarray(raw,np.float32),np.asarray(faces,np.int32),np.asarray(canonical,np.float32)
    if raw.shape!=canonical.shape or not np.array_equal(faces,canonical_faces):
        raise ValueError('same ordered own nofix/qsphere required')
    image=nib.load(brain);volume=np.asarray(image.dataobj)
    matrix=source_topology_ras_to_voxel(volume_header=image.header)
    metadata=dict(use_real_ras=0,hemisphere=0 if hemisphere=='lh'else 1,status=0,
        origxyz_status=0,volume_geometry=dict(geometry))
    stages['load']=time.perf_counter()-tick
    rng=TopologyRandomStream(seed=seed);random_hash=hashlib.sha256();observed_draws=0
    def draw(low,high):
        nonlocal observed_draws
        value=rng(low,high)
        random_hash.update(struct.pack('<Qddd',observed_draws,float(low),float(high),value))
        observed_draws+=1
        return value
    with thread_budget(threads=threads) as budget:
        tick=time.perf_counter()
        base=prepare_nofix_genetic_base(vertices=raw,faces=faces,qsphere_vertices=canonical,
            volume=volume,ras_to_voxel=matrix,source_metadata=metadata)
        stages['raw_base']=time.perf_counter()-tick
        state=base['surface_state'];labels=base['merge']['labels'];translation=base['vertex_translation']
        original=base['prepared']['base_arguments'];metric=base['metric']
        tick=time.perf_counter()
        priors=global_topology_priors(principal_curvature_fields=base['sff1'],
            smoothed_original_vertices=original['source_vertex_fields']['original_vertices'],faces=faces,
            neighbor_offsets=metric['one_ring_offsets'],neighbors=metric['one_ring_neighbors'],
            vertex_rip=metric['sff_arguments']['vertex_rip'],surface_status='sphere')
        context=prepare_candidate_input_context(vertices=raw,faces=faces,defect_vertices=labels!=0,
            vertex_translation=translation,volume=volume,ras_to_voxel=matrix,global_priors=priors)
        stages['global_prior_and_input_context']=time.perf_counter()-tick
        p=context.preparation;tick=time.perf_counter()
        edges=ordered_genetic_defect_inputs(canonical_vertices=original['source_vertex_fields']['canonical_vertices'],
            original_vertices=p['original_vertices'],original_normals=p['original_normals'],faces=faces,
            defect_labels=labels,ordered_regions=base['merge']['regions'],retention_status=base['retention_status'],
            vertex_translation=translation,white_values=p['white_values'],gray_values=p['gray_values'],
            volume=volume,ras_to_voxel=matrix,defect_number=0,use_overlap=False)
        stages['ordered_edges']=time.perf_counter()-tick
        tick=time.perf_counter()
        candidate=prepare_defect_candidate_inputs(context=context,original_defect_hull=edges['original_hull_vertices'],
            candidate_edges=edges['edge_vertices'],current_corrected_original_vertices=state['original_vertices'])
        stages['local_prior']=time.perf_counter()-tick
        tick=time.perf_counter()
        result=search_and_finalize_defect(surface_state=state,defect_inputs=edges,
            candidate_inputs=candidate['candidate_kwargs'],draw_random_number=draw,
            population_size=population_size,max_unchanged=10,niters=niters,initial_selection=True,vertex_eliminate=True,
            deletion_count=1,elimination_threshold=np.float32(4),check_surface_intersection=False,
            intersection_checker=None,face_detection_cache=True,blocker_scan_backend='numba',distance_backend=distance_backend)
        stages['search_and_final_API']=time.perf_counter()-tick
    search=result['search'];final=result['finalization']
    tick=time.perf_counter()
    payload,control=candidate_payload(final['final_evaluation']['candidate'])
    np.savez_compressed(out/'final_candidate.private.npz',**payload)
    write(out/'candidate_control.private.json',json_safe(control))
    dump_state_checkpoint(out/'final_state.private.npz',result['surface_state'])
    np.savez_compressed(out/'search.private.npz',**{key:search[key]for key in (
        'best_ordering','best_discarded','population_orderings','population_fitness','population_ranks',
        'vertex_rip','use_counts','vertex_fitness')})
    scores=np.asarray([row['fitness']for row in search['evaluation_trace']]+[final['fitness']],np.float64)
    np.savez_compressed(out/'all_scores.private.npz',fitness=scores)
    scientific_control={key:search[key]for key in ('counts','evaluation_trace','generation_trace',
        'termination','generations','best_fitness','deletion_count','elimination_threshold')}
    scientific_control.update(actual_policy=result['actual_policy'],final_fitness=final['fitness'],
        fitness_matches_best=final['fitness_matches_best'],additional_discarded=final['cleanup']['additional_discarded_count'],
        random_draws=rng.draws,observed_random_draws=observed_draws,random_sequence_sha256=random_hash.hexdigest())
    write(out/'search_control.private.json',json_safe(scientific_control))
    stages['write']=time.perf_counter()-tick
    validate_source(frozen_source)
    actual_loaded_source=validate_loaded_source(frozen_source,source)
    loaded_after={name:sha(importlib.import_module(name).__file__)for name in loaded}
    distance_policy=result['final_distance_policy']
    expected_mask=threads if distance_backend=='parallel' else None
    passed=bool(distance_policy['executed'] and distance_policy['actual_backend']==distance_backend
        and distance_policy['caller_numba_thread_mask']==expected_mask
        and budget['restoration_complete'] and search['counts']['evaluated']==expected_evaluations
        and len(scores)==expected_scores and rng.draws==observed_draws
        and (not complete_default_ga or(len(search['generation_trace'])==29 and search['generations']==28))
        and result['distance_backend']==distance_backend and loaded==loaded_after)
    report=dict(status=('complete_default_first_GA_explicit_distance_API'if complete_default_ga else'complete_six_score_production_API_stage')if passed else'failed_stage',
        distance_backend=distance_backend,distance_policy=distance_policy,thread_budget=budget,
        blocker_scan_backend='numba',face_detection_cache=True,source_manifest_sha256=manifest_sha,
        actual_loaded_module_sha256=loaded_after,
        actual_loaded_source_sha256=actual_loaded_source,diagnostic_entry_sha256=sha(__file__),
        input_sha256={key:sha(path)for key,path in dict(nofix=input_surface,qsphere=input_qsphere,brain=brain).items()},
        policy=dict(population_size=population_size,max_unchanged=10,niters=niters,initial_selection=True,vertex_eliminate=True,
            seed=seed,threads=threads),scores=len(scores),search_evaluations=search['counts']['evaluated'],
        generation_trace_count=len(search['generation_trace']),last_zero_based_generation_index=search['generations'],
        stages=stages,nested_search_seconds=result['seconds'],random_draws=rng.draws,
        random_sequence_sha256=random_hash.hexdigest(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        target_cuda='hidden',allocated_bytes=None,reserved_bytes=None,
        self_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        checkpoint_sha256={name:sha(out/name)for name in (
            'final_candidate.private.npz','final_state.private.npz','search.private.npz','all_scores.private.npz',
            'candidate_control.private.json','search_control.private.json')},
        complete_worker_seconds_before_report=time.perf_counter()-started,
        explicit_production_search_parameter=True,explicit_distance_backend_parameter=True,global_function_replacement=False,
        complete_default_GA=complete_default_ga,whole_recon_all='not_assessed',final_mesh_quality='not_assessed',
        overall_metric_equivalence='not_assessed')
    write(out/'report.private.json',report)
    if not passed:raise RuntimeError('declared complete score stage incomplete; all outputs retained')
    write(out/'COMPLETE.json',dict(status=report['status'],report_sha256=sha(out/'report.private.json')))
    return report


def run_pair(*,frozen_source,input_surface,input_qsphere,brain,hemisphere,output,threads,seed,complete_default_ga=False):
    """两fresh子进程同预算仅选择距离后端；两者全部保存后再比较完整字段。"""
    if type(complete_default_ga)is not bool:raise ValueError('complete_default_ga must be bool')
    started=time.perf_counter();out=Path(output)
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    _,manifest_sha=validate_source(frozen_source);records=[]
    for label,enabled in (('serial','serial'),('parallel','parallel')):
        command=[sys.executable,str(Path(__file__).resolve()),'--worker']
        for name,value in dict(frozen_source=frozen_source,input_surface=input_surface,input_qsphere=input_qsphere,
                brain=brain,hemisphere=hemisphere,output=out/label,threads=threads,seed=seed).items():
            command.extend(('--'+name.replace('_','-'),str(value)))
        command.extend(('--distance-backend',enabled))
        if complete_default_ga:command.append('--complete-default-ga')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NUMBA_CACHE_DIR=str(out/(label+'-numba-cache')),
            **{key:str(threads) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')})
        tick=time.perf_counter()
        with(out/(label+'.private.log')).open('w')as stream:
            child=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        seconds=time.perf_counter()-tick
        write(out/(label+'.EXIT.json'),dict(exit_code=child.returncode,fresh_cli_seconds=seconds))
        if child.returncode:raise RuntimeError('fresh explicit production stage failed; diagnostics retained')
        report=json.loads((out/label/'report.private.json').read_text())
        complete=json.loads((out/label/'COMPLETE.json').read_text())
        if complete['status']!=report['status'] or complete['report_sha256']!=sha(out/label/'report.private.json'):
            raise ValueError('complete producer report binding required')
        records.append(dict(label=label,fresh_cli_seconds=seconds,report=report))
    # First read of completed old/disabled results for comparison; no old state
    # or winner is passed to the enabled computation.
    sys.path[:0]=[str(Path(frozen_source)/'src'),str(Path(frozen_source)/'tools')]
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(Path(frozen_source)/'src/fnit/recon_all'))
    from benchmark_recon_topology_genetic_optimization import compare_npz
    groups={name:compare_npz(candidate=out/'parallel'/name,reference=out/'serial'/name)for name in (
        'final_candidate.private.npz','final_state.private.npz','search.private.npz','all_scores.private.npz')}
    from export_recon_topology_genetic_distance_wiring import same_json
    manifest,_=validate_source(frozen_source)
    validate_loaded_source(frozen_source,manifest)
    controls={name:same_json(json.loads((out/'serial'/name).read_text()),json.loads((out/'parallel'/name).read_text()))
        for name in ('candidate_control.private.json','search_control.private.json')}
    passed=bool(records[0]['report']['random_sequence_sha256']==records[1]['report']['random_sequence_sha256']
        and records[0]['report']['cpu_affinity']==records[1]['report']['cpu_affinity']
        and len(records[0]['report']['cpu_affinity'])==threads and all(value['all_fields_byte_equal']and value['complete_file_sha_equal']for value in groups.values())
        and all(controls.values()) and records[0]['report']['input_sha256']==records[1]['report']['input_sha256'])
    report=dict(status=('passed_default_first_GA_explicit_distance_pair'if complete_default_ga else'passed_six_score_explicit_distance_pair')if passed else'failed_pair',
        source_manifest_sha256=manifest_sha,records=records,comparisons=groups,controls=controls,
        compared_numeric_fields=sum(len(value['fields'])for value in groups.values()),
        complete_pair_wall_seconds=time.perf_counter()-started,production_default_changed=False,global_function_replacement=False,
        actual_random_bounds_and_values_sha256=records[0]['report']['random_sequence_sha256'],
        complete_default_GA=complete_default_ga,whole_recon_all='not_assessed',final_mesh_quality='not_assessed',
        overall_metric_equivalence='not_assessed',algorithm_reference_inputs=False)
    write(out/'pair.private.json',report)
    if not passed:raise RuntimeError('explicit distance stage differs; outputs retained')
    write(out/'COMPLETE.json',dict(status=report['status'],report_sha256=sha(out/'pair.private.json')))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-source','input-surface','input-qsphere','brain','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--hemisphere',choices=('lh','rh'),required=True)
    parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--complete-default-ga',action='store_true',help='完整首缺陷默认10/-1；要求495+final、29条代际trace')
    parser.add_argument('--distance-backend',choices=('serial','parallel'),default='serial')
    options=vars(parser.parse_args());child=options.pop('worker');enabled=options.pop('distance_backend')
    report=worker(distance_backend=enabled,**options)if child else run_pair(**options)
    print(json.dumps({key:report[key]for key in ('status',) },indent=2))


if __name__=='__main__':
    main()
