"""raw nofix/qsphere/MRI→自产首缺陷完整GA；参考只在最终写回之后读取。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.freesurfer.io as fsio

from fnit.recon_all.topology_genetic_base_prepare import prepare_nofix_genetic_base
from fnit.recon_all.topology_global_priors import global_topology_priors
from fnit.recon_all.topology_candidate_inputs import (prepare_candidate_input_context,
    prepare_defect_candidate_inputs)
from fnit.recon_all.topology_genetic_defect_inputs import ordered_genetic_defect_inputs
from fnit.recon_all.topology_genetic_search import search_and_finalize_defect,TopologyRandomStream
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
from fnit.recon_all.thread_budget import thread_budget
import fnit.recon_all.topology_candidate_cycle as cycle

from benchmark_recon_topology_prepared_cache import candidate_payload
from benchmark_recon_topology_candidate_state import dump_state_checkpoint
from benchmark_recon_topology_patch_fitness import difference,json_safe
from benchmark_recon_topology_mri_likelihood import load_statistics_capture
from benchmark_recon_topology_unmri_fields import load_volume_capture


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, input_surface,input_qsphere,brain,hemisphere,output,threads,
        reference_statistics,reference_volume,reference_fitness,
        population_size,max_unchanged,niters,seed,initial_selection,vertex_eliminate):
    """三项自身输入/显式策略→完整首缺陷搜索与初始候选事后参考诊断。

    所有路径是Path；output不存在。输入网格为相同有序nofix/qsphere，
    MRI来自同一FNIT连续上游。不加载before属性/边表/PDF/最佳排序。
    原metadata仅从自身surface footer和fresh triangular规则派生。
    所有候选与最终结果写出后再读取三项独立参考，只核首候选；
    完整GA赢家/最终参考未捕获，不能把首候选零误差当最终严格复现。
    只运行首缺陷完整搜索（包括声明的停止/剔除和最终写回），不截断
    迭代来宣称默认通过；niters/其他策略和实际counts全部入报告。
    发生异常写failure.json/trace，继续抛出，不生成COMPLETE。
    """
    started=time.perf_counter();output=Path(output)
    if output.exists():raise FileExistsError('new raw genetic search diagnostic directory required')
    output.mkdir(parents=True)
    original_evaluate=cycle.evaluate_candidate;first=None;first_values=None;first_control=None
    records=[];stages={}
    def observe(**kw):
        nonlocal first,first_values,first_control
        tick=time.perf_counter();result=original_evaluate(**kw)
        if first is None:
            first=result;values,first_control=candidate_payload(result)
            first_values={k:np.array(v,copy=True)for k,v in values.items()}
        records.append(dict(candidate=len(records),fitness=result['fitness']['fitness'],
            candidate_seconds=time.perf_counter()-tick,subseconds=result['seconds']))
        with (output/'progress.jsonl').open('a')as f:f.write(json.dumps(json_safe(records[-1]),allow_nan=False)+'\n')
        return result
    try:
        tick=time.perf_counter()
        raw,faces,geometry=fsio.read_geometry(str(input_surface),read_metadata=True)
        sphere,sfaces=fsio.read_geometry(str(input_qsphere))
        raw,faces,sphere=np.asarray(raw,np.float32),np.asarray(faces,np.int32),np.asarray(sphere,np.float32)
        if raw.shape!=sphere.shape or not np.array_equal(faces,sfaces):raise ValueError('same ordered own nofix/qsphere required')
        if hemisphere not in ('lh','rh'):raise ValueError('explicit lh or rh required')
        image=nib.load(brain);volume=np.asarray(image.dataobj)
        matrix=source_topology_ras_to_voxel(volume_header=image.header)
        stages['load_seconds']=time.perf_counter()-tick
        metadata=dict(use_real_ras=0,hemisphere=0 if hemisphere=='lh'else 1,
            status=0,origxyz_status=0,volume_geometry=dict(geometry))
        with thread_budget(threads=threads):
            tick=time.perf_counter()
            base=prepare_nofix_genetic_base(vertices=raw,faces=faces,qsphere_vertices=sphere,
                volume=volume,ras_to_voxel=matrix,source_metadata=metadata)
            stages['own_base_seconds']=time.perf_counter()-tick
            state=base['surface_state'];labels=base['merge']['labels'];translation=base['vertex_translation']
            source=base['prepared']['base_arguments'];metric=base['metric']
            tick=time.perf_counter()
            priors=global_topology_priors(principal_curvature_fields=base['sff1'],
                smoothed_original_vertices=source['source_vertex_fields']['original_vertices'],faces=faces,
                neighbor_offsets=metric['one_ring_offsets'],neighbors=metric['one_ring_neighbors'],
                vertex_rip=metric['sff_arguments']['vertex_rip'],surface_status='sphere')
            stages['own_global_prior_seconds']=time.perf_counter()-tick
            tick=time.perf_counter()
            context=prepare_candidate_input_context(vertices=raw,faces=faces,defect_vertices=labels!=0,
                vertex_translation=translation,volume=volume,ras_to_voxel=matrix,global_priors=priors)
            stages['own_input_context_seconds']=time.perf_counter()-tick
            p=context.preparation
            tick=time.perf_counter()
            edges=ordered_genetic_defect_inputs(canonical_vertices=source['source_vertex_fields']['canonical_vertices'],
                original_vertices=p['original_vertices'],original_normals=p['original_normals'],faces=faces,
                defect_labels=labels,ordered_regions=base['merge']['regions'],retention_status=base['retention_status'],
                vertex_translation=translation,white_values=p['white_values'],gray_values=p['gray_values'],
                volume=volume,ras_to_voxel=matrix,defect_number=0,use_overlap=False)
            stages['own_ordered_edges_seconds']=time.perf_counter()-tick
            tick=time.perf_counter()
            candidate=prepare_defect_candidate_inputs(context=context,original_defect_hull=edges['original_hull_vertices'],
                candidate_edges=edges['edge_vertices'],current_corrected_original_vertices=state['original_vertices'])
            stages['own_local_prior_seconds']=time.perf_counter()-tick
            # No implicit stream reset between generations or final scoring.
            random=TopologyRandomStream(seed=seed);cycle.evaluate_candidate=observe
            tick=time.perf_counter()
            result=search_and_finalize_defect(surface_state=state,defect_inputs=edges,
                candidate_inputs=candidate['candidate_kwargs'],draw_random_number=random,
                population_size=population_size,max_unchanged=max_unchanged,niters=niters,
                initial_selection=initial_selection,vertex_eliminate=vertex_eliminate,
                deletion_count=1,elimination_threshold=np.float32(4),
                check_surface_intersection=False,intersection_checker=None)
            stages['connected_search_and_finalization_seconds']=time.perf_counter()-tick
        cycle.evaluate_candidate=original_evaluate
        tick=time.perf_counter()
        if first_values is None:raise RuntimeError('no independently computed first candidate')
        np.savez_compressed(output/'first_candidate.private.npz',**first_values)
        dump_state_checkpoint(output/'final_state.private.npz',result['surface_state'])
        search=result['search']
        np.savez_compressed(output/'search.private.npz',**{k:search[k]for k in (
            'best_ordering','best_discarded','population_orderings','population_fitness','population_ranks',
            'vertex_rip','use_counts','vertex_fitness')})
        (output/'search_control.json').write_text(json.dumps(json_safe({k:search[k]for k in (
            'counts','evaluation_trace','generation_trace','termination','generations','best_fitness',
            'deletion_count','elimination_threshold')}),indent=2,allow_nan=False)+'\n')
        stages['write_candidate_and_final_seconds']=time.perf_counter()-tick
        # This is the first time any reference path is opened.
        reference=load_statistics_capture(capture_path=reference_statistics)
        v=load_volume_capture(capture_path=reference_volume)
        f=json.loads(Path(reference_fitness).read_text())
        xyz=result['surface_state']['original_vertices']
        core=np.asarray([first['fitness']['composition']['tp_fields'][k]for k in (
            'face_ll','vertex_ll','curv_ll','qcurv_ll','unmri_ll')],np.float32)
        source_bits=np.asarray(f['tp_field_uint32_bits'],np.uint32)
        checks=dict(first_core=difference(core,source_bits[[0,1,2,3,5]].view(np.float32)),
            first_fitness=difference(np.asarray(first['fitness']['fitness'],np.float64),np.asarray(f['fitness'],np.float64)),
            first_coordinates=difference(first['geometry']['original_vertices'],reference['vertex_fields'][:,:3]),
            first_ordered_faces=difference(first['geometry']['faces'],reference['faces']),
            first_normals=difference(first['vertex_normals'],reference['vertex_fields'][:,6:9]),
            first_k1=difference(first['sff']['k1'],reference['vertex_fields'][:,10]),
            first_k2=difference(first['sff']['k2'],reference['vertex_fields'][:,11]),
            first_distance=difference(first['fitness']['distance']['distance_mm'],v['signed_distance']),
            first_curvbak=difference(first['vertex_curvbak'],v['reference_curvbak']))
        passed=all(c['byte_equal']for c in checks.values())
        report=dict(status='complete_self_generated_first_defect_search'if passed else'first_source_candidate_gate_failed',
            scope='raw nofix/qsphere/MRI -> self-generated base/global/ETable/calibration/local prior -> complete declared first defect GA/finalization',
            reference_only_read_after_all_outputs=True,frozen_before_inputs=False,all_algorithm_inputs_self_generated=True,
            strict_first_source_candidate_passed=passed,first_source_checks=checks,
            full_source_final_reference='not_assessed',whole_mesh_quality='not_assessed',
            complete_ga=False,complete_first_defect=True,whole_recon_all='not_assessed',
            overall_metric_equivalence='not_assessed',production_default_changed=False,
            input_sha256={k:sha(p)for k,p in dict(nofix=input_surface,qsphere=input_qsphere,brain=brain).items()},
            reference_sha256={k:sha(p)for k,p in dict(statistics=reference_statistics,volume=reference_volume,fitness=reference_fitness).items()},
            actual_module_sha256={name:sha(m.__file__)for name,m in sorted(sys.modules.items())
                if name.startswith('fnit.')and getattr(m,'__file__',None)and str(m.__file__).endswith('.py')},
            tool_sha256=sha(__file__),seed=seed,random_draws=random.draws,threads=threads,
            cpu_affinity=sorted(os.sched_getaffinity(0)),target_cuda='hidden',gpu_memory_bytes=None,
            base_vertices=len(xyz),base_faces=state['nfaces'],all_defects=len(base['merge']['regions']),
            first_defect_edges=len(edges['edge_vertices']),first_defect_vertices=len(edges['inside_vertices']),
            actual_policy=result['actual_policy'],search_counts=search['counts'],search_termination=search['termination'],
            search_generations=search['generations'],best_fitness=search['best_fitness'],
            actual_final_fitness=result['finalization']['fitness'],fitness_matches_best=result['finalization']['fitness_matches_best'],
            final_faces=result['surface_state']['nfaces'],final_additional_discarded=result['finalization']['cleanup']['additional_discarded_count'],
            stages=stages,search_api_seconds=result['seconds'],candidate_timings=records,
            complete_tool_before_final_report_seconds=time.perf_counter()-started,
            checkpoint_sha256={name:sha(output/name)for name in (
                'first_candidate.private.npz','final_state.private.npz','search.private.npz','search_control.json')})
        (output/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
        if not passed:raise RuntimeError('source first candidate differs; result and report retained')
        (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],report_sha256=sha(output/'report.json')))+'\n')
        return report
    except Exception as e:
        (output/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,message=str(e),
            stages=stages,candidates_evaluated=len(records),partial_wall_seconds=time.perf_counter()-started,
            whole_recon_all='not_assessed'),indent=2,allow_nan=False)+'\n')
        raise
    finally:cycle.evaluate_candidate=original_evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('input-surface','input-qsphere','brain','output','reference-statistics','reference-volume','reference-fitness'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--hemisphere',choices=['lh','rh'],required=True)
    for k in ('threads','population-size','max-unchanged','niters','seed'):p.add_argument('--'+k,type=int,required=True)
    p.add_argument('--initial-selection',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--vertex-eliminate',action=argparse.BooleanOptionalAction,default=True)
    report=run(**vars(p.parse_args()))
    print(json.dumps(json_safe({k:report[k]for k in ('status','search_counts','search_generations','best_fitness','actual_final_fitness','stages')}),indent=2))


if __name__=='__main__':main()
