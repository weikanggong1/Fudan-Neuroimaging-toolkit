"""从自身第三缺陷检查点独立重放第四缺陷首候选；只诊断，不改完整GA。"""
from __future__ import annotations
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import resource
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(*, input_surface, input_qsphere, brain, state_checkpoint,
        producer_progress, frozen_source_manifest, expected_source_manifest_sha256,
        expected_checkpoint_sha256, output, defect_number, threads,
        blocker_scan_backend='python',audit_sequence=False,skip_profile=False):
    """仅使用自身完整checkpoint+原MRI/表面重新准备同首候选。

    三个原输入为同链nofix/qsphere/brain的Path；state_checkpoint为前一
    缺陷完整写回的无pickle typed NPZ，producer_progress核其SHA/缺陷编号。
    manifest与两个expected SHA必须匹配实际冻结字节；defect_number仅
    接受3，threads总预算4；output必须为新目录。此工具不接受旧赢家、
    原程序或参考after路径。先自产同raw base/global priors/校准，再从
    typed checkpoint恢复全部持久K/E/fx/curvbak/有序容量，生成该缺陷
    当前边和local priors。首population_index=0按成熟源规则是identity，
    不消费随机；不伪造前面三缺陷的随机历史或继续完整GA搜索。

    三次同初态首候选独立执行：cold、warm、warm+cProfile；保存现有
    evaluate_candidate/geometry/五项/闭环实际timers、返回state/几何/反馈
    全字段自重复及剖析。cProfile秒含观测开销，不是未剖析吞吐。输入
    不修改；不存在完整初态则明确失败，不从参考补字段。不进行最终
    GA写回、SOAP、压缩、remesh/相交或全质量验收。
    blocker_scan_backend默认python，numba显式传给成熟重连接口；
    audit_sequence仅第三次逐条核谓词序列，skip_profile=True时第三次
    不开cProfile。普通冷/热调用均关闭序列审计，不把审计当吞吐。
    """
    start=time.perf_counter()
    if output.exists(): raise FileExistsError('fresh diagnostic directory required')
    output.mkdir(parents=True)
    if defect_number != 3 or threads != 4:
        raise ValueError('this frozen protocol requires defect index3 and total4 threads')
    if sha(frozen_source_manifest)!=expected_source_manifest_sha256:
        raise ValueError('actual independent frozen source manifest mismatch')
    if sha(state_checkpoint)!=expected_checkpoint_sha256:
        raise ValueError('complete own previous-defect checkpoint mismatch')
    progress=[json.loads(row) for row in producer_progress.read_text().splitlines() if row]
    matching=[r for r in progress if r['defect_number']==defect_number-1]
    if len(matching)!=1 or matching[0]['checkpoint_sha256']!=expected_checkpoint_sha256:
        raise ValueError('previous complete own producer progress/checkpoint binding required')
    import numpy as np
    import nibabel as nib
    import nibabel.freesurfer.io as fsio
    from fnit.recon_all.thread_budget import thread_budget
    from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
    from fnit.recon_all.topology_genetic_base_prepare import prepare_nofix_genetic_base
    from fnit.recon_all.topology_global_priors import global_topology_priors
    from fnit.recon_all.topology_candidate_inputs import prepare_candidate_input_context,prepare_defect_candidate_inputs
    from fnit.recon_all.topology_genetic_defect_inputs import prepare_ordered_genetic_defect_context
    from fnit.recon_all.topology_genetic_next_defect_inputs import next_ordered_genetic_defect_inputs
    from fnit.recon_all.topology_genetic_search import prepare_genetic_defect_evaluator
    from fnit.recon_all.topology_face_detection_cache import prepare_face_detection_context
    from fnit.recon_all.topology_ga_population import generate_initial_ordering
    from benchmark_recon_topology_candidate_state import load_state_checkpoint,dump_state_checkpoint
    tick=time.perf_counter()
    raw,faces,geometry=fsio.read_geometry(str(input_surface),read_metadata=True)
    sphere,sfaces=fsio.read_geometry(str(input_qsphere))
    raw=np.asarray(raw,np.float32);sphere=np.asarray(sphere,np.float32);faces=np.asarray(faces,np.int32)
    if not np.array_equal(faces,sfaces) or raw.shape!=sphere.shape:
        raise ValueError('same own nofix/qsphere indexed surfaces required')
    image=nib.load(brain);volume=np.asarray(image.dataobj)
    matrix=source_topology_ras_to_voxel(volume_header=image.header)
    metadata=dict(use_real_ras=0,hemisphere=0,status=0,origxyz_status=0,volume_geometry=dict(geometry))
    state=load_state_checkpoint(state_checkpoint)
    loading=time.perf_counter()-tick;stages={};records=[]
    with thread_budget(threads=threads):
        tick=time.perf_counter()
        base=prepare_nofix_genetic_base(vertices=raw,faces=faces,qsphere_vertices=sphere,
            volume=volume,ras_to_voxel=matrix,source_metadata=metadata)
        stages['own_raw_base']=time.perf_counter()-tick
        # The source has not compacted yet; all attributes and capacities must survive.
        if (not set(base['surface_state']).issubset(state) or len(state['original_vertices'])!=len(raw)
                or not np.array_equal(state['canonical_vertices'],base['surface_state']['canonical_vertices'])):
            raise ValueError('typed checkpoint lacks full same-mapping precompact state')
        source=base['prepared']['base_arguments'];metric=base['metric']
        labels=base['merge']['labels'];translation=base['vertex_translation'];regions=base['merge']['regions']
        tick=time.perf_counter()
        priors=global_topology_priors(principal_curvature_fields=base['sff1'],
            smoothed_original_vertices=source['source_vertex_fields']['original_vertices'],faces=faces,
            neighbor_offsets=metric['one_ring_offsets'],neighbors=metric['one_ring_neighbors'],
            vertex_rip=metric['sff_arguments']['vertex_rip'],surface_status='sphere')
        context=prepare_candidate_input_context(vertices=raw,faces=faces,defect_vertices=labels!=0,
            vertex_translation=translation,volume=volume,ras_to_voxel=matrix,global_priors=priors)
        edge_context=prepare_ordered_genetic_defect_context(canonical_vertices=source['source_vertex_fields']['canonical_vertices'],
            faces=faces,defect_labels=labels,ordered_regions=regions,vertex_translation=translation)
        stages['own_global_priors_and_shared_inputs']=time.perf_counter()-tick
        p=context.preparation;tick=time.perf_counter()
        edges=next_ordered_genetic_defect_inputs(canonical_vertices=source['source_vertex_fields']['canonical_vertices'],
            original_vertices=p['original_vertices'],original_normals=p['original_normals'],faces=faces,
            defect_labels=labels,ordered_regions=regions,retention_status=base['retention_status'],
            vertex_translation=translation,white_values=p['white_values'],gray_values=p['gray_values'],
            volume=volume,ras_to_voxel=matrix,defect_number=defect_number,
            current_surface_state=state,use_overlap=False,context=edge_context)
        candidate=prepare_defect_candidate_inputs(context=context,original_defect_hull=edges['original_hull_vertices'],
            candidate_edges=edges['edge_vertices'],current_corrected_original_vertices=state['original_vertices'])
        stages['fourth_dynamic_edges_and_local_priors']=time.perf_counter()-tick
        first_arrays=None
        def forbid_draw(low,high):
            raise RuntimeError('source initial population_index0 must not draw RNG')
        last_name='audit' if skip_profile and audit_sequence else ('repeat' if skip_profile else 'profile')
        first_control=None
        for name in ('cold','warm',last_name):
            tick=time.perf_counter()
            prepared=prepare_genetic_defect_evaluator(surface_state=state,defect_inputs=edges,
                candidate_inputs=candidate['candidate_kwargs'])
            evaluator=prepared['evaluator'];r=evaluator.retessellation_inputs
            r['face_detection_context']=prepare_face_detection_context(base_faces=r['base_faces'],vertex_count=len(raw))
            r['blocker_scan_backend']=blocker_scan_backend
            r['blocker_scan_audit']=bool(audit_sequence and name==last_name)
            preparation_seconds=time.perf_counter()-tick
            ordering=generate_initial_ordering(edge_count=len(edges['edge_vertices']),population_index=0,
                initial_selection=True,segment_edge_numbers=prepared['segment_edge_numbers'],
                segment_rows=prepared['segment_rows'],**prepared['overlap_arguments'],draw_random_number=forbid_draw)
            args=dict(ordering=ordering['ordering'],discarded_defect_vertices=r['inside_discarded'],
                vertex_rip=np.asarray(state['vertex_rip'],np.int32),
                use_counts=np.zeros(len(r['inside_vertices']),np.int32),
                vertex_fitness=np.zeros(len(r['inside_vertices']),np.float32))
            profile=cProfile.Profile() if name=='profile' else None
            tick=time.perf_counter()
            if profile:profile.enable()
            feedback=evaluator(**args)
            if profile:profile.disable()
            elapsed=time.perf_counter()-tick;last=evaluator.last_result;actual=last['candidate']
            data={'fitness':np.asarray(feedback['fitness'],np.float64),
                'vertex_rip':feedback['vertex_rip'],'use_counts':feedback['use_counts'],
                'vertex_fitness':feedback['vertex_fitness'],'edge_used':last['edge_used']}
            data.update({key:value for key,value in actual['geometry'].items() if isinstance(value,np.ndarray)})
            data.update({'sff_'+key:value for key,value in actual['sff'].items() if isinstance(value,np.ndarray)})
            np.savez_compressed(output/(name+'.private.npz'),**data)
            dump_state_checkpoint(output/(name+'.state.private.npz'),evaluator.surface_state)
            arrays=dict(data)
            with np.load(output/(name+'.state.private.npz'),allow_pickle=False) as z:
                arrays.update({'state_'+key:z[key].copy() for key in z.files})
            equal=first_arrays is None or set(arrays)==set(first_arrays) and all(
                arrays[key].dtype==first_arrays[key].dtype and arrays[key].shape==first_arrays[key].shape
                and arrays[key].tobytes()==first_arrays[key].tobytes() for key in arrays)
            if first_arrays is None:first_arrays=arrays
            geometry=actual['geometry']
            control=dict(retessellation_trace=geometry['retessellation_trace'],
                face_decisions=geometry['face_decisions'],rounds=geometry['rounds'],
                ninside=geometry['ninside'],ndiscarded=geometry['ndiscarded'])
            control_equal=first_control is None or control==first_control
            if first_control is None:first_control=control
            write(output/(name+'.control.private.json'),control)
            records.append(dict(run=name,preparation_seconds=preparation_seconds,complete_cycle_seconds=elapsed,
                fitness=feedback['fitness'],cycle_seconds=last['seconds'],candidate_seconds=actual['seconds'],
                geometry_seconds=actual['geometry']['stage_seconds'],five_seconds=actual['fitness'].get('seconds'),
                retessellation_seconds=geometry.get('retessellation_seconds'),face_detection_seconds=geometry.get('face_detection_seconds'),
                blocker_scan_backend=geometry.get('blocker_scan_backend'),blocker_scan_preparation_seconds=geometry.get('blocker_scan_preparation_seconds'),
                blocker_scan_calls=geometry.get('blocker_scan_calls'),blocker_predicate_calls=geometry.get('blocker_predicate_calls'),
                blocker_predicate_sequence_sha256=geometry.get('blocker_predicate_sequence_sha256'),
                complete_control_sha256=sha(output/(name+'.control.private.json')),all_repeated_controls_equal=control_equal,
                repeat_field_count=len(arrays),all_repeated_bytes_equal=equal,
                candidate_sha256=sha(output/(name+'.private.npz')),state_sha256=sha(output/(name+'.state.private.npz'))))
            if profile:
                profile.dump_stats(str(output/'candidate.cprofile.private'))
                stats=pstats.Stats(profile)
                rows=[]
                for (filename,line,function),(primitive,total,self_seconds,cumulative,callers) in stats.stats.items():
                    rows.append(dict(module=Path(filename).name,line=line,function=function,
                        primitive_calls=primitive,total_calls=total,self_seconds=self_seconds,cumulative_seconds=cumulative))
                write(output/'profile.public.json',dict(top_self=sorted(rows,key=lambda x:x['self_seconds'],reverse=True)[:30],
                    top_cumulative=sorted(rows,key=lambda x:x['cumulative_seconds'],reverse=True)[:30],
                    profile_overhead_included=True))
    modules={name:sha(module.__file__) for name,module in sys.modules.items()
        if name.startswith('fnit.') and getattr(module,'__file__',None) and module.__file__.endswith('.py')}
    report=dict(status='complete_fixed_fourth_initial_candidate_diagnostic',defect_number=defect_number,
        population_index=0,identity_ordering=True,actual_rng_draws=0,
        checkpoint_complete_fields_restored=True,checkpoint_fields_count=len(state),
        initial_state_from_own_previous_complete_checkpoint=True,initial_attributes_from_reference=False,
        shared_priors_and_local_mri_self_generated=True,ordered_edge_count=len(edges['edge_vertices']),
        inside_count=len(edges['inside_vertices']),hull_count=len(edges['hull_vertices']),
        local_volume_shape=list(map(int,candidate['candidate_kwargs']['local_priors']['volume_shape'])),
        input_sha256=dict(nofix=sha(input_surface),qsphere=sha(input_qsphere),brain=sha(brain),
            checkpoint=sha(state_checkpoint),producer_progress_snapshot=sha(producer_progress)),
        frozen_source_manifest_sha256=sha(frozen_source_manifest),actual_tool_sha256=sha(__file__),
        actual_module_sha256=modules,records=records,stages_seconds=stages,loading_seconds=loading,
        complete_tool_seconds_before_report=time.perf_counter()-start,
        threads=threads,cpu_affinity=sorted(os.sched_getaffinity(0)),cuda_hidden=os.environ.get('CUDA_VISIBLE_DEVICES')=='',
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        full_ga=False,final_mesh_quality='not_assessed',overall_equivalence='not_assessed',
        whole_recon_all='not_assessed',all_repeated_fields_byte_equal=all(r['all_repeated_bytes_equal'] for r in records))
    report['blocker_scan_backend']=blocker_scan_backend
    report['sequence_audit_enabled_only_last']=audit_sequence
    report['all_repeated_controls_equal']=all(r['all_repeated_controls_equal'] for r in records)
    write(output/'report.private.json',report)
    if not report['all_repeated_fields_byte_equal'] or not report['all_repeated_controls_equal']:
        raise RuntimeError('same first-candidate repeat fields or controls differ')
    write(output/'COMPLETE.json',dict(status=report['status'],report_sha256=sha(output/'report.private.json')))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-surface','input-qsphere','brain','state-checkpoint','producer-progress','frozen-source-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-source-manifest-sha256','expected-checkpoint-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--defect-number',type=int,required=True)
    parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--blocker-scan-backend',choices=('python','numba'),default='python')
    parser.add_argument('--audit-sequence',action='store_true')
    parser.add_argument('--skip-profile',action='store_true')
    args=vars(parser.parse_args())
    try:run(**args)
    except Exception as error:
        if args['output'].exists():write(args['output']/'failure.private.json',dict(type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':main()
