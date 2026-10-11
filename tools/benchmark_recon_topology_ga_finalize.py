"""冻结真实输入：自产候选best→最终写回重复；参考只检查首候选。"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

import benchmark_recon_topology_candidate_pipeline as benchmark
from benchmark_recon_topology_candidate_cycle import build_frozen_cycle
from benchmark_recon_topology_candidate_state import dump_state_checkpoint
from benchmark_recon_topology_patch_fitness import difference, json_safe
from benchmark_recon_topology_prepared_cache import candidate_payload
from benchmark_recon_topology_retess_fields import load_retessellation_capture
from fnit.recon_all.topology_ga_finalize import finalize_defect_patch


def run(**kwargs):
    """同原候选工具10路径/threads参数→两条相同冻结末段诊断的新报告。

    原MRI/no-fix/qsphere自产校准，before属性及全局prior显式冻结。
    每条先自行完整评分一次建立本条best_fitness，随后从它的连续
    状态最终评分/恢复→完整重连/MRI匹配→清理；不把参考fitness或
    后态当best/初态，不宣称这条fixed-ordering是完整GA最优。
    原工具在全部调用/写出之后检查两条首候选，最终写回仅做自身
    全字段重复及对自己首候选几何的比较，没有完整source final捕获。

    output必须不存在，保存完整私有NPZ/JSON与可公开数值摘要。
    全诊断墙钟含加载、一次校准、两条完整序列和读写/事后比较；
    API子步骤另计，不能合称原始T1整例。线程/输入验收沿用原工具；
    子错误保留已写产物并抛出，不忽略初候选strict gate失败。
    """
    start=time.perf_counter();output=Path(kwargs['output'])
    if output.exists():raise FileExistsError('new finalization diagnostic directory required')
    before=load_retessellation_capture(capture_path=kwargs['before_retess'])
    original=benchmark.evaluate_candidate
    records=[];payloads=[];controls=[];calls=0
    def observed(**kw):
        nonlocal calls
        label='A' if calls==0 else 'B';calls+=1
        if calls>2:raise RuntimeError('only two declared independent sequences expected')
        inputs={key:value for key,value in kw.items()if key not in ('surface_state','retessellation_inputs')}
        tick=time.perf_counter()
        evaluator=build_frozen_cycle(surface_state=kw['surface_state'],
            retessellation_inputs=kw['retessellation_inputs'],vertex_marks=before['vertex_flags'][:,0],
            candidate_inputs=inputs)
        preparation_seconds=time.perf_counter()-tick
        r=evaluator.retessellation_inputs
        count=np.zeros(len(r['inside_vertices']),np.int32)
        feedback=np.zeros(len(count),np.float32)
        tick=time.perf_counter()
        initial=evaluator(ordering=r['ordering'],discarded_defect_vertices=r['inside_discarded'],
            vertex_rip=evaluator.surface_state['vertex_rip'],use_counts=count,vertex_fitness=feedback)
        initial_seconds=time.perf_counter()-tick
        first=evaluator.last_result['candidate']
        tick=time.perf_counter()
        result=finalize_defect_patch(surface_state=evaluator.surface_state,snapshot=evaluator.snapshot,
            retessellation_inputs=evaluator.retessellation_inputs,candidate_inputs=inputs,
            best_ordering=r['ordering'],best_discarded=r['inside_discarded'],best_fitness=initial['fitness'],
            use_counts=initial['use_counts'],vertex_fitness=initial['vertex_fitness'],
            check_surface_intersection=False,intersection_checker=None)
        final_seconds=time.perf_counter()-tick
        state=result['surface_state'];values,control=candidate_payload(result['final_evaluation']['candidate'])
        for name in ('original_vertices','vertex_normals','faces','vertex_topology','vertex_rip','vertex_marks',
                     'face_area','vertex_area','flattened_vertices','curvbak','k1','k2','tangent_e1','tangent_e2'):
            values['final_'+name]=np.asarray(state[name])
        for name in ('neighbor_offsets','ordered_neighbors','incident_offsets','incident_faces','incident_corners',
                     'selected_vertices','selected_faces','face_normals'):
            values['final_'+name]=np.asarray(result['final_geometry'][name])
        for name in ('edge_used','discarded_defect_vertices','use_counts','vertex_fitness'):
            values['final_'+name]=np.asarray(result[name])
        values['actual_final_fitness']=np.asarray(result['fitness'],np.float64)
        values['newly_discarded_entries']=result['cleanup']['newly_discarded_entries']
        control=dict(score_candidate=control,additional_discarded_count=result['cleanup']['additional_discarded_count'],
            fitness_matches_own_initial=result['fitness_matches_best'],source_intersections=result['source_intersections'])
        tick=time.perf_counter()
        np.savez_compressed(output/(label+'.final.private.npz'),**values)
        dump_state_checkpoint(output/(label+'.state.private.npz'),state)
        (output/(label+'.control.private.json')).write_text(json.dumps(json_safe(control),allow_nan=False)+'\n')
        writing=time.perf_counter()-tick
        payloads.append(values);controls.append(control)
        records.append(dict(label=label,full_state_preparation_seconds=preparation_seconds,
            initial_candidate_cycle_seconds=initial_seconds,finalization_api_seconds=final_seconds,
            finalization_subseconds=result['seconds'],write_seconds=writing,
            actual_own_initial_fitness=initial['fitness'],actual_final_fitness=result['fitness'],
            fitness_matches_own_initial=result['fitness_matches_best'],final_vertices=len(state['original_vertices']),
            final_faces=state['nfaces'],additional_discarded_count=result['cleanup']['additional_discarded_count'],
            RP_max_count=int(result['use_counts'].max(initial=0)),
            final_geometry_against_own_initial=dict(
                ordered_faces=difference(state['faces'],first['geometry']['faces']),
                original_vertices=difference(state['original_vertices'],first['geometry']['original_vertices']),
                vertex_normals=difference(state['vertex_normals'],first['vertex_normals'])),
            final_mesh_quality='not_assessed',full_source_final_reference='not_assessed'))
        return first
    benchmark.evaluate_candidate=observed
    try:source_gate=benchmark.run(**kwargs)
    finally:benchmark.evaluate_candidate=original
    repeat={key:difference(payloads[0][key],payloads[1][key])for key in payloads[0]}
    passed=controls[0]==controls[1]and all(record['byte_equal']for record in repeat.values())
    import fnit.recon_all.topology_ga_finalize as final_module
    import fnit.recon_all.topology_candidate_cycle as cycle_module
    import benchmark_recon_topology_candidate_cycle as cycle_tool
    hashes={name:hashlib.sha256(Path(path).read_bytes()).hexdigest()for name,path in (
        ('topology_ga_finalize',final_module.__file__),('topology_candidate_cycle',cycle_module.__file__),
        ('cycle_diagnostic_helper',cycle_tool.__file__),('finalization_diagnostic',__file__))}
    report=dict(status='bounded_real_finalization_repeat_passed'if passed else 'bounded_real_finalization_repeat_failed',
        declared_case='public ds000114 sub-07 ses-test LH; FNIT 79a41cdd upstream',
        scope='same declared frozen attributes/global priors; own initial score then final materialization, no full GA search',
        source_sha256=hashes,initial_inputs_sha256=source_gate['initial_inputs_sha256'],
        actual_loaded_fnit_source_sha256={name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name,module in sorted(sys.modules.items())
            if name.startswith('fnit.') and getattr(module,'__file__',None)
            and str(module.__file__).endswith('.py')},
        comparison_reference_sha256=source_gate['comparison_reference_sha256'],
        initial_candidate_strict_source_gate=source_gate['status'],
        raw_calibration_seconds=source_gate['raw_calibration_seconds'],records=records,
        numeric_repeat=repeat,control_repeat_equal=controls[0]==controls[1],
        frozen_initial_attributes=source_gate['frozen_initial_attribute_names'],
        frozen_global_priors=source_gate['frozen_global_priors'],
        reference_after_used_to_generate=False,full_source_final_reference='not_assessed',
        own_best_policy='sole explicitly declared ordering; no population search or winner claim',
        source_intersection_optional_check='disabled as declared fixed-ga default; full QC not assessed',
        complete_base_producer=False,complete_ga=False,final_mesh_quality='not_assessed',
        overall_metric_equivalence='not_assessed',whole_recon_all='not_assessed',
        target_cuda='hidden',process_gpu_memory_bytes=None,threads=kwargs['threads'],
        full_diagnostic_before_final_report_seconds=time.perf_counter()-start)
    (output/'finalization.report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    if not passed:raise RuntimeError('final materialization repeat failed; report preserved')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before-retess','initial-attributes','initial-global-priors','input-surface','input-qsphere','brain',
                 'after-statistics-reference','after-volume-reference','fitness-reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,required=True)
    print(json.dumps(json_safe(run(**vars(p.parse_args()))),indent=2,allow_nan=False))


if __name__=='__main__':main()
