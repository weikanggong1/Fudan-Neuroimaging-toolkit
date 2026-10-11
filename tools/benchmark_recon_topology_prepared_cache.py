"""同一声明初态的完整候选ABBA；只改变显式不可变基底缓存。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

import benchmark_recon_topology_candidate_pipeline as benchmark
from benchmark_recon_topology_patch_fitness import difference,json_safe
from fnit.recon_all.topology_retessellation_dynamic import prepare_retessellation_base
from fnit.recon_all.topology_preflight_python import _edge_intersects
from fnit.recon_all.topology_retessellation_python import _contains_neighbor_on_sphere


def candidate_payload(result):
    """完整候选数值/控制输出→诊断NPZ字段与决策；不读参考后态。"""
    geometry=result['geometry'];fit=result['fitness'];sff=result['sff']
    names=('faces','patch_faces','selected_faces','selected_vertices','selected_edges','edge_used',
        'marks','removed_vertices','temporary_faces','original_vertices','smoothed_original_vertices',
        'face_normals','selected_vertex_normals','cached_original_face_area_mm2','incident_offsets',
        'incident_faces','incident_corners','neighbor_offsets','ordered_neighbors')
    values={name:np.asarray(geometry[name]) for name in names}
    values.update({name:np.asarray(sff[name]) for name in ('k1','k2','tangent_e1','tangent_e2')})
    values.update(vertex_normals=result['vertex_normals'],vertex_curvbak=result['vertex_curvbak'],
        flattened_vertices=result['flattened_vertices'],face_area=result['face_area'],
        displacement_vertices=result['displacement']['vertex_indices'],displacement_mm=result['displacement']['displacement_mm'],
        distance_mm=fit['distance']['distance_mm'],fitness=np.asarray(fit['fitness'],np.float64),
        core_fields=np.asarray([fit['composition']['tp_fields'][name] for name in
            ('face_ll','vertex_ll','curv_ll','qcurv_ll','unmri_ll')],np.float32))
    control=json_safe(dict(retessellation_trace=geometry['retessellation_trace'],face_decisions=geometry['face_decisions'],
        rounds=geometry['rounds'],ninside=geometry['ninside'],ndiscarded=geometry['ndiscarded'],
        face_check=result['face_check']))
    return values,control


def run(**kwargs):
    """复用现有真实输入准备和事后参考检查，完整warm候选A/B/B/A。

    参数与benchmark_recon_topology_candidate_pipeline.run完全相同，output
    必须为空新目录。先一次prepare并运行ordinary cold与first prepared，
    单列两种数组布局的首次编译signature，再同线程预算依次运行
    ordinary/prepared/prepared/ordinary完整warm API。每次完整NPZ写出
    和控制决策比较计时另列；完整CLI包含原始nofix/MRI准备、缓存
    校验/复制、六次API、读写及事后参考检查。
    缓存的三个只读对象显式同时传入，不缓存坐标迭代/动态状态。
    参考后态只由原benchmark在所有candidate输出完成后打开，未生成
    基底/global priors的范围保持显式冻结，不称全GA/原始T1整例。
    """
    original=benchmark.evaluate_candidate;records=[];context=None;baseline=None;baseline_control=None
    output=Path(kwargs['output']);start=time.perf_counter();calls=0
    def one(label,enabled,call_kwargs):
        nonlocal baseline,baseline_control
        args=call_kwargs['retessellation_inputs']
        if enabled:
            args=dict(args,canonical=context.canonical,base_faces=context.base_faces,
                neighbors=context.neighbors,base_context=context)
        signatures=lambda:dict(edge_intersection=len(_edge_intersects.signatures),
            sphere_face_contains=len(_contains_neighbor_on_sphere.signatures))
        before=signatures()
        tick=time.perf_counter()
        result=original(**dict(call_kwargs,retessellation_inputs=args))
        actual_api=time.perf_counter()-tick;after=signatures();values,control=candidate_payload(result)
        tick=time.perf_counter();np.savez_compressed(output/(label+'.npz'),**values)
        write_seconds=time.perf_counter()-tick
        if baseline is None:baseline=values;baseline_control=control
        check={key:difference(value,baseline[key]) for key,value in values.items()}
        decisions_equal=control==baseline_control
        (output/(label+'.control.private.json')).write_text(json.dumps(control,allow_nan=False)+'\n')
        records.append(dict(label=label,prepared_base_enabled=enabled,complete_candidate_api_seconds=actual_api,
            component_seconds=result['seconds'],geometry_subseconds=result['geometry']['stage_seconds'],
            write_seconds=write_seconds,strict_numeric_checks=check,strict_decisions_equal=decisions_equal,
            prepared_copied_rows=result['geometry']['prepared_copied_rows'],
            compiled_signature_counts_before=before,compiled_signature_counts_after=after,
            new_signature_created=before!=after))
        return result
    def observed(**call_kwargs):
        nonlocal calls,context
        calls+=1
        if calls==1:
            args=call_kwargs['retessellation_inputs']
            context=prepare_retessellation_base(canonical=args['canonical'],base_faces=args['base_faces'],neighbors=args['neighbors'])
            return one('cold_ordinary',False,call_kwargs)
        if calls!=2:raise RuntimeError('expected exactly two existing diagnostic API slots')
        one('first_prepared',True,call_kwargs)
        last=None
        for label,enabled in [('A0_ordinary',False),('B0_prepared',True),('B1_prepared',True),('A1_ordinary',False)]:
            last=one(label,enabled,call_kwargs)
        return last
    benchmark.evaluate_candidate=observed
    try:reference_report=benchmark.run(**kwargs)
    finally:benchmark.evaluate_candidate=original
    if calls!=2 or len(records)!=6:raise RuntimeError('full ABBA incomplete')
    strict=all(record['strict_decisions_equal'] and all(item['byte_equal'] for item in record['strict_numeric_checks'].values()) for record in records)
    median_a=float(np.median([r['complete_candidate_api_seconds'] for r in records if r['label'].startswith('A')]))
    median_b=float(np.median([r['complete_candidate_api_seconds'] for r in records if r['label'].startswith('B')]))
    report=dict(status='strict_prepared_base_candidate_abba_passed' if strict else 'strict_prepared_base_candidate_abba_failed',
        scope='immutable base cache only; complete same prepared-state candidate warm ABBA; no full GA or original-T1 whole run',
        declared_case='public ds000114 sub-07 ses-test LH; FNIT 79a41cdd upstream',threads=kwargs['threads'],target_cuda='hidden',
        cache_preparation_seconds=context.preparation_seconds,immutable_base_state_sha256=context.state_sha256,
        cache_is_independent_immutable_snapshot=True,cache_identity_and_metadata_validation=True,
        records=records,ordinary_warm_api_median_seconds=median_a,prepared_warm_api_median_seconds=median_b,
        warm_complete_candidate_speedup=median_a/median_b,full_diagnostic_before_report_seconds=time.perf_counter()-start,
        existing_post_reference_gate=reference_report['status'],reference_after_used_as_input=False,
        initial_inputs_sha256=reference_report['initial_inputs_sha256'],comparison_reference_sha256=reference_report['comparison_reference_sha256'],
        complete_base_producer=False,complete_ga=False,final_mesh_quality='not_assessed',overall_metric_equivalence='not_assessed',
        source_sha256={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
            for name in ('src/fnit/recon_all/topology_retessellation_dynamic.py','src/fnit/recon_all/topology_candidate_geometry.py',
                         'src/fnit/recon_all/topology_candidate_pipeline.py','tools/benchmark_recon_topology_prepared_cache.py')})
    (output/'cache.report.private.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    if not strict:raise RuntimeError('immutable prepared base changed full candidate; report preserved')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('before-retess','initial-attributes','initial-global-priors','input-surface','input-qsphere','brain',
            'after-statistics-reference','after-volume-reference','fitness-reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,required=True)
    print(json.dumps(json_safe(run(**vars(parser.parse_args()))),indent=2,allow_nan=False))


if __name__=='__main__':main()
