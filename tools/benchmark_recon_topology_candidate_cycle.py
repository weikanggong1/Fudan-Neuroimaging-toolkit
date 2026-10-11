"""声明冻结初态的两次连续候选循环；参考仅由原工具事后比较。"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import benchmark_recon_topology_candidate_pipeline as benchmark
from benchmark_recon_topology_prepared_cache import candidate_payload
from benchmark_recon_topology_patch_fitness import difference,json_safe
from benchmark_recon_topology_candidate_state import dump_state_checkpoint
from benchmark_recon_topology_retess_fields import load_retessellation_capture
from fnit.recon_all.topology_candidate_cycle import CandidateCycleEvaluator
from fnit.recon_all.topology_candidate_state import record_candidate_state
from fnit.recon_all.topology_candidate_geometry import candidate_incidents
from fnit.recon_all.topology_retessellation_dynamic import prepare_retessellation_base


def build_frozen_cycle(*,surface_state,retessellation_inputs,vertex_marks,candidate_inputs):
    """诊断初态→串行回调；不加载参考 after 或创建 global priors。

    surface_state 是原工具明确声明的冻结 before 属性；r 为同网格的
    完整成熟 retessellation 关键字；vertex_marks 为 before int32(N,)。
    一环元数据按源 base calloc/加边规则构造，不能称 raw base producer。
    candidate_inputs 明确传自产 MRI/局部 priors 与冻结 global priors。
    返回 CandidateCycleEvaluator；其 snapshot/cache 可供最终重建工具
    复用。所有数组输入不改，缺域/mode 不符报错，无隐式参考/设备切换。
    """
    r=dict(retessellation_inputs);state=dict(surface_state);n=len(state['original_vertices'])
    if r.get('mode')!='all':raise ValueError('only declared all-mode initial-state cycle')
    marks=np.asarray(vertex_marks)
    if marks.dtype!=np.int32 or marks.shape!=(n,):raise ValueError('explicit before int32(N,) marks required')
    degree=np.asarray([len(row) for row in r['neighbors']],np.int32)
    top=np.zeros((n,7),np.int32);top[degree>0,:2]=1;top[:,2]=top[:,3]=degree
    incident=candidate_incidents(faces=state['faces'],vertex_count=n)
    top[:,6]=np.diff(incident['incident_offsets'])
    state.update(nfaces=len(state['faces']),vertex_topology=top,
        vertex_rip=np.asarray(r['vertex_rip'],np.int32).copy(),vertex_marks=marks.copy(),
        neighbor_storage=[np.asarray(q,np.int32).copy()for q in r['neighbors']],
        incident_faces=[incident['incident_faces'][incident['incident_offsets'][v]:incident['incident_offsets'][v+1]].copy()for v in range(n)],
        incident_corners=[incident['incident_corners'][incident['incident_offsets'][v]:incident['incident_offsets'][v+1]].copy()for v in range(n)])
    snapshot=record_candidate_state(surface_state=state,defect_vertices=r['inside_vertices'],convex_hull_vertices=r['hull_vertices'],
        border_vertices=r['border_vertices'],vertex_translation=np.arange(n,dtype=np.int32))
    prepared=prepare_retessellation_base(canonical=r['canonical'],base_faces=r['base_faces'],neighbors=r['neighbors'])
    r.update(canonical=prepared.canonical,base_faces=prepared.base_faces,neighbors=prepared.neighbors,base_context=prepared)
    callback=CandidateCycleEvaluator(surface_state=state,snapshot=snapshot,retessellation_inputs=r,candidate_inputs=candidate_inputs)
    callback.diagnostic_preparation_seconds=prepared.preparation_seconds
    return callback


def run(**kwargs):
    """复用唯一 raw-MRI/校准准备，运行两次自产连续序列 A0/A1 与 B0/B1。

    参数与原 bounded pipeline 复现工具相同；不读 reference after 到
    任一候选初态。冻结 before 原属性/global prior 边界保持；一环
    元数据按源 base 的零分配/加边规则构造而非 after 捕获，多环状态
    不支持。初始 RP 明确按源首次 calloc 为零，不按参考 after 填。
    两个独立序列均将 K/E/fx/curvbak/RP 从第一个候选传入第二个；
    同一 ordering，不代表种群、mutation 或完整随机序列。原参考只
    覆盖每条序列的首候选，第二候选仅自产全字段/控制重复性。
    输出新目录中的完整私有 state/candidate NPZ 与公开白名单指标。
    完整 CLI 含 raw 准备、四次 API、状态/候选读写与参考；GPU 隐藏。
    """
    start=time.perf_counter();output=Path(kwargs['output']);calls=0;records=[];payloads=[];controls=[]
    before=load_retessellation_capture(capture_path=kwargs['before_retess'])
    original=benchmark.evaluate_candidate
    def observed(**kw):
        nonlocal calls
        calls+=1;label='A' if calls==1 else 'B'
        if calls>2:raise RuntimeError('expected two frozen diagnostic sequence slots')
        r=dict(kw['retessellation_inputs'])
        inputs={key:value for key,value in kw.items()if key not in ('surface_state','retessellation_inputs')}
        callback=build_frozen_cycle(surface_state=kw['surface_state'],retessellation_inputs=r,
            vertex_marks=before['vertex_flags'][:,0].copy(),candidate_inputs=inputs)
        count=np.zeros(len(r['inside_vertices']),np.int32);fitness=np.zeros(len(count),np.float32)
        first=None
        for iteration in range(2):
            tick=time.perf_counter()
            feedback=callback(ordering=r['ordering'],discarded_defect_vertices=r['inside_discarded'],vertex_rip=r['vertex_rip'],
                use_counts=count,vertex_fitness=fitness)
            count,fitness=feedback['use_counts'],feedback['vertex_fitness'];result=callback.last_result
            seconds=time.perf_counter()-tick;values,control=candidate_payload(result['candidate'])
            values.update(reset_edge_used=result['edge_used'],use_counts=count,vertex_fitness=fitness,
                restored_original=callback.surface_state['original_vertices'],restored_normals=callback.surface_state['vertex_normals'],
                restored_topology=callback.surface_state['vertex_topology'],restored_curvbak=callback.surface_state['curvbak'],
                restored_k1=callback.surface_state['k1'],restored_k2=callback.surface_state['k2'],
                restored_tangent_e1=callback.surface_state['tangent_e1'],restored_tangent_e2=callback.surface_state['tangent_e2'],
                restored_flattened=callback.surface_state['flattened_vertices'],restored_rip=callback.surface_state['vertex_rip'],
                restored_marks=callback.surface_state['vertex_marks'],
                logical_faces=np.asarray(callback.surface_state['nfaces'],np.int32))
            tick=time.perf_counter();np.savez_compressed(output/(label+str(iteration)+'.private.npz'),**values)
            dump_state_checkpoint(output/(label+str(iteration)+'.state.private.npz'),callback.surface_state)
            writing=time.perf_counter()-tick
            payloads.append(values);controls.append(control)
            records.append(dict(sequence=label,iteration=iteration,complete_cycle_seconds=seconds,
                write_seconds=writing,cycle_subseconds=result['seconds'],candidate_subseconds=result['candidate']['seconds'],
                cache_preparation_seconds=callback.diagnostic_preparation_seconds if iteration==0 else None,
                RP_updated_entries=len(result['statistics']['updated_defect_entries']),RP_counts_min=int(count.min(initial=0)),
                RP_counts_max=int(count.max(initial=0)),full_restore_records=result['restore']['restored_records'],
                source_reset_new_to_not_and_both_to_original=True,logical_faces=callback.surface_state['nfaces']))
            if iteration==0:first=result['candidate']
        return first
    # This hook is confined to a diagnostic driver; production imports/callers
    # retain their explicit API. The original gate consumes references last.
    benchmark.evaluate_candidate=observed
    try:source_gate=benchmark.run(**kwargs)
    finally:benchmark.evaluate_candidate=original
    checks=[]
    for i in range(2):
        checks.append(dict(iteration=i,numeric={key:difference(payloads[i][key],payloads[i+2][key])for key in payloads[i]},
                           controls_equal=controls[i]==controls[i+2]))
    passed=all(c['controls_equal']and all(x['byte_equal']for x in c['numeric'].values())for c in checks)
    root=Path(__file__).resolve().parents[1]
    report=dict(status='bounded_two_candidate_cycle_repeat_passed'if passed else 'bounded_two_candidate_cycle_repeat_failed',
        scope='same frozen initial attributes/global prior; independently prepared MRI/local prior; two sequential same-order candidates',
        declared_case='public ds000114 sub-07 ses-test LH; 79a41cdd producer',records=records,repeat_checks=checks,
        first_candidate_existing_reference_gate=source_gate['status'],second_candidate_source_reference='not_assessed',
        initial_DVS_metadata='source one-ring zero-allocation base rules; no independent complete before-DVS capture',
        frozen_initial_attributes=source_gate['frozen_initial_attribute_names'],frozen_global_priors=source_gate['frozen_global_priors'],
        initial_RP='source first-candidate zero-initialized counts and fitness',
        reference_after_used_as_input=False,complete_base_producer=False,complete_ga=False,
        final_mesh_quality='not_assessed',overall_metric_equivalence='not_assessed',
        threads=kwargs['threads'],target_cuda='hidden',gpu_memory=None,
        initial_inputs_sha256=source_gate['initial_inputs_sha256'],comparison_reference_sha256=source_gate['comparison_reference_sha256'],
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()for name in (
            'src/fnit/recon_all/topology_candidate_cycle.py','src/fnit/recon_all/topology_candidate_pipeline.py',
            'src/fnit/recon_all/topology_candidate_state.py','src/fnit/recon_all/topology_candidate_statistics.py',
            'tools/benchmark_recon_topology_candidate_cycle.py')},
        full_diagnostic_before_report_seconds=time.perf_counter()-start)
    (output/'cycle.report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    if not passed:raise RuntimeError('two candidate cycles changed repeat; all outputs retained')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before-retess','initial-attributes','initial-global-priors','input-surface','input-qsphere','brain',
                 'after-statistics-reference','after-volume-reference','fitness-reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,required=True)
    print(json.dumps(json_safe(run(**vars(p.parse_args()))),indent=2,allow_nan=False))


if __name__=='__main__':main()
