"""仅发布完整独立候选的已知数值、枚举和SHA，保留失败及完整trace。"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


_HELPER=Path(__file__).with_name('export_qc.py')
_SPEC=importlib.util.spec_from_file_location('candidate_qc_schema',_HELPER)
_MODULE=importlib.util.module_from_spec(_SPEC);_SPEC.loader.exec_module(_MODULE)
exact,digest,number,summary=_MODULE.exact,_MODULE.digest,_MODULE.number,_MODULE.summary


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(value):
    if type(value) is not bool:raise ValueError('typed boolean required')
    return value


def finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('typed finite number required')
    return value


def trace(row):
    required={'sweep','pair','moving_face','ring','plane','moving_vertices','affected_faces',
              'optimizer_success','optimizer_status','optimizer_iterations','minimum_optimizer_constraint',
              'optimizer_seconds','unit_plane_constraint_error','decision'}
    optional={'source_directed_pairs','float64_pairs','white_pial_crossing_qc','new_proper_crossings','new_nonmedial_exact_contacts'}
    if not isinstance(row,dict) or not required<=set(row) or not set(row)<=required|optional:
        raise ValueError('unknown candidate trace schema')
    reasons={'eligible','optimizer_infeasible','total_displacement_limit','face_orientation','face_area',
             'new_positive_pair','not_strictly_reduced_in_both_detectors','new_white_pial_proper_crossing',
             'new_white_pial_nonmedial_exact_contact'}
    if row['decision'] not in reasons or row['plane'] not in {'normal_positive','normal_negative','edge_outside_0','edge_outside_1','edge_outside_2'}:
        raise ValueError('unknown trace enumeration')
    if type(row['pair']) is not list or len(row['pair'])!=2:raise ValueError('ordered pair required')
    result={k:number(row[k],True) for k in ('sweep','moving_face','ring','moving_vertices','affected_faces','optimizer_status','optimizer_iterations')}
    result.update(pair=[number(v,True) for v in row['pair']],plane=row['plane'],decision=row['decision'],
                  optimizer_success=boolean(row['optimizer_success']),minimum_optimizer_constraint=finite(row['minimum_optimizer_constraint']),
                  optimizer_seconds=number(row['optimizer_seconds']),unit_plane_constraint_error=number(row['unit_plane_constraint_error']))
    if result['sweep']>=4 or result['ring']>2:raise ValueError('predeclared sweep/ring exceeded')
    self_group={'source_directed_pairs','float64_pairs'};cross_group={'white_pial_crossing_qc','new_proper_crossings','new_nonmedial_exact_contacts'}
    for group in (self_group,cross_group):
        if set(row)&group and not group<=set(row):raise ValueError('incomplete candidate QC group')
    if row['decision']=='eligible' and not self_group|cross_group<=set(row):raise ValueError('accepted trial lacks crossing/self QC')
    for key in self_group|{'new_proper_crossings','new_nonmedial_exact_contacts'}:
        if key in row:result[key]=number(row[key],True)
    if 'white_pial_crossing_qc' in row:result['white_pial_crossing_qc']=summary(row['white_pial_crossing_qc'])
    if row['decision']=='eligible' and (row['new_proper_crossings']!=0 or row['new_nonmedial_exact_contacts']!=0):
        raise ValueError('accepted trial introduced crossing/contact')
    return result


def export(*, report_path: Path, timer_path: Path, complete_path: Path,
           diagnostic_case: Path, input_hashes: Path, output: Path):
    """只读本候选/冷计时/完成收据，校验全部输入/产物SHA后导出。

    六个具名参数均为Path，无默认值；output须不存在。输入仅独立
    自产候选任务，不得传整例生产报告。输出含完整typed trace、所有
    接受/拒绝、双碰撞/穿越、位移/面积/距离、拓扑和时间/RSS，原始
    失败完整保留；不复制argv、私人路径、坐标、参考或许可证。未知
    字段、漂移/不完整输入、布尔伪整数或状态矛盾先失败，不写文件。
    返回dict，无MRI/优化器/GPU/原程序调用；不声明整体指标等效。
    """
    if output.exists():raise FileExistsError(output)
    raw=report_path.read_bytes();d=json.loads(raw)
    keys={'schema_version','execution_status','scope','input_sha256','terminal_sha256','protocol_sha256','protocol','script_sha256',
          'threads','cpu_affinity','cuda_visible_devices','gpu_memory_bytes','gpu_computation','whole_pipeline_speedup',
          'strict_reproduction','overall_metric_equivalence','versions','coordinate_space','coordinate_unit','original_stage_program_called',
          'source_sha256','mesh','qc','fixed_point_diagnosis','topology','nonmanifold_vertex_links','edge_orientation_conflicts','resolver',
          'output_sha256','geometry_guards','ordered_displacement_mm','bidirectional_vertex_to_triangle','surface_distance_seconds',
          'surface_distance_module_sha256','residual_self_intersection_quality','quality_scope','peak_process_cpu_memory_bytes',
          'complete_wall_seconds_before_report'}
    exact(d,keys)
    if type(d['schema_version']) is not int or d['schema_version']!=1 or d['execution_status']!='complete_diagnosis':raise ValueError('complete own diagnosis required')
    if d['scope']!='frozen same-input local residual cleanup; not raw-T1 workflow' or d['strict_reproduction']!='independent_algorithm_not_source_reproduction':
        raise ValueError('unknown candidate scope')
    if d['overall_metric_equivalence']!='not_assessed' or d['coordinate_space']!='surface_RAS' or d['coordinate_unit']!='mm':raise ValueError('unexpected equivalence/space')
    if d['gpu_computation'] is not False or d['original_stage_program_called'] is not False or d['gpu_memory_bytes'] is not None or d['whole_pipeline_speedup'] is not None:
        raise ValueError('unexpected runtime route/resource claims')
    if type(d['threads']) is not int or d['threads']!=4 or d['cpu_affinity']!=[112,113,114,115] or d['cuda_visible_devices']!='':raise ValueError('fixed four-core CPU budget required')
    required={'surf/lh.white','surf/autodet.gw.stats.lh.dat','label/lh.cortex.label','label/lh.cortex+hipamyg.label',
              'mri/brain.finalsurfs.mgz','mri/wm.mgz','mri/aseg.presurf.mgz'}
    expected=json.loads(input_hashes.read_text());exact(expected,required);exact(d['input_sha256'],required)
    if d['input_sha256']!=expected:raise ValueError('input map changed')
    for name,value in expected.items():
        if sha(diagnostic_case/'subject'/name)!=digest(value):raise ValueError('actual source input changed')
    if sha(diagnostic_case/'cleanup-result.private.npy')!=digest(d['terminal_sha256']):raise ValueError('own terminal changed')
    exact(d['output_sha256'],{'candidate.private.npy','candidate.pial'})
    for name,value in d['output_sha256'].items():
        if sha(report_path.parent/name)!=digest(value):raise ValueError('written candidate changed')
    complete_raw=complete_path.read_bytes();c=json.loads(complete_raw)
    exact(c,{'execution_status','quality','report_sha256','output_sha256','complete_wall_seconds'})
    if c['execution_status']!='complete_diagnosis' or digest(c['report_sha256'])!=hashlib.sha256(raw).hexdigest() or c['output_sha256']!=d['output_sha256']:
        raise ValueError('complete receipt does not bind report/outputs')
    params={'max_displacement_mm':1.,'minimum_area_ratio':.2,'clearance_mm':.002,'max_sweeps':4,'optimizer_iterations':100,
            'diagnostic_pair_limit':None,'projection_mode':'free_separating_plane','crossing_guard':True}
    if d['protocol'].get('algorithm')!='independent_free_plane_pair_solver_v2_crossing_guard' or d['protocol'].get('parameters')!=params:
        raise ValueError('predeclared complete candidate protocol required')
    for key,value in params.items():
        if type(d['protocol']['parameters'][key]) is not type(value):raise ValueError('typed protocol parameter required')
    q=d['resolver']
    rkeys={'schema_version','algorithm','parameters','source_directed_pairs_before','float64_pairs_before','fixed_vertices',
           'candidate_trace','accepted','limited_initial_pairs','strict_reproduction','overall_metric_equivalence','white_pial_guard',
           'status','scope','source_directed_pairs_after','float64_pairs_after','source_intersecting_faces_after','float64_intersecting_faces_after',
           'fixed_different_elements','changed_vertices','mean_displacement_mm','p99_displacement_mm','max_displacement_mm','global_checks','full_resolver_seconds'}
    exact(q,rkeys)
    rparams=params|{'rings':[0,1,2],'optimizer_ftol':1e-9,'optimizer_feasibility_tolerance':1e-9,
                   'free_plane_direction_regularization':1e-6,'free_plane_unit_norm_tolerance':1e-9}
    if q['parameters']!=rparams or q['algorithm']!='independent_free_plane_pair_solver_v2_crossing_guard' or q['scope']!='complete_residual_stage' or q['limited_initial_pairs'] is not None:
        raise ValueError('full resolver protocol drift')
    if type(q['schema_version']) is not int or q['schema_version']!=1 or q['strict_reproduction']!='independent_algorithm_not_source_reproduction' or q['overall_metric_equivalence']!='not_assessed':
        raise ValueError('unknown resolver schema/claims')
    for key,value in rparams.items():
        if type(q['parameters'][key]) is not type(value):raise ValueError('typed resolver parameter required')
    ints={'source_directed_pairs_before','float64_pairs_before','fixed_vertices','source_directed_pairs_after','float64_pairs_after',
          'source_intersecting_faces_after','float64_intersecting_faces_after','fixed_different_elements','changed_vertices'}
    resolver={k:number(q[k],True) for k in ints}|{k:number(q[k]) for k in ('mean_displacement_mm','p99_displacement_mm','max_displacement_mm','full_resolver_seconds')}
    checks={'full_source_pair_set_matches_local_update','full_float64_pair_set_matches_local_update','no_new_source_directed_pairs','no_new_float64_pairs'}
    if type(q['global_checks']) is not list or len(q['global_checks'])!=1:raise ValueError('complete global checks required')
    exact(q['global_checks'][0],checks)
    if any(v is not True for v in q['global_checks'][0].values()) or q['fixed_different_elements']!=0:raise ValueError('failed source/global/fixed guard')
    traces=[trace(v) for v in q['candidate_trace']];accepted=[trace(v) for v in q['accepted']]
    if any(v['decision']!='eligible' for v in accepted):raise ValueError('accepted decision missing/incorrect')
    trace_text={json.dumps(v,sort_keys=True) for v in traces}
    if any(json.dumps(v,sort_keys=True) not in trace_text for v in accepted):raise ValueError('accepted row not in original trace')
    w=q['white_pial_guard'];wkeys={'enabled','initial','initial_nonmedial_exact_contacts','rule','final','checks','final_nonmedial_exact_contacts',
                                   'no_new_gate','final_proper_crossings_zero','final_nonmedial_exact_contacts_zero'}
    exact(w,wkeys);cross_checks={'full_proper_pair_set_matches_local_update','full_nonmedial_contact_set_matches_local_update','no_new_proper_crossings','no_new_nonmedial_exact_contacts'}
    exact(w['checks'],cross_checks)
    if w['enabled'] is not True or w['no_new_gate']!='passed' or any(v is not True for v in w['checks'].values()):raise ValueError('crossing/contact guard failed')
    cross={'initial':summary(w['initial']),'final':summary(w['final']),
           'initial_nonmedial_exact_contacts':number(w['initial_nonmedial_exact_contacts'],True),
           'final_nonmedial_exact_contacts':number(w['final_nonmedial_exact_contacts'],True),
           'checks':{k:True for k in cross_checks},'no_new_gate':'passed',
           'final_proper_crossings_zero':boolean(w['final_proper_crossings_zero']),
           'final_nonmedial_exact_contacts_zero':boolean(w['final_nonmedial_exact_contacts_zero'])}
    if cross['final_proper_crossings_zero']!=(cross['final']['proper_crossing_pairs']==0) or cross['final_nonmedial_exact_contacts_zero']!=(cross['final_nonmedial_exact_contacts']==0):
        raise ValueError('zero flags contradict actual initial/final counts')
    resolver.update(algorithm=q['algorithm'],parameters=rparams,candidate_trace=traces,accepted=accepted,
                    decision_counts=dict(sorted(Counter(v['decision'] for v in traces).items())),
                    global_checks=q['global_checks'],white_pial_guard=cross,status=q['status'],scope=q['scope'])
    qc=[]
    if type(d['qc']) is not list or len(d['qc'])!=2:raise ValueError('before/after full QC required')
    qkeys={'state','source_faces','source_vertices','float64_faces','float64_vertices','source_seconds','float64_and_area_seconds',
           'finite','zero_area_faces','minimum_area_mm2','marked_vertex_sets_equal'}
    for row,state in zip(d['qc'],('initial_terminal','candidate')):
        exact(row,qkeys)
        if row['state']!=state:raise ValueError('unknown QC state')
        qc.append({'state':state,**{k:number(row[k],True) for k in ('source_faces','source_vertices','float64_faces','float64_vertices','zero_area_faces')},
                   **{k:number(row[k]) for k in ('source_seconds','float64_and_area_seconds','minimum_area_mm2')},
                   **{k:boolean(row[k]) for k in ('finite','marked_vertex_sets_equal')}})
    quality='passed' if qc[-1]['source_faces']==qc[-1]['float64_faces']==0 else 'failed'
    if any(not row['finite'] or row['zero_area_faces'] for row in qc):raise ValueError('nonfinite or degenerate candidate mesh')
    if d['residual_self_intersection_quality']!=quality or c['quality']!=quality or q['status']!=('passed_zero_intersections' if quality=='passed' else 'failed_residual'):
        raise ValueError('residual failure hidden or status inconsistent')
    exact(d['mesh'],{'vertices','faces','fixed_vertices','ripped_vertices','outside_cortex_elements_different_from_white_at_entry','ripped_elements_different_from_white_at_entry'})
    mesh={k:number(v,True) for k,v in d['mesh'].items()}
    exact(d['topology'],{'vertices','faces','edges','euler','components','boundary_edges','nonmanifold_edges','degenerate_index_faces','duplicate_faces','valid'})
    topology={k:(boolean(v) if k=='valid' else finite(v) if k=='euler' else number(v,True)) for k,v in d['topology'].items()}
    exact(d['fixed_point_diagnosis'],{'marked_nonrip_vertices','vertices_with_nonzero_source_soap_update','source_soap_skips_ripped_neighbors'})
    fixed={k:(boolean(v) if k=='source_soap_skips_ripped_neighbors' else number(v,True)) for k,v in d['fixed_point_diagnosis'].items()}
    g=d['geometry_guards'];exact(g,{'fixed_different_elements','ordered_faces_equal','nonpositive_initial_normal_dot_faces','minimum_area_ratio','white_pial_crossing'})
    if g['white_pial_crossing']!=w:raise ValueError('duplicated guard differs')
    geometry={'fixed_different_elements':number(g['fixed_different_elements'],True),'ordered_faces_equal':boolean(g['ordered_faces_equal']),
              'nonpositive_initial_normal_dot_faces':number(g['nonpositive_initial_normal_dot_faces'],True),'minimum_area_ratio':number(g['minimum_area_ratio'])}
    exact(d['ordered_displacement_mm'],{'different_elements','changed_vertices','mean','p99','max','changed_vertex_p99'})
    displacement={k:number(v,k in ('different_elements','changed_vertices')) for k,v in d['ordered_displacement_mm'].items()}
    exact(d['bidirectional_vertex_to_triangle'],{'candidate_to_entry','entry_to_candidate'})
    distances={}
    for direction,values in d['bidirectional_vertex_to_triangle'].items():
        exact(values,{'mean_mm','p99_mm','max_mm'});distances[direction]={k:number(v) for k,v in values.items()}
    sources={}
    for name,value in d['source_sha256'].items():
        if not re.fullmatch(r'fnit\.recon_all\.[A-Za-z_0-9]+',name):raise ValueError('private/unknown source module')
        sources[name]=digest(value)
    if sources.get('fnit.recon_all.white_pial_crossing_python')!=d['protocol'].get('crossing_helper_sha256'):raise ValueError('actual lazy helper source differs')
    exact(d['versions'],{'python','numpy','numba','nibabel','scipy'})
    if any(not isinstance(v,str) or not re.fullmatch('[0-9]+[.][0-9]+[.][0-9]+(?:[a-zA-Z0-9.+-]*)',v) for v in d['versions'].values()):raise ValueError('unknown version text')
    timer_raw=timer_path.read_bytes();t=json.loads(timer_raw)
    exact(t,{'schema_version','scope','command_sha256','timer_script_sha256','gpu_memory_bytes',
       'simultaneous_parent_child_peak_cpu_memory_bytes','execution_status','child_exit_code','cold_cli_wall_seconds','max_child_cpu_rss_bytes'})
    if type(t['schema_version']) is not int or t['schema_version']!=1 or t['scope']!='single frozen Python benchmark cold CLI; not end-to-end recon-all' or t['execution_status']!='complete' or type(t['child_exit_code']) is not int or t['child_exit_code']!=0:
        raise ValueError('complete cold CLI required')
    if t['gpu_memory_bytes'] is not None or t['simultaneous_parent_child_peak_cpu_memory_bytes'] is not None:raise ValueError('unmeasured resources must stay null')
    result={'schema_version':1,'execution_status':'complete_diagnosis','scope':'complete frozen-own-input residual candidate; not raw T1',
            'raw_report_sha256':hashlib.sha256(raw).hexdigest(),'raw_complete_sha256':hashlib.sha256(complete_raw).hexdigest(),
            'input_sha256':expected,'input_hashes_unchanged_after_candidate':True,'terminal_sha256':digest(d['terminal_sha256']),
            'script_sha256':digest(d['script_sha256']),'protocol_sha256':digest(d['protocol_sha256']),
            'source_sha256':sources,'mesh':mesh,'qc':qc,'topology':topology,'fixed_point_diagnosis':fixed,
            'nonmanifold_vertex_links':number(d['nonmanifold_vertex_links'],True),'edge_orientation_conflicts':number(d['edge_orientation_conflicts'],True),
            'resolver':resolver,'output_sha256':d['output_sha256'],'geometry_guards':geometry,'ordered_displacement_mm':displacement,
            'bidirectional_vertex_to_triangle':distances,'surface_distance_seconds':number(d['surface_distance_seconds']),
            'surface_distance_module_sha256':digest(d['surface_distance_module_sha256']),
            'residual_self_intersection_quality':quality,'strict_reproduction':'independent_algorithm_not_source_reproduction',
            'overall_metric_equivalence':'not_assessed','gpu_computation':False,'gpu_memory_bytes':None,'whole_pipeline_speedup':None,
            'original_stage_program_called':False,'threads':4,'cpu_affinity':[112,113,114,115],'versions':d['versions'],
            'peak_process_cpu_memory_bytes':number(d['peak_process_cpu_memory_bytes'],True),
            'complete_wall_seconds_before_report':number(d['complete_wall_seconds_before_report']),
            'complete_receipt_wall_seconds':number(c['complete_wall_seconds']),
            'cold_cli':{'raw_timer_sha256':hashlib.sha256(timer_raw).hexdigest(),'child_exit_code':0,
                'cold_cli_wall_seconds':number(t['cold_cli_wall_seconds']),'max_child_cpu_rss_bytes':number(t['max_child_cpu_rss_bytes'],True),
                'command_sha256':digest(t['command_sha256']),'timer_script_sha256':digest(t['timer_script_sha256']),
                'gpu_memory_bytes':None,'simultaneous_parent_child_peak_cpu_memory_bytes':None},
            'export_qc_schema_source_sha256':sha(_HELPER)}
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('report-path','timer-path','complete-path','diagnostic-case','input-hashes','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    r=export(**vars(parser.parse_args()));print(json.dumps({'execution_status':r['execution_status'],'quality':r['residual_self_intersection_quality'],'raw_report_sha256':r['raw_report_sha256']}))
