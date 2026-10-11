"""仅导出已知QC数值/枚举/哈希；拒绝未知字段和私人路径。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def digest(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('SHA-256 required')
    return value


def number(value,integer=False):
    if type(value) not in ((int,) if integer else (int,float)) or value<0 or not math.isfinite(value):
        raise ValueError('nonnegative typed finite number required')
    return value


def exact(mapping,keys):
    if not isinstance(mapping,dict) or set(mapping)!=set(keys):raise ValueError('unexpected schema keys')


def summary(value):
    numbers={'query_faces','broad_phase_candidates','tested_pairs','proper_crossing_pairs','exact_contact_pairs',
             'source_predicate_positive_pairs','source_negative_proper_crossings','legal_fixed_contact_pairs',
             'short_positive_crossings_not_exempted'}
    floats={'maximum_intersection_length_mm','maximum_pial_plane_depth_mm','maximum_white_plane_depth_mm','query_seconds',
            'proper_crossing_minimum_length_mm','near_contact_display_length_mm'}
    fixed={'classes','coordinate_space','coordinate_unit','source_predicate_contacts_are_not_strict_crossings'}
    adaptive={'strict_relation_backend','exact_fallback_pairs','exact_shared_corner_plane_zero_proofs','adaptive_error_bound_epsilon_multiple'}
    keys=set(value) if isinstance(value,dict) else set()
    if keys!=numbers|floats|fixed and keys!=numbers|floats|fixed|adaptive:raise ValueError('unknown QC summary fields')
    result={k:number(value[k],True) for k in numbers}|{k:number(value[k]) for k in floats}
    exact(value['classes'],{'proper_crossing','noncoplanar_edge_or_point_contact','exact_coplanar_area_overlap',
                           'exact_coplanar_edge_or_point_contact','source_tolerance_contact'})
    result['classes']={k:number(v,True) for k,v in value['classes'].items()}
    if value['coordinate_space']!='surface_RAS' or value['coordinate_unit']!='mm' or value['source_predicate_contacts_are_not_strict_crossings'] is not True:
        raise ValueError('unexpected coordinate/contact semantics')
    result.update(coordinate_space='surface_RAS',coordinate_unit='mm',source_predicate_contacts_are_not_strict_crossings=True)
    if adaptive<=keys:
        if value['strict_relation_backend']!='F64_with_exact_IEEE_fraction_fallback':raise ValueError('unknown backend')
        result['strict_relation_backend']=value['strict_relation_backend']
        for key in adaptive-{'strict_relation_backend'}:result[key]=number(value[key],True)
    return result


def export(*, report_path: Path, timer_path: Path, output: Path):
    """读取本任务原始QC/计时JSON，按固定schema导出新公共JSON。

    输入仅本只读任务，禁止整例生产report/log/影像；输出须不存在。
    包含全部穿越/接触原始聚合指标、输入/源码SHA、CPU预算和冷CLI/RSS，
    不复制任何私人路径、主机/GPU标识或argv。未知键、类型、枚举/哈希
    不合约即抛ValueError，失败不生成输出。返回导出dict，不运行MRI。
    """
    if output.exists():raise FileExistsError(output)
    raw=report_path.read_bytes();d=json.loads(raw)
    common={'schema_version','execution_status','scope','input_sha256','input_hashes_unchanged','terminal_sha256',
            'script_sha256','protocol','protocol_sha256','source_sha256','mesh','initial_crossing_qc','local_query',
            'local_full_proper_contact_legal_sets_equal','source_self_intersecting_faces','affected_faces',
            'proper_crossing_pair_set_sha256','exact_contact_pair_set_sha256','legal_fixed_contact_pair_set_sha256',
            'versions','threads','cpu_affinity','cuda_visible_devices','gpu_computation','gpu_memory_bytes',
            'original_stage_program_called','overall_metric_equivalence','whole_pipeline_speedup',
            'initial_nonzero_crossings_are_not_a_no_new_gate_pass','peak_process_cpu_rss_bytes','wall_seconds_before_report_write'}
    if set(d) not in (common,common|{'prototype_predicate_comparison'}):raise ValueError('unknown producer fields')
    if type(d['schema_version']) is not int or d['schema_version']!=1 or d['execution_status']!='complete_readonly_crossing_check':
        raise ValueError('completed readonly QC required')
    for key in ('input_hashes_unchanged','initial_nonzero_crossings_are_not_a_no_new_gate_pass'):
        if d[key] is not True:raise ValueError('input/scope guard failed')
    for key in ('gpu_computation','original_stage_program_called'):
        if d[key] is not False:raise ValueError('unexpected runtime route')
    for key in ('gpu_memory_bytes','whole_pipeline_speedup'):
        if d[key] is not None:raise ValueError('GPU/whole was not measured')
    if d['cuda_visible_devices']!='' or d['overall_metric_equivalence']!='not_assessed':raise ValueError('unexpected scope')
    if d['local_full_proper_contact_legal_sets_equal']!=[True,True,True] or any(type(v) is not bool for v in d['local_full_proper_contact_legal_sets_equal']):
        raise ValueError('full/local set regression failed')
    exact(d['input_sha256'],{'surf/lh.white','surf/autodet.gw.stats.lh.dat','label/lh.cortex.label',
        'label/lh.cortex+hipamyg.label','mri/brain.finalsurfs.mgz','mri/wm.mgz','mri/aseg.presurf.mgz'})
    inputs={k:digest(v) for k,v in d['input_sha256'].items()}
    source={}
    if not isinstance(d['source_sha256'],dict) or 'fnit.recon_all.white_pial_crossing_python' not in d['source_sha256']:
        raise ValueError('actual crossing helper source binding required')
    for k,v in d['source_sha256'].items():
        if not re.fullmatch(r'fnit\.recon_all\.[A-Za-z_0-9]+',k):raise ValueError('private/unknown source key')
        source[k]=digest(v)
    exact(d['mesh'],{'vertices','faces','fixed_vertices','outside_cortex_elements_different_from_white_at_entry'})
    exact(d['versions'],{'python','numpy','numba','nibabel','scipy'})
    if any(not isinstance(v,str) or not re.fullmatch('[0-9]+[.][0-9]+[.][0-9]+(?:[a-zA-Z0-9.+-]*)',v) for v in d['versions'].values()):
        raise ValueError('unknown version text')
    if d['threads']!=4 or type(d['threads']) is not int or d['cpu_affinity']!=[112,113,114,115]:raise ValueError('resource budget differs')
    result={'schema_version':1,'execution_status':'complete_readonly_crossing_check',
        'scope':'frozen own pial terminal; no optimizer/GPU/whole execution',
        'raw_report_sha256':hashlib.sha256(raw).hexdigest(),'input_sha256':inputs,'source_sha256':source,
        'mesh':{k:number(v,True) for k,v in d['mesh'].items()},'versions':d['versions'],
        'initial_crossing_qc':summary(d['initial_crossing_qc']),'local_query':summary(d['local_query']),
        'local_full_proper_contact_legal_sets_equal':[True,True,True],
        'source_self_intersecting_faces':number(d['source_self_intersecting_faces'],True),'affected_faces':number(d['affected_faces'],True),
        'input_hashes_unchanged':True,'threads':4,'cpu_affinity':[112,113,114,115],
        'gpu_computation':False,'gpu_memory_bytes':None,'original_stage_program_called':False,
        'overall_metric_equivalence':'not_assessed','whole_pipeline_speedup':None,
        'peak_process_cpu_rss_bytes':number(d['peak_process_cpu_rss_bytes'],True),
        'wall_seconds_before_report_write':number(d['wall_seconds_before_report_write'])}
    for k in ('terminal_sha256','script_sha256','protocol_sha256','proper_crossing_pair_set_sha256',
              'exact_contact_pair_set_sha256','legal_fixed_contact_pair_set_sha256'):result[k]=digest(d[k])
    comparison=d.get('prototype_predicate_comparison')
    if comparison is not None:
        counts={'baseline_raw_F64_proper_pairs','adaptive_exact_proper_pairs','removed_raw_prototype_pairs',
                'added_exactly_resolved_pairs','removed_pair_pial_faces_fixed_exact_coincident_with_own_white'}
        exact(comparison,counts|{'baseline_helper_sha256','removed_pair_shared_corner_counts','baseline_raw_summary',
                               'prototype_raw_counts_are_not_physical_crossing_counts'})
        if comparison['prototype_raw_counts_are_not_physical_crossing_counts'] is not True:raise ValueError('prototype scope missing')
        exact(comparison['removed_pair_shared_corner_counts'],{'0','1','2','3'})
        result['prototype_predicate_comparison']={k:number(comparison[k],True) for k in counts}|{
            'baseline_helper_sha256':digest(comparison['baseline_helper_sha256']),
            'removed_pair_shared_corner_counts':{k:number(v,True) for k,v in comparison['removed_pair_shared_corner_counts'].items()},
            'baseline_raw_summary':summary(comparison['baseline_raw_summary']),
            'prototype_raw_counts_are_not_physical_crossing_counts':True}
    traw=timer_path.read_bytes();t=json.loads(traw)
    exact(t,{'schema_version','scope','command_sha256','timer_script_sha256','gpu_memory_bytes',
        'simultaneous_parent_child_peak_cpu_memory_bytes','execution_status','child_exit_code','cold_cli_wall_seconds','max_child_cpu_rss_bytes'})
    if type(t['schema_version']) is not int or t['schema_version']!=1 or t['scope']!='single frozen Python benchmark cold CLI; not end-to-end recon-all':
        raise ValueError('unknown timer scope')
    if t['execution_status']!='complete' or type(t['child_exit_code']) is not int or t['child_exit_code']!=0:
        raise ValueError('cold CLI did not complete')
    if t['gpu_memory_bytes'] is not None or t['simultaneous_parent_child_peak_cpu_memory_bytes'] is not None:raise ValueError('unmeasured resources must remain null')
    result['cold_cli']={'raw_timer_sha256':hashlib.sha256(traw).hexdigest(),'child_exit_code':0,
        'cold_cli_wall_seconds':number(t['cold_cli_wall_seconds']),'max_child_cpu_rss_bytes':number(t['max_child_cpu_rss_bytes'],True),
        'command_sha256':digest(t['command_sha256']),'timer_script_sha256':digest(t['timer_script_sha256']),
        'gpu_memory_bytes':None,'simultaneous_parent_child_peak_cpu_memory_bytes':None}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('report-path','timer-path','output'):p.add_argument('--'+name,type=Path,required=True)
    r=export(**vars(p.parse_args()));print(json.dumps({'status':r['execution_status'],'raw_report_sha256':r['raw_report_sha256']}))
