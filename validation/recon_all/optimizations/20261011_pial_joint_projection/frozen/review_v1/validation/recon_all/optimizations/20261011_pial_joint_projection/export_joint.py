"""联合阶段原样数值的固定typed导出；拒绝未知键/私人字符串，不运行MRI。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import re

SCHEMA_SHA='8b6a919f5d14f7e16e47c81013187a8b7f6468a1c22796c00723b7bf1f867388'
INPUTS={'surf/lh.white','surf/autodet.gw.stats.lh.dat','label/lh.cortex.label','label/lh.cortex+hipamyg.label','mri/brain.finalsurfs.mgz','mri/wm.mgz','mri/aseg.presurf.mgz'}
PLANES={'normal_positive','normal_negative','edge_outside_0','edge_outside_1','edge_outside_2'}
STATUS={'passed_zero_in_source_float64_and_affected_exact','failed_residual','already_zero_in_source_and_float64'}


def checked_report(d,s):
    """已声明完整producer dict及冻结schema模块→同dict；仅验证不改值。"""
    def boolean(v):
        if type(v) is not bool:raise ValueError('typed bool required')
    def signed(v):
        if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('typed finite signed scalar required')
    def rec(v,integers=(),floats=(),bools=(),constants=None,extra=(),hashes=()):
        constants=constants or {};s.exact(v,set(integers)|set(floats)|set(bools)|set(constants)|set(extra)|set(hashes))
        for k in integers:s.number(v[k],True)
        for k in floats:s.number(v[k])
        for k in bools:boolean(v[k])
        for k,expected in constants.items():
            if v[k]!=expected or type(v[k]) is not type(expected):raise ValueError('unknown fixed enum/value')
        for k in hashes:s.digest(v[k])
    def pairs(v,faces,unordered=False,independent_surfaces=False):
        if type(v) is not list:raise ValueError('pair list required')
        for pair in v:
            if type(pair) is not list or len(pair)!=2 or any(type(x) is not int for x in pair) or not 0<=pair[0]<faces or not 0<=pair[1]<faces or pair[0]==pair[1] and not independent_surfaces:raise ValueError('pair IDs invalid')
        if [tuple(x) for x in v]!=sorted(set(tuple(x) for x in v)):raise ValueError('pair list must be sorted unique')
        if unordered and any(a>=b for a,b in v):raise ValueError('unordered pairs require ascending IDs')
    constants={'schema_version':1,'execution_status':'complete_independent_joint_diagnosis','scope':'frozen own seven-input residual stage; not raw T1 or original program comparison','threads':4,'cpu_affinity':[112,113,114,115], 'gpu_computation':False,'gpu_memory_bytes':None,'coordinate_space':'surface_RAS','coordinate_unit':'mm','inputs_unchanged':True,'original_stage_program_called':False,'strict_reproduction':'independent_algorithm_not_source_reproduction','overall_metric_equivalence':'not_assessed','production_default_change':False}
    rec(d,integers=('nonmanifold_vertex_links','edge_orientation_conflicts','peak_process_cpu_rss_bytes'),floats=('wall_seconds_before_report_write',),bools=('trial_retained',),constants=constants,
        hashes=('protocol_sha256','script_sha256','terminal_sha256','exact_report_sha256','preflight_source_sha256','exact_audit_sha256','distance_source_sha256'),extra=('input_sha256','source_version_bindings','mesh','topology_shared_by_entry_trial_and_result','diagnosis','geometry','bidirectional_vertex_to_triangle','output_sha256','versions'))
    s.exact(d['input_sha256'],INPUTS)
    for v in d['input_sha256'].values():s.digest(v)
    if type(d['source_version_bindings']) is not dict or not d['source_version_bindings']:raise ValueError('complete loaded source map required')
    for name,value in d['source_version_bindings'].items():
        if type(name) is not str or not re.fullmatch(r'fnit\.recon_all\.[A-Za-z_0-9.]+',name):raise ValueError('unknown/private source key')
        s.digest(value)
    required_modules={'fnit.recon_all.place_surface_residual_joint','fnit.recon_all.place_surface_residual_cleanup','fnit.recon_all.place_surface_residual_projection','fnit.recon_all.white_pial_crossing_python'}
    if not required_modules<=d['source_version_bindings'].keys():raise ValueError('all four actual solver source bindings required')
    s.exact(d['versions'],{'python','numpy','numba','scipy','nibabel'})
    if any(type(v) is not str or not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+-]*',v) for v in d['versions'].values()):raise ValueError('unknown version text')
    rec(d['mesh'],integers=('vertices','faces','fixed_vertices','outside_cortex_elements_different_from_white_at_entry'))
    n,f=d['mesh']['vertices'],d['mesh']['faces']
    if not n or not f or d['mesh']['fixed_vertices']>n:raise ValueError('valid mesh domain required')
    rec(d['topology_shared_by_entry_trial_and_result'],integers=('vertices','faces','edges','components','boundary_edges','nonmanifold_edges','degenerate_index_faces','duplicate_faces'),constants={'euler':2,'valid':True})
    topology=d['topology_shared_by_entry_trial_and_result']
    if topology['vertices']!=n or topology['faces']!=f or topology['components']!=1 or n-topology['edges']+f!=2 or 2*topology['edges']!=3*f or any(topology[k]!=0 for k in ('boundary_edges','nonmanifold_edges','degenerate_index_faces','duplicate_faces')) or d['nonmanifold_vertex_links'] or d['edge_orientation_conflicts']:raise ValueError('closed oriented manifold required')
    r=d['diagnosis'];trial=d['trial_retained']
    common={'schema_version','algorithm','family','optimizer','parameters','source_directed_pairs_before','float64_pairs_before','exact_before','self_qc_before','strict_reproduction','overall_metric_equivalence','original_stage_program_called','production_default_change','white_pial_guard','exact_affected_self_guard','decision','status','source_directed_pairs_after','float64_pairs_after','final_coordinates_equal_entry','fixed_different_elements','full_resolver_seconds','self_qc_final'}
    more={'exact_trial','self_qc_trial','trial_checks','source_directed_pairs_trial','float64_pairs_trial','new_source_pairs','new_float64_pairs','trial_changed_vertices','trial_max_displacement_mm','trial_p99_displacement_mm','trial_changed_vertex_p99_mm','trial_minimum_area_ratio','trial_nonpositive_initial_normal_dot_faces','trial_fixed_different_elements'}
    s.exact(r,common|(more if trial else set()))
    rec(r['parameters'],floats=('max_displacement_mm','minimum_area_ratio','clearance_mm'),integers=('optimizer_iterations',))
    if r['parameters']!={'max_displacement_mm':1.,'minimum_area_ratio':.2,'clearance_mm':.002,'optimizer_iterations':100}:raise ValueError('unchanged candidate constraints required')
    for k in ('source_directed_pairs_before','float64_pairs_before','source_directed_pairs_after','float64_pairs_after','fixed_different_elements'):s.number(r[k],True)
    s.number(r['full_resolver_seconds']);boolean(r['final_coordinates_equal_entry'])
    if r['schema_version']!=1 or type(r['schema_version']) is not int or r['algorithm']!='independent_joint_fixed_face_family_v1' or r['strict_reproduction']!=constants['strict_reproduction'] or r['overall_metric_equivalence']!='not_assessed' or r['original_stage_program_called'] is not False or r['production_default_change'] is not False or r['status'] not in STATUS:raise ValueError('known independent diagnosis required')
    family=r['family'];rec(family,integers=('group_vertices','fixed_group_vertices','moving_vertices','slsqp_dimensions','affected_faces','halfspace_rows'),bools=('existing_fixed_contact_margin_changed',),constants={'family_completeness':'selected fixed-face halfspaces only; not all collision-free geometries'},extra=('fixed_face_selections',))
    if family['group_vertices']!=family['fixed_group_vertices']+family['moving_vertices'] or family['slsqp_dimensions']!=3*family['moving_vertices'] or family['existing_fixed_contact_margin_changed']:raise ValueError('group/margin inconsistency')
    if type(family['fixed_face_selections']) is not list:raise ValueError('full selection list required')
    for selection in family['fixed_face_selections']:
        rec(selection,integers=('fixed_face','moving_face'),extra=('pair','candidate_planes','selected_plane','decision'));pairs([selection['pair']],f)
        if selection['fixed_face'] not in selection['pair'] or selection['moving_face'] not in selection['pair'] or selection['fixed_face']==selection['moving_face']:raise ValueError('invalid selected face identity')
        if selection['selected_plane'] not in PLANES|{None} or selection['decision'] not in {'selected_once','no_fixed_compatible_halfspace_in_selected_family'}:raise ValueError('unknown plane selection')
        if type(selection['candidate_planes']) is not list or [v.get('plane') for v in selection['candidate_planes']]!=['normal_positive','normal_negative','edge_outside_0','edge_outside_1','edge_outside_2']:raise ValueError('five original directions required')
        for v in selection['candidate_planes']:
            rec(v,integers=('moving_corners',),floats=('optimistic_squared_projection_mm2',),extra=('plane','fixed_corner_signs','decision'))
            if v['decision'] not in {'eligible_fixed_domain','fixed_corner_wrong_side'} or type(v['fixed_corner_signs']) is not list or any(type(x) is not int or x not in (-1,0,1) for x in v['fixed_corner_signs']) or len(v['fixed_corner_signs'])+v['moving_corners']!=3:raise ValueError('exact fixed domain signs required')
    opt=r['optimizer'];allowed={'empty_selected_family','selected_family_lp_failed','selected_family_optimizer_infeasible','finite_quantized_trial'}
    if opt.get('decision') not in allowed:raise ValueError('unknown optimizer decision')
    keys={'decision','lp_calls','slsqp_calls'}
    if opt['lp_calls']:keys|={'lp_status','lp_success','lp_seconds','lp_dimensions_with_l1_auxiliary','lp_infeasible_claim_scope'}
    if opt['slsqp_calls']:keys|={'slsqp_dimensions','optimizer_status','optimizer_success','optimizer_iterations','optimizer_seconds','minimum_optimizer_constraint'}
    s.exact(opt,keys)
    for k in ('lp_calls','slsqp_calls'):
        s.number(opt[k],True)
        if opt[k] not in (0,1):raise ValueError('single attempt only')
    if opt['lp_calls']:
        for k in ('lp_status','lp_dimensions_with_l1_auxiliary'):s.number(opt[k],True)
        boolean(opt['lp_success']);s.number(opt['lp_seconds'])
        if opt['lp_dimensions_with_l1_auxiliary']!=2*family['slsqp_dimensions'] or opt['lp_infeasible_claim_scope']!='this preselected halfspace family under coordinate box only':raise ValueError('LP scope mismatch')
    if opt['slsqp_calls']:
        for k in ('slsqp_dimensions','optimizer_status','optimizer_iterations'):s.number(opt[k],True)
        boolean(opt['optimizer_success']);s.number(opt['optimizer_seconds']);signed(opt['minimum_optimizer_constraint'])
        if opt['slsqp_dimensions']!=family['slsqp_dimensions'] or opt['optimizer_iterations']>100:raise ValueError('optimizer budget mismatch')
    if trial!=(opt['decision']=='finite_quantized_trial') or opt['slsqp_calls']>opt['lp_calls'] or opt['slsqp_calls'] and not opt['lp_success']:raise ValueError('optimizer/trial route mismatch')
    def self_qc(q):rec(q,integers=('source_directed_pairs','float64_pairs','source_faces','float64_faces','source_vertices','float64_vertices'))
    for k in ('self_qc_before','self_qc_final')+ (('self_qc_trial',) if trial else ()):self_qc(r[k])
    def exact_rows(rows,source_count,float64_count):
        if type(rows) is not list:raise ValueError('complete exact detector list required')
        for v in rows:
            rec(v,integers=('exact_relation','source_directed_orientations'),bools=('float64_positive',),extra=('pair',));pairs([v['pair']],f,True)
            if v['exact_relation'] not in (0,1,2,3) or v['source_directed_orientations']>2 or not (v['source_directed_orientations'] or v['float64_positive']):raise ValueError('known exact/member relation required')
        pairs([v['pair'] for v in rows],f,True)
        if sum(v['source_directed_orientations'] for v in rows)!=source_count or sum(v['float64_positive'] for v in rows)!=float64_count:raise ValueError('full exact rows/counts inconsistent')
    exact_rows(r['exact_before'],r['source_directed_pairs_before'],r['float64_pairs_before'])
    for q,state in ((r['self_qc_before'],'before'),(r['self_qc_final'],'after')):
        if q['source_directed_pairs']!=r['source_directed_pairs_'+state] or q['float64_pairs']!=r['float64_pairs_'+state] or q['source_faces']>f or q['float64_faces']>f or q['source_vertices']>n or q['float64_vertices']>n:raise ValueError('self QC domain/count mismatch')
    if any(v['exact_relation']!=1 for v in r['exact_before']):raise ValueError('all initial detector pairs must be proper')
    if trial:
        for k in ('source_directed_pairs_trial','float64_pairs_trial','trial_changed_vertices','trial_nonpositive_initial_normal_dot_faces','trial_fixed_different_elements'):s.number(r[k],True)
        for k in ('trial_max_displacement_mm','trial_p99_displacement_mm','trial_changed_vertex_p99_mm','trial_minimum_area_ratio'):s.number(r[k])
        exact_rows(r['exact_trial'],r['source_directed_pairs_trial'],r['float64_pairs_trial'])
        for k in ('new_source_pairs','new_float64_pairs'):pairs(r[k],f,k=='new_float64_pairs')
        if r['self_qc_trial']['source_directed_pairs']!=r['source_directed_pairs_trial'] or r['self_qc_trial']['float64_pairs']!=r['float64_pairs_trial']:raise ValueError('trial self QC count mismatch')
        checks={'fixed_coordinates_equal','only_group_movable_vertices_changed','euclidean_displacement_within_limit','all_initial_normal_dots_positive','all_area_ratios_within_limit','source_strict_subset','float64_strict_subset','no_new_exact_affected_proper_pairs','no_new_exact_affected_coplanar_area_pairs','no_new_exact_affected_positive_pairs','no_new_white_proper_pairs','no_new_nonmedial_exact_contacts'}
        s.exact(r['trial_checks'],checks)
        for v in r['trial_checks'].values():boolean(v)
        g=r['white_pial_guard'];rec(g,integers=('initial_nonmedial_exact_contacts','trial_nonmedial_exact_contacts'),bools=('trial_zero_proper','trial_zero_nonmedial_contact','final_zero_proper','final_zero_nonmedial_contact'),extra=('initial','trial','new_proper_pairs','new_nonmedial_exact_contact_pairs'))
        s.summary(g['initial']);s.summary(g['trial'])
        for k in ('new_proper_pairs','new_nonmedial_exact_contact_pairs'):pairs(g[k],f,independent_surfaces=True)
        e=r['exact_affected_self_guard'];rec(e,integers=('initial_proper_pairs','trial_proper_pairs','final_proper_pairs','initial_coplanar_area_pairs','trial_coplanar_area_pairs','final_coplanar_area_pairs'),bools=('unchanged_faces_retained_bitwise',),constants={'global_existing_exact_intersections':'not_assessed'},extra=('initial','trial','new_proper_pairs','new_coplanar_area_pairs','new_positive_pairs'))
        for state in ('initial','trial'):
            rec(e[state],integers=('query_faces','complete_bbox_pairs','exact_tested_pairs','positive_pairs_with_shared_vertex_ids'),floats=('bbox_enumeration_seconds','exact_classification_seconds'),constants={'candidate_rule':'inclusive original-IEEE bounds; all grid faces; no shared-ID exclusion','global_exact_scope':'not_assessed'},extra=('exact_classes',))
            s.exact(e[state]['exact_classes'],{'none','proper','edge_point_contact','coplanar_area_overlap','coplanar_edge_point_contact'})
            for v in e[state]['exact_classes'].values():s.number(v,True)
            if sum(e[state]['exact_classes'].values())!=e[state]['exact_tested_pairs'] or e[state]['exact_tested_pairs']!=e[state]['complete_bbox_pairs']:raise ValueError('complete bbox/clip counts inconsistent')
        for k in ('new_proper_pairs','new_coplanar_area_pairs','new_positive_pairs'):pairs(e[k],f,True)
        for state in ('initial','trial'):
            if e[state+'_proper_pairs']!=e[state]['exact_classes']['proper'] or e[state+'_coplanar_area_pairs']!=e[state]['exact_classes']['coplanar_area_overlap'] or e[state]['query_faces']!=family['affected_faces']:raise ValueError('exact local classes/domain mismatch')
        if r['decision'] not in {'accepted_once','rejected_by_complete_quality_gates'} or (r['decision']=='accepted_once')!=all(r['trial_checks'].values()):raise ValueError('acceptance inconsistent with gates')
        accepted=r['decision']=='accepted_once';final='trial' if accepted else 'initial'
        if e['final_proper_pairs']!=e[final+'_proper_pairs'] or e['final_coplanar_area_pairs']!=e[final+'_coplanar_area_pairs']:raise ValueError('returned exact QC state mismatch')
        if e['unchanged_faces_retained_bitwise']!=r['trial_checks']['only_group_movable_vertices_changed']:raise ValueError('exact local unchanged-domain proof mismatch')
        for check,key,block in (('no_new_exact_affected_proper_pairs','new_proper_pairs',e),('no_new_exact_affected_coplanar_area_pairs','new_coplanar_area_pairs',e),('no_new_exact_affected_positive_pairs','new_positive_pairs',e),('no_new_white_proper_pairs','new_proper_pairs',g),('no_new_nonmedial_exact_contacts','new_nonmedial_exact_contact_pairs',g)):
            if r['trial_checks'][check]!=(not block[key]):raise ValueError('new pair list/acceptance gate mismatch')
        if g['trial_zero_proper']!=(g['trial']['proper_crossing_pairs']==0) or g['trial_zero_nonmedial_contact']!=(g['trial_nonmedial_exact_contacts']==0) or g['final_zero_proper']!=(g[final]['proper_crossing_pairs']==0) or g['final_zero_nonmedial_contact']!=(g[final+'_nonmedial_exact_contacts']==0):raise ValueError('white trial/final zero semantics mismatch')
    elif r['white_pial_guard'] is not None or r['exact_affected_self_guard'] is not None or r['decision']!=opt['decision']:raise ValueError('no-trial guards must stay unmeasured')
    names={'result','trial'} if trial else {'result'}
    s.exact(d['geometry'],names);s.exact(d['bidirectional_vertex_to_triangle'],names)
    for name in names:
        rec(d['geometry'][name],integers=('fixed_different_elements','different_elements','changed_vertices','zero_area_faces','nonpositive_initial_normal_dot_faces'),floats=('maximum_displacement_mm','p99_displacement_mm','changed_vertex_p99_mm','minimum_area_ratio'),constants={'finite':True,'ordered_faces_equal':True})
        v=d['bidirectional_vertex_to_triangle'][name]
        if v.get('calculation') not in {'bitwise_equal_geometry_identity','existing_bidirectional_vertex_to_triangle'}:raise ValueError('known distance boundary required')
        s.exact(v,{'xyz_to_entry','entry_to_xyz','calculation'}|({'seconds'} if v['calculation']=='existing_bidirectional_vertex_to_triangle' else set()))
        for key in ('xyz_to_entry','entry_to_xyz'):rec(v[key],floats=('mean_mm','p99_mm','max_mm'))
        if 'seconds' in v:s.number(v['seconds'])
    outputs={'result.private.npy','result.pial'}|({'trial.private.npy','trial.pial'} if trial else set());s.exact(d['output_sha256'],outputs)
    for v in d['output_sha256'].values():s.digest(v)
    if r['fixed_different_elements'] or d['geometry']['result']['fixed_different_elements']:raise ValueError('returned fixed coordinates changed')
    if r['decision']!='accepted_once' and (not r['final_coordinates_equal_entry'] or d['geometry']['result']['different_elements'] or r['source_directed_pairs_after']!=r['source_directed_pairs_before'] or r['float64_pairs_after']!=r['float64_pairs_before']):raise ValueError('rejected result must retain entry')
    accepted=r['decision']=='accepted_once'
    passed=accepted and not r['source_directed_pairs_after'] and not r['float64_pairs_after'] and not r['exact_affected_self_guard']['final_proper_pairs'] and not r['exact_affected_self_guard']['final_coplanar_area_pairs']
    expected_status='passed_zero_in_source_float64_and_affected_exact' if passed else 'failed_residual' if r['source_directed_pairs_before'] or r['float64_pairs_before'] else 'already_zero_in_source_and_float64'
    if r['status']!=expected_status:raise ValueError('quality status/count scope mismatch')
    return d


def export(*,report_path:Path,timer_path:Path,complete_path:Path,schema_source:Path,output:Path):
    """五必填Path→新公共JSON；typed producer/冷CLI/COMPLETE与成熟schema。

    schema_source固定SHA，输出须不存在。保留全部失败/名单/数值/时间
    和来源SHA，不读坐标/影像，不替换私人字符串；未知字段/类型/
    枚举、计数不一致、false退出码或收据不合抛ValueError且不写文件。
    返回白名单完整报告及计时和三原始收据SHA；无MRI/GPU执行。
    """
    if output.exists():raise FileExistsError(output)
    if hashlib.sha256(schema_source.read_bytes()).hexdigest()!=SCHEMA_SHA:raise ValueError('fixed shared schema source required')
    spec=importlib.util.spec_from_file_location('joint_safe_shared_schema',schema_source);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    raw=report_path.read_bytes();d=checked_report(json.loads(raw),s);timer_raw=timer_path.read_bytes();t=json.loads(timer_raw)
    s.exact(t,{'schema_version','scope','command_sha256','timer_script_sha256','gpu_memory_bytes','simultaneous_parent_child_peak_cpu_memory_bytes','execution_status','child_exit_code','cold_cli_wall_seconds','max_child_cpu_rss_bytes'})
    if type(t['schema_version']) is not int or t['schema_version']!=1 or t['scope']!='single frozen Python benchmark cold CLI; not end-to-end recon-all' or t['execution_status']!='complete' or type(t['child_exit_code']) is not int or t['child_exit_code']!=0 or t['gpu_memory_bytes'] is not None or t['simultaneous_parent_child_peak_cpu_memory_bytes'] is not None:raise ValueError('complete measured cold CLI required')
    for k in ('command_sha256','timer_script_sha256'):s.digest(t[k])
    s.number(t['cold_cli_wall_seconds']);s.number(t['max_child_cpu_rss_bytes'],True)
    complete_raw=complete_path.read_bytes();c=json.loads(complete_raw);s.exact(c,{'execution_status','stage_quality','report_sha256','output_sha256','wall_seconds_after_report_write'})
    if c['execution_status']!=d['execution_status'] or c['stage_quality']!=d['diagnosis']['status'] or c['report_sha256']!=hashlib.sha256(raw).hexdigest() or c['output_sha256']!=d['output_sha256']:raise ValueError('COMPLETE differs from actual report/output')
    s.number(c['wall_seconds_after_report_write'])
    result=d|{'cold_cli':t,'raw_report_sha256':hashlib.sha256(raw).hexdigest(),'raw_timer_sha256':hashlib.sha256(timer_raw).hexdigest(),'raw_complete_sha256':hashlib.sha256(complete_raw).hexdigest(),'safe_exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'shared_schema_sha256':SCHEMA_SHA}
    output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('report-path','timer-path','complete-path','schema-source','output'):parser.add_argument('--'+name,type=Path,required=True)
    result=export(**vars(parser.parse_args()));print(json.dumps({'execution_status':result['execution_status'],'quality':result['diagnosis']['status'],'raw_report_sha256':result['raw_report_sha256']}))
