"""同冻结自产pial的单次联合候选；没有生产接线或参考坐标入口。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time

INPUTS={"surf/lh.white","surf/autodet.gw.stats.lh.dat","label/lh.cortex.label","label/lh.cortex+hipamyg.label","mri/brain.finalsurfs.mgz","mri/wm.mgz","mri/aseg.presurf.mgz"}
MODULES={"fnit.recon_all.place_surface_residual_joint","fnit.recon_all.place_surface_residual_cleanup","fnit.recon_all.place_surface_residual_projection","fnit.recon_all.white_pial_crossing_python"}
PARAMETERS={"max_displacement_mm":1.,"minimum_area_ratio":.2,"clearance_mm":.002,"optimizer_iterations":100}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_protocol(settings):
    """预定typed协议dict→同dict；未知键、bool预算、来源或空间错误失败。

    不运行MRI/优化器；只接受本单组/单LP/单SLSQP配置，几何候选
    约束不是整体等效标准，返回原dict且不更改阈值。
    """
    keys={"schema_version","algorithm","parameters","input_sha256","terminal_sha256","exact_report_sha256","source_sha256","script_sha256","preflight_source_sha256","exact_audit_sha256","distance_source_sha256","expected_group_vertices","expected_group_fixed_vertices","expected_group_movable_vertices","expected_source_directed_pairs","expected_float64_pairs","threads","cpu_affinity","cuda_visible_devices","scope","production_default_change","overall_metric_equivalence","candidate_constraints_are_not_overall_equivalence_thresholds","exact_self_guard"}
    if type(settings) is not dict or set(settings)!=keys:raise ValueError("exact fixed joint protocol required")
    if type(settings['schema_version']) is not int or settings['schema_version']!=1 or settings['algorithm']!='independent_joint_fixed_face_family_v1':raise ValueError("known joint protocol version required")
    if settings['parameters']!=PARAMETERS or any(type(settings['parameters'].get(k)) is not (int if k=='optimizer_iterations' else float) for k in PARAMETERS):raise ValueError("fixed typed candidate parameters required")
    if type(settings['input_sha256']) is not dict or set(settings['input_sha256'])!=INPUTS or type(settings['source_sha256']) is not dict or set(settings['source_sha256'])!=MODULES:raise ValueError("complete input/source maps required")
    digests=list(settings['input_sha256'].values())+list(settings['source_sha256'].values())+[settings[k] for k in ('terminal_sha256','exact_report_sha256','script_sha256','preflight_source_sha256','exact_audit_sha256','distance_source_sha256')]
    if any(type(v) is not str or not re.fullmatch('[0-9a-f]{64}',v) for v in digests):raise ValueError("SHA-256 digests required")
    integer_values={"expected_group_vertices":35,"expected_group_fixed_vertices":20,"expected_group_movable_vertices":15,"expected_source_directed_pairs":80,"expected_float64_pairs":40,"threads":4}
    if any(type(settings[k]) is not int or settings[k]!=v for k,v in integer_values.items()):raise ValueError("fixed complete input grid and budget required")
    if settings['cpu_affinity']!=[112,113,114,115] or any(type(v) is not int for v in settings['cpu_affinity']) or settings['cuda_visible_devices']!='':raise ValueError("fixed hidden-CUDA four-core budget required")
    if settings['scope']!='frozen own seven-input residual stage; not raw T1 or original program comparison' or settings['production_default_change'] is not False or settings['overall_metric_equivalence']!='not_assessed' or settings['candidate_constraints_are_not_overall_equivalence_thresholds'] is not True:raise ValueError("known candidate scope required")
    if settings['exact_self_guard']!='complete_before_after_affected_bbox_exact_no_shared_ID_exemption':raise ValueError("complete independent exact affected-face guard required")
    return settings


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run(*,installed_directory:Path,overlay_directory:Path,diagnostic_case:Path,input_hashes:Path,
        protocol:Path,exact_report:Path,preflight_source:Path,exact_audit_source:Path,
        distance_module:Path,output_directory:Path,threads:int=4):
    """10必填Path和threads=4→完整候选报告；所有时间含校验/加载/读写。

    installed_directory冻结依赖包；overlay_directory冻结四模块；
    diagnostic_case含subject七前置及自产cleanup-result.private.npy。
    input_hashes七SHA，protocol预定参数/来源/35点组，exact_report为
    完整既有40对proper名单；preflight_source为固定域/名单预检工具，
    exact_audit_source为独立精确逐边裁剪，distance_module为已验证的
    双向顶点到三角面函数，新output_directory须不存在。末态float32
    (N,3)、有序面整数(F,3)、固定bool(N,)，surface RAS/mm。
    私有输出result NPY/表面、有限trial NPY/表面、report/COMPLETE。
    拒绝时result等entry，trial保留。无参考参数/原程序/GPU；非法
    协议/源码/哈希/空间/资源或写出差异抛异常，不能当生产完成。
    """
    started=time.perf_counter();settings=validate_protocol(json.loads(protocol.read_text()))
    if type(threads) is not int or threads!=4 or threads!=settings['threads']:raise ValueError("fixed four threads required")
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or sorted(os.sched_getaffinity(0))!=settings['cpu_affinity']:raise ValueError("actual CPU/CUDA budget differs")
    if output_directory.exists():raise FileExistsError(output_directory)
    expected=json.loads(input_hashes.read_text())
    if expected!=settings['input_sha256']:raise ValueError("fixed seven input map differs")
    for path,key in ((Path(__file__),'script_sha256'),(preflight_source,'preflight_source_sha256'),(exact_audit_source,'exact_audit_sha256'),(distance_module,'distance_source_sha256'),(exact_report,'exact_report_sha256')):
        if sha(path)!=settings[key]:raise ValueError("frozen task source/report differs")
    fixed_task_files={path:sha(path) for path in (Path(__file__),protocol,input_hashes,preflight_source,exact_audit_source,distance_module,exact_report)}
    terminal_path=diagnostic_case/'cleanup-result.private.npy'
    before={k:sha(diagnostic_case/'subject'/k) for k in sorted(INPUTS)}
    if before!=expected or sha(terminal_path)!=settings['terminal_sha256']:raise ValueError("frozen own stage input differs")
    preflight=load_module('joint_fixed_preflight',preflight_source);exact=load_module('joint_independent_exact',exact_audit_source);distance=load_module('joint_existing_distance',distance_module)
    observed=json.loads(exact_report.read_text())
    if observed['execution_status']!='complete_readonly_exact_pair_audit' or observed['input_sha256']!=expected or observed['terminal_sha256']!=settings['terminal_sha256']:raise ValueError("complete own exact list provenance required")
    sys.path.insert(0,str(installed_directory));import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(overlay_directory))
    import numpy as np
    import nibabel.freesurfer as fs
    import numba
    import scipy
    numba.set_num_threads(threads)
    white,faces=fs.read_geometry(str(diagnostic_case/'subject/surf/lh.white'))
    white,faces,cortex,rip,fixed=preflight.checked_fixed_domain(white_vertices=white,faces=faces,
        cortex_indices=fs.read_label(str(diagnostic_case/'subject/label/lh.cortex.label')),
        hipamyg_indices=fs.read_label(str(diagnostic_case/'subject/label/lh.cortex+hipamyg.label')))
    entry=np.load(terminal_path)
    if entry.dtype!=np.float32 or entry.shape!=white.shape or not np.isfinite(entry).all():raise ValueError("same ordered finite float32 terminal required")
    if type(observed.get('faces')) is not int or observed['faces']!=len(faces) or type(observed.get('vertices')) is not int or observed['vertices']!=len(white) or type(observed.get('fixed_vertices')) is not int or observed['fixed_vertices']!=int(fixed.sum()) or type(observed.get('states')) is not list or len(observed['states'])!=2:raise ValueError("actual list grid/fixed domain differs")
    rows=preflight.checked_pair_rows(record=observed['states'][0],state='entry',face_count=len(faces))
    group=np.unique(faces[np.array([r['pair'] for r in rows])].ravel())
    if len(group)!=settings['expected_group_vertices'] or int(fixed[group].sum())!=settings['expected_group_fixed_vertices'] or int((~fixed[group]).sum())!=settings['expected_group_movable_vertices']:raise ValueError("frozen complete group differs")
    from fnit.recon_all.place_surface_residual_joint import resolve_joint_residual_intersections
    from fnit.recon_all.compare_subject import _topology
    from fnit.recon_all.topology_volume_python import _nonmanifold_vertex_links
    from fnit.recon_all.mris_remove_intersection_python import _write_preserving_surface
    def source_hashes():
        return {name:sha(module.__file__) for name,module in sorted(sys.modules.copy().items()) if name.startswith('fnit.recon_all.') and getattr(module,'__file__',None) and Path(module.__file__).is_file()}
    bindings=source_hashes()
    if any(bindings.get(k)!=v for k,v in settings['source_sha256'].items()):raise ValueError("loaded frozen solver modules differ")
    topology=_topology(entry.astype(float),faces);links=int(_nonmanifold_vertex_links(faces,len(entry)))
    directed=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]));_,groups=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True)
    conflicts=int(np.count_nonzero(np.bincount(groups,weights=np.where(directed[:,0]<directed[:,1],1,-1))))
    if not topology['valid'] or links or conflicts:raise ValueError("initial closed manifold oriented grid required")
    result,trial,diagnosis=resolve_joint_residual_intersections(vertices=entry,faces=faces,fixed_vertices=fixed,white_vertices=white,
        exact_relation=exact.exact_clip_relation,**settings['parameters'])
    if diagnosis['source_directed_pairs_before']!=settings['expected_source_directed_pairs'] or diagnosis['float64_pairs_before']!=settings['expected_float64_pairs'] or [r['pair'] for r in diagnosis['exact_before']]!=[r['pair'] for r in rows]:raise ValueError("actual complete initial pairs differ")
    if not np.array_equal(result[fixed].view(np.uint32),entry[fixed].view(np.uint32)):raise RuntimeError("fixed domain bits changed")
    output_directory.mkdir(parents=True);output_hashes={};geometry={};distances={}
    for name,xyz in [('result',result)]+([] if trial is None else [('trial',trial)]):
        npy=output_directory/(name+'.private.npy');surface=output_directory/(name+'.pial');np.save(npy,xyz)
        _write_preserving_surface(diagnostic_case/'subject/surf/lh.white',surface,xyz)
        decoded,ordered=fs.read_geometry(str(surface))
        if not np.array_equal(decoded.astype(np.float32).view(np.uint32),xyz.view(np.uint32)) or not np.array_equal(ordered,faces):raise RuntimeError("saved ordered geometry differs")
        output_hashes[npy.name]=sha(npy);output_hashes[surface.name]=sha(surface)
        original_normal=np.cross(entry.astype(float)[faces[:,1]]-entry.astype(float)[faces[:,0]],entry.astype(float)[faces[:,2]]-entry.astype(float)[faces[:,0]])
        normal=np.cross(xyz.astype(float)[faces[:,1]]-xyz.astype(float)[faces[:,0]],xyz.astype(float)[faces[:,2]]-xyz.astype(float)[faces[:,0]])
        movement=np.linalg.norm(xyz.astype(float)-entry,axis=1);changed=movement[movement!=0]
        geometry[name]={'finite':bool(np.isfinite(xyz).all()),'ordered_faces_equal':True,'fixed_different_elements':int(np.count_nonzero(xyz[fixed].view(np.uint32)!=entry[fixed].view(np.uint32))),
            'different_elements':int(np.count_nonzero(xyz.view(np.uint32)!=entry.view(np.uint32))),'changed_vertices':len(changed),'maximum_displacement_mm':float(movement.max()),
            'p99_displacement_mm':float(np.quantile(movement,.99)),'changed_vertex_p99_mm':float(np.quantile(changed,.99)) if len(changed) else 0.,
            'minimum_area_ratio':float((np.linalg.norm(normal,axis=1)/np.linalg.norm(original_normal,axis=1)).min()),
            'zero_area_faces':int(np.count_nonzero(np.linalg.norm(normal,axis=1)==0)),
            'nonpositive_initial_normal_dot_faces':int(np.count_nonzero(np.sum(normal*original_normal,axis=1)<=0))}
        if np.array_equal(xyz.view(np.uint32),entry.view(np.uint32)):
            distances[name]={'xyz_to_entry':{'mean_mm':0.,'p99_mm':0.,'max_mm':0.},'entry_to_xyz':{'mean_mm':0.,'p99_mm':0.,'max_mm':0.},'calculation':'bitwise_equal_geometry_identity'}
        else:
            tick=time.perf_counter();values={}
            for direction,a,b in (('xyz_to_entry',xyz,entry),('entry_to_xyz',entry,xyz)):
                value=distance._point_to_mesh(a,b,faces)
                values[direction]={'mean_mm':float(value.mean()),'p99_mm':float(np.quantile(value,.99)),'max_mm':float(value.max())}
            distances[name]=values|{'calculation':'existing_bidirectional_vertex_to_triangle','seconds':time.perf_counter()-tick}
    if before!={k:sha(diagnostic_case/'subject'/k) for k in sorted(INPUTS)} or sha(terminal_path)!=settings['terminal_sha256'] or sha(exact_report)!=settings['exact_report_sha256']:raise RuntimeError("frozen input sources changed")
    final_bindings=source_hashes()
    if any(final_bindings.get(k)!=v for k,v in bindings.items()) or any(sha(path)!=digest for path,digest in fixed_task_files.items()):raise RuntimeError("loaded task sources changed during stage")
    report={'schema_version':1,'execution_status':'complete_independent_joint_diagnosis','scope':settings['scope'],'protocol_sha256':sha(protocol),
        'script_sha256':sha(__file__),'input_sha256':before,'terminal_sha256':settings['terminal_sha256'],'exact_report_sha256':sha(exact_report),
        'preflight_source_sha256':sha(preflight_source),'exact_audit_sha256':sha(exact_audit_source),'distance_source_sha256':sha(distance_module),'source_version_bindings':final_bindings,
        'mesh':{'vertices':len(entry),'faces':len(faces),'fixed_vertices':int(fixed.sum()),'outside_cortex_elements_different_from_white_at_entry':int(np.count_nonzero(entry[~cortex]!=white[~cortex]))},
        'topology_shared_by_entry_trial_and_result':topology,'nonmanifold_vertex_links':links,'edge_orientation_conflicts':conflicts,
        'diagnosis':diagnosis,'geometry':geometry,'bidirectional_vertex_to_triangle':distances,'output_sha256':output_hashes,
        'threads':threads,'cpu_affinity':sorted(os.sched_getaffinity(0)),'gpu_computation':False,'gpu_memory_bytes':None,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'numba':numba.__version__,'scipy':scipy.__version__,'nibabel':sys.modules['nibabel'].__version__},
        'coordinate_space':'surface_RAS','coordinate_unit':'mm','inputs_unchanged':True,'trial_retained':trial is not None,'original_stage_program_called':False,
        'strict_reproduction':'independent_algorithm_not_source_reproduction','overall_metric_equivalence':'not_assessed','production_default_change':False,
        'wall_seconds_before_report_write':time.perf_counter()-started,'peak_process_cpu_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
    (output_directory/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    (output_directory/'COMPLETE.json').write_text(json.dumps({'execution_status':report['execution_status'],'stage_quality':diagnosis['status'],'report_sha256':sha(output_directory/'report.json'),'output_sha256':output_hashes,'wall_seconds_after_report_write':time.perf_counter()-started},sort_keys=True,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('installed-directory','overlay-directory','diagnostic-case','input-hashes','protocol','exact-report','preflight-source','exact-audit-source','distance-module','output-directory'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    report=run(**vars(parser.parse_args()));print(json.dumps({'execution_status':report['execution_status'],'quality':report['diagnosis']['status'],'decision':report['diagnosis']['decision']}))
