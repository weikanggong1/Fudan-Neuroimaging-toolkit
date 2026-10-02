"""比较冻结同输入阶段：有序面先验、轨迹、拓扑、翻折和自相交。"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,pathlib,sys
import nibabel.freesurfer.io as fsio
import numpy as np
from fnit.recon_all.compare_subject import _topology
from fnit.recon_all.mris_remesh_python import _footer_offset
from fnit.recon_all.mris_remove_intersection_python import mark_intersections
from fnit.recon_all.sphere_standard_unfold import _face_geometry


def clean_updates(updates):
    return [{k:v for k,v in row.items() if k!='seconds'} for row in updates]


def compare(a,b,stage,hemi,quality,link_checker=None):
    ra,rb=(json.loads((p/'report.json').read_text()) for p in (a,b))
    if ra['status']!='complete' or rb['status']!='complete':raise ValueError('both runs must complete')
    av,af=fsio.read_geometry(str(a/(hemi+'.'+stage)));bv,bf=fsio.read_geometry(str(b/(hemi+'.'+stage)))
    same=len(av)==len(bv) and np.array_equal(af,bf)
    paths=[p/(hemi+'.'+stage) for p in (a,b)]
    footer_hashes=[hashlib.sha256(p.read_bytes()[_footer_offset(p):]).hexdigest() for p in paths]
    ma=json.loads((a/'monitor/monitor.json').read_text()) if (a/'monitor/monitor.json').exists() else None
    mb=json.loads((b/'monitor/monitor.json').read_text()) if (b/'monitor/monitor.json').exists() else None
    if ma:ra['command_wall_seconds']=ma['command_wall_seconds'];ra['gpu_memory']=ma
    if mb:rb['command_wall_seconds']=mb['command_wall_seconds'];rb['gpu_memory']=mb
    row={'stage':stage,'hemisphere':hemi,'baseline_commit':ra['args']['commit'],'candidate_commit':rb['args']['commit'],'same_input_sha256':ra['input_sha256']==rb['input_sha256'],'ordered_faces_equal':bool(same),'baseline_seconds':ra['command_wall_seconds'],'candidate_seconds':rb['command_wall_seconds'],'speedup':ra['command_wall_seconds']/rb['command_wall_seconds'],'baseline_gpu':ra['gpu_memory'],'candidate_gpu':rb['gpu_memory'],'coordinate_array_equal':bool(same and np.array_equal(av,bv)),'coordinate_dtype_read':str(bv.dtype),'coordinate_storage':'FreeSurfer float32','space':'surface RAS','unit':'mm','overall_equivalence':'not_assessed'}
    row['geometry_footer_sha256']=footer_hashes
    row['geometry_footer_equal']=footer_hashes[0]==footer_hashes[1]
    row['coordinate_float32_bits_equal']=bool(same and av.astype('>f4').tobytes()==bv.astype('>f4').tobytes())
    if same:
        delta=np.linalg.norm(av-bv,axis=1);row['vertex_displacement_mm']={'max':float(delta.max(initial=0)),'p99':float(np.percentile(delta,99)),'different_vertices':int(np.count_nonzero(delta))}
    else:
        row['point_to_triangle_distance']='not_assessed; unexpected topology change blocks this optimization acceptance'
    row['trajectory_equal']=True
    if stage=='sphere':
        row['trajectory_equal']=clean_updates(ra['stage']['updates'])==clean_updates(rb['stage']['updates']) and ra['stage']['negative_counts']==rb['stage']['negative_counts']
    if stage=='register':
        row['trajectory_equal']=all(clean_updates(ra['stage'][name]['updates'])==clean_updates(rb['stage'][name]['updates']) for name in ('sulc_pass','smoothwm_pass'))
        row['rigid_search']={k:[ra['stage']['sulc_pass'][k],rb['stage']['sulc_pass'][k]] for k in ('rigid_angles','rigid_score','rigid_evaluations','rigid_seconds')}
        row['cleanup_counts_equal']=ra['stage']['smoothwm_pass']['negative_counts']==rb['stage']['smoothwm_pass']['negative_counts']
        row['stopping_equal']=ra['stage']['sulc_pass']['last_iteration']==rb['stage']['sulc_pass']['last_iteration'] and ra['stage']['smoothwm_pass']['seed_iteration']==rb['stage']['smoothwm_pass']['seed_iteration']
        row['rigid_search_exact']=all(ra['stage']['sulc_pass'][k]==rb['stage']['sulc_pass'][k] for k in ('rigid_angles','rigid_score','rigid_evaluations'))
        row['trajectory_equal']=row['trajectory_equal'] and row['cleanup_counts_equal'] and row['stopping_equal'] and row['rigid_search_exact']
    if quality:
        row['baseline_topology']=_topology(av,af);row['candidate_topology']=_topology(bv,bf)
        if link_checker is not None:
            row['candidate_vertex_links']=link_checker(bf,len(bv))
            row['baseline_vertex_links']=dict(row['candidate_vertex_links'],reuse_reason='identical ordered topology') if same else link_checker(af,len(av))
        # 逐元素相等时共享同一独立质量检查，明确输出可复用理由。
        targets=[('candidate',bv,bf)] if row['coordinate_array_equal'] else [('baseline',av,af),('candidate',bv,bf)]
        for name,v,f in targets:
            marks,nfaces=mark_intersections(v,f)
            row[name+'_self_intersection']={'marked_faces':nfaces,'marked_vertices':int(marks.sum()),'rule':'existing FNIT exact candidate broad phase; exclude vertex-sharing faces; plane tolerance 1e-6'}
            if stage!='remesh':
                area,_=_face_geometry(v.astype(np.float32),f.astype(np.int32));row[name+'_sphere_fold']={'negative_faces':int(np.count_nonzero(area<0)),'zero_area_faces':int(np.count_nonzero(area==0)),'negative_area_mm2':float(-area[area<0].sum(dtype=np.float64)),'radius_max_deviation_mm':float(np.max(np.abs(np.linalg.norm(v,axis=1)-100)))}
        if row['coordinate_array_equal']:
            row['baseline_self_intersection']=dict(row['candidate_self_intersection'],reuse_reason='identical vertices and ordered faces')
            if stage!='remesh':row['baseline_sphere_fold']=dict(row['candidate_sphere_fold'])
    row['strict_stage_reproduction']=row['same_input_sha256'] and row['coordinate_array_equal'] and row['coordinate_float32_bits_equal'] and row['geometry_footer_equal'] and row['trajectory_equal']
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--quality',action='store_true');p.add_argument('--quality-helper',type=pathlib.Path);a=p.parse_args();rows=[]
    link_checker=None
    if a.quality:
        path=a.quality_helper or pathlib.Path(__file__).with_name('surface_quality_helper.py')
        if not path.exists():path=pathlib.Path(__file__).resolve().parents[3]/'python_gpu_port/benchmark_surface_quality_extended.py'
        spec=importlib.util.spec_from_file_location('_fnit_quality_helpers',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);link_checker=module.vertex_links
    for base in sorted(a.pairs.glob('*/*/*/baseline/report.json')):
        folder=base.parent.parent;candidate=folder/'candidate';subject,hemi,stage=folder.relative_to(a.pairs).parts
        if not (candidate/'report.json').exists():continue
        row=compare(folder/'baseline',candidate,stage,hemi,a.quality,link_checker);row['subject']=subject;rows.append(row)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'comparison.json').write_text(json.dumps({'rows':rows,'strict_138':'coordinator final whole run retains existing diagnostic; not rerun by this stage script','regional_thickness_area_volume':'requires downstream whole outputs; coordinator acceptance','overall_equivalence':'not_assessed'},indent=2)+'\n')
    columns=['subject','hemisphere','stage','baseline_seconds','candidate_seconds','speedup','same_input_sha256','ordered_faces_equal','coordinate_array_equal','trajectory_equal','strict_stage_reproduction']
    with (a.output/'timings.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(json.dumps([{k:r[k] for k in columns} for r in rows],indent=2))

if __name__=='__main__':main()
