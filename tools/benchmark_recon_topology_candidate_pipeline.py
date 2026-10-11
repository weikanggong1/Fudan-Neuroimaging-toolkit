"""唯一初态/明确冻结属性与先验→自产候选几何、局部MRI、SFF和完整评分。"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from benchmark_recon_topology_retess_fields import load_retessellation_capture
from benchmark_recon_topology_match_fields import load_match_capture
from benchmark_recon_topology_mri_likelihood import load_statistics_capture
from benchmark_recon_topology_unmri_fields import load_volume_capture
from benchmark_recon_topology_patch_fitness import difference,json_safe
from fnit.recon_all.topology_candidate_pipeline import evaluate_candidate
from fnit.recon_all.topology_defect_calibration import calibrated_defect_volume_priors
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
from fnit.recon_all.topology_mri_likelihood import joint_gray_white_pdf
from fnit.recon_all.topology_preflight_python import (project_and_smooth_sphere,center_sphere,
    defect_component_labels,defect_regions,genetic_base_translation)
from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.thread_budget import thread_budget


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*,before_retess,initial_attributes,initial_global_priors,input_surface,input_qsphere,
        brain,after_statistics_reference,after_volume_reference,fitness_reference,output,threads):
    """冻结retess before与MAT phase0属性/global prior→完整候选自产后比较。

    本工具不声称complete base/global-prior producer；MAT只提供本链初始
    全法向/K/E/area/curvbak/fx，初始global仅明确dot/K PDF，两个属性输入
    是声明的冻结准备状态。原nofix/qsphere/MRI独立生成缺陷/排列/白灰
    校准/localpriors及jointPDF，参考后态只有候选NPZ保存后才打开。
    输出完整数组/计时/哈希/严格诊断到新目录，不写生产；失败保留报告。
    """
    start=time.perf_counter()
    paths={k:Path(v) for k,v in dict(before_retess=before_retess,initial_attributes=initial_attributes,
        initial_global_priors=initial_global_priors,input_surface=input_surface,input_qsphere=input_qsphere,brain=brain,
        after_statistics_reference=after_statistics_reference,after_volume_reference=after_volume_reference,fitness_reference=fitness_reference).items()}
    output=Path(output)
    if output.exists():raise FileExistsError('fresh bounded candidate pipeline required')
    if not isinstance(threads,int) or threads<1:raise ValueError('positive fixed thread budget required')
    b=load_retessellation_capture(capture_path=paths['before_retess'])
    a=load_match_capture(capture_path=paths['initial_attributes'])
    global_prior=load_statistics_capture(capture_path=paths['initial_global_priors'])
    if int(b['header'][-1])!=0 or int(a['header'][7])!=0:raise ValueError('explicit pre-retess and pre-match attribute phases required')
    raw,faces=fsio.read_geometry(str(input_surface));q,qfaces=fsio.read_geometry(str(input_qsphere))
    raw,faces=np.asarray(raw,np.float32),np.asarray(faces,np.int32)
    if raw.shape!=q.shape or not np.array_equal(faces,qfaces):raise ValueError('same ordered nofix/qsphere input required')
    image=nib.load(brain);volume=np.asarray(image.dataobj);matrix=source_topology_ras_to_voxel(volume_header=image.header)
    n,k=int(b['header'][0]),int(b['header'][3]);off,ov=b['neighbor_offsets'],b['overlap_offsets']
    args=dict(canonical=b['vertex_fields'][:,6:9],base_faces=b['faces'],neighbors=[b['neighbors'][off[v]:off[v+1]] for v in range(n)],
        candidate_edges=b['edge_fields'][:,:2],edge_used=b['edge_fields'][:,2],ordering=b['ordering'],
        inside_vertices=b['inside_fields'][:,0],inside_discarded=b['inside_fields'][:,1]==1,
        border_vertices=b['border_vertices'],hull_vertices=b['hull_vertices'],vertex_rip=b['vertex_flags'][:,1]!=0,
        fixed_mapping=b['vertex_flags'][:,2]==0 if int(b['header'][8]) else None,
        mode='all' if int(b['header'][7]) else 'some',overlaps=[b['overlaps'][ov[e]:ov[e+1]] for e in range(k)] if int(b['header'][9]) else None,
        overlap_complete=b['edge_fields'][:,3]!=0 if int(b['header'][9]) else None)
    v=a['vertex_fields']
    # Only unchanged initial attributes are consumed. New faces, selected rows,
    # matched XYZ, k1/k2 after SFF and after-fitness volume are not inputs.
    initial=dict(original_vertices=b['vertex_fields'][:,:3],vertex_normals=v[:,9:12],tangent_e1=v[:,12:15],tangent_e2=v[:,15:18],
        flattened_vertices=v[:,6:9],k1=v[:,18],k2=v[:,19],vertex_area=v[:,20],curvbak=v[:,24],faces=b['faces'],face_area=np.zeros(len(b['faces']),np.float32))
    loading=time.perf_counter()-start;output.mkdir(parents=True)
    with thread_budget(threads=threads):
        tick=time.perf_counter()
        canonical,_=center_sphere(project_and_smooth_sphere(q,faces))
        labels=defect_component_labels(canonical,faces);regions=defect_regions(labels,faces)
        translation,_=genetic_base_translation(labels,faces);inside,border=regions[0]
        neighbors=ordered_neighbors(faces,len(raw));occupied=set(inside)|set(border);hull=list(border)
        for vertex in border:
            for other in neighbors[vertex]:
                if other not in occupied:occupied.add(other);hull.append(other)
        calibrated=calibrated_defect_volume_priors(vertices=raw,faces=faces,defect_vertices=labels!=0,
            defect_hull=np.asarray(hull,np.int32),candidate_edges=args['candidate_edges'],vertex_translation=translation,volume=volume,ras_to_voxel=matrix)
        prep=calibrated['preparation'];local=calibrated['priors']
        joint=joint_gray_white_pdf(white_values=prep['white_values'],gray_values=prep['gray_values'],excluded_vertices=labels!=0)
        preparation=time.perf_counter()-tick
        if not np.array_equal(prep['original_vertices'][np.argsort(translation)].view(np.uint32),initial['original_vertices'].view(np.uint32)):
            raise ValueError('raw independently prepared original coordinates do not match declared initial state')
        histogram=lambda name:dict(bins=global_prior[name+'_bins'],counts=global_prior[name+'_counts'],bin_size=global_prior[name+'_bin_size'][0])
        kw=dict(surface_state=initial,retessellation_inputs=args,volume=volume,ras_to_voxel=matrix,local_priors=local,gray_white_pdf=joint,
            k1_mean=float(a['parameters'][2]),k2_mean=float(a['parameters'][3]),normal_dot_histogram=histogram('dot'),
            k1_histogram=histogram('k1'),k2_histogram=histogram('k2'),joint_curvature_pdf=global_prior['k_pdf'],
            joint_origin_mm_inv=global_prior['k_pdf_geometry'][:2],joint_spacing_mm_inv=global_prior['k_pdf_geometry'][2:])
        results=[];candidates=[]
        for label in ['cold','repeat']:
            result=evaluate_candidate(**kw);fit=result['fitness'];geo=result['geometry'];components=fit['components']
            core=np.array([fit['composition']['tp_fields'][key] for key in ['face_ll','vertex_ll','curv_ll','qcurv_ll','unmri_ll']],np.float32)
            values=dict(matched=geo['original_vertices'],faces=geo['faces'],normals=result['vertex_normals'],
                k1=result['sff']['k1'],k2=result['sff']['k2'],e1=result['sff']['tangent_e1'],e2=result['sff']['tangent_e2'],
                core_fields=core,fitness=np.array(fit['fitness'],np.float64),distance=fit['distance']['distance_mm'],curvbak=result['vertex_curvbak'],
                sampling=result['flattened_vertices'],joint_pdf=joint,local_intensity=local['defect_intensity_volume'],
                local_white_log=local['white_log_probability'],local_gray_log=local['gray_log_probability'])
            tick=time.perf_counter();np.savez_compressed(output/(label+'.npz'),**values)
            results.append(dict(label=label,seconds=result['seconds'],write_seconds=time.perf_counter()-tick,
                face_check=result['face_check'],five_component_seconds=fit['seconds'],diagnostic_bits=dict(
                    fll=components['mri']['area_weighted_face_bits'],vll=components['mri']['area_weighted_vertex_bits'],
                    cll=components['normal']['cll_float32_bits'],qcll=components['curvature']['qcll_float32_bits'])))
            candidates.append(values)
    # Every own candidate artifact exists before any post-fitness reference is read.
    ref=load_statistics_capture(capture_path=paths['after_statistics_reference']);vol=load_volume_capture(capture_path=paths['after_volume_reference'])
    full=json.loads(paths['fitness_reference'].read_text());bits=np.asarray(full['tp_field_uint32_bits'],np.uint32)
    core_ref=bits[[0,1,2,3,5]].view(np.float32)
    checks=[]
    for candidate in candidates:
        checks.append(dict(core=difference(candidate['core_fields'],core_ref),fitness=difference(candidate['fitness'],np.array(full['fitness'],np.float64)),
            original=difference(candidate['matched'],ref['vertex_fields'][:,:3]),faces=difference(candidate['faces'],ref['faces']),
            normals=difference(candidate['normals'],ref['vertex_fields'][:,6:9]),k1=difference(candidate['k1'],ref['vertex_fields'][:,10]),
            k2=difference(candidate['k2'],ref['vertex_fields'][:,11]),distance=difference(candidate['distance'],vol['signed_distance']),
            # FNITVOL1: orig 0:3, normals 3:6, source fx/fy/fz 6:9.
            curvbak=difference(candidate['curvbak'],vol['reference_curvbak']),sampling=difference(candidate['sampling'],vol['vertex_fields'][:,6:9]),
            joint_pdf=difference(candidate['joint_pdf'],global_prior['gray_white_pdf']),local_intensity=difference(candidate['local_intensity'],vol['intensity'].astype(np.uint8)),
            local_white=difference(candidate['local_white_log'],vol['white_log_pdf']),local_gray=difference(candidate['local_gray_log'],vol['gray_log_pdf'])))
    repeat={key:difference(candidates[0][key],candidates[1][key]) for key in candidates[0]}
    passed=all(item.get('byte_equal') for check in checks for item in check.values()) and all(q.get('byte_equal') for q in repeat.values())
    code_root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(code_root)):sha(p) for p in (code_root/'src/fnit/recon_all').glob('topology_*.py')}
    sources[str(Path(__file__).relative_to(code_root))]=sha(__file__)
    report=dict(status='strict_bounded_candidate_pipeline_passed' if passed else 'strict_bounded_candidate_pipeline_failed',
        scope='frozen before attributes/global K-dot prior; raw nofix/MRI independently generates gray-white calibration/local fields; full continuous candidate only',
        initial_inputs_sha256={key:sha(paths[key]) for key in ['before_retess','initial_attributes','initial_global_priors','input_surface','input_qsphere','brain']},
        comparison_reference_sha256={key:sha(paths[key]) for key in ['after_statistics_reference','after_volume_reference','fitness_reference']},
        loading_seconds=loading,raw_calibration_seconds=preparation,raw_calibration_subtimers=calibrated['stage_seconds'],results=results,checks=checks,
        repeat=repeat,source_sha256=sources,threads=threads,source_calloc_face_area=True,
        frozen_initial_attribute_names=['normal','k1','k2','e1','e2','area','curvbak','fx'],frozen_global_priors=['dot_hist','k1_hist','k2_hist','joint_Kpdf','smooth_Kmean'],
        reference_after_consumed_to_generate=False,complete_base_producer=False,complete_ga=False,final_mesh_quality='not_assessed',overall_metric_equivalence='not_assessed',
        pipeline_wall_before_report_seconds=time.perf_counter()-start)
    (output/'report.private.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    if not passed:raise RuntimeError('strict bounded candidate gate failed; report retained')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['before-retess','initial-attributes','initial-global-priors','input-surface','input-qsphere','brain','after-statistics-reference','after-volume-reference','fitness-reference','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,required=True)
    print(json.dumps(json_safe(run(**vars(p.parse_args()))),indent=2,allow_nan=False))


if __name__=='__main__':main()
