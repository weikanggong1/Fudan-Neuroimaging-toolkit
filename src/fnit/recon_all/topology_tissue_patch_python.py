"""独立组织代价局部patch实验：复用已有直方图与40轮MRI拟合。

该原语不进行GA边重连/切补，也不等于完整体素组织代价拓扑求解。
不使用被试坐标、Unknown标签或脑mask直接剪枝；无原版程序调用。
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .place_surface_geometry import surface_ras_to_voxel
from .place_surface_normals import initial_vertex_normals
from .smooth_surface_python import ordered_neighbors
from .topology_intensity_histogram import first_defect_intensity_histograms
from .topology_defect_mri_match import defect_mri_match, _sample
from .topology_preflight_python import topology_counts
from .topology_volume_python import _nonmanifold_vertex_links
from .mris_remove_intersection_python import mark_intersections, _write_preserving_surface
from .place_surface_collision import asynchronous_first_step


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@njit(cache=True, fastmath=False)
def _sample_sides(xyz,normals,volume,matrix,offset):
    white=np.empty(len(xyz),np.float32);gray=np.empty(len(xyz),np.float32)
    for vertex in range(len(xyz)):
        inner=xyz[vertex]-offset*normals[vertex];outer=xyz[vertex]+offset*normals[vertex]
        white[vertex]=np.float32(_sample(volume,matrix,inner[0],inner[1],inner[2]))
        gray[vertex]=np.float32(_sample(volume,matrix,outer[0],outer[1],outer[2]))
    return white,gray


@njit(cache=True, fastmath=False)
def _median_twice(values,neighbors,degrees):
    current=values.copy()
    for _ in range(2):
        result=np.empty_like(current)
        for vertex in range(len(current)):
            local=np.empty(degrees[vertex]+1,np.float32);local[0]=current[vertex]
            for slot in range(degrees[vertex]): local[slot+1]=current[neighbors[vertex,slot]]
            result[vertex]=np.float32(np.median(local))
        current=result
    return current


def _score(white,gray,neighbors,log_shape_prior):
    degrees=np.asarray([len(row) for row in neighbors],np.int32)
    index=np.zeros((len(neighbors),int(degrees.max(initial=0))),np.int32)
    for vertex,row in enumerate(neighbors): index[vertex,:len(row)]=row
    white_median=_median_twice(white,index,degrees);gray_median=_median_twice(gray,index,degrees)
    hist=first_defect_intensity_histograms(white=white_median,gray=gray_median,
        hull=list(range(len(neighbors))),neighbors=neighbors,marked=np.zeros(len(neighbors),bool))
    bins=np.clip(np.floor(white_median.astype(np.float64)+.5),0,255).astype(np.int64)
    white_prob=np.maximum(hist['white_smooth'][bins].astype(np.float64),1e-30)
    gray_prob=np.maximum(hist['gray_smooth'][bins].astype(np.float64),1e-30)
    score=np.log(gray_prob/white_prob)-log_shape_prior
    return score,hist


def _summary(values):
    values=np.asarray(values,np.float64)
    return {'mean':float(values.mean()) if len(values) else 0.0,
            'p99':float(np.percentile(values,99)) if len(values) else 0.0,
            'maximum':float(values.max()) if len(values) else 0.0,
            'minimum':float(values.min()) if len(values) else 0.0}


def _face_cross(vertices,faces):
    points=np.asarray(vertices,np.float64)[faces]
    return np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])


def _ordered_collision_acceptor(faces,selected,trace,motion_limit_mm, *, active_cache=None):
    """复用完整顺序接受器；每轮面反转时二分回退，最多12次。"""
    ripped=~np.asarray(selected,bool)

    def accept(current,proposal,iteration):
        started=time.perf_counter()
        delta=proposal.astype(np.float64)-current
        distances=np.linalg.norm(delta,axis=1)
        clipped=distances>motion_limit_mm
        delta[clipped]*=(motion_limit_mm/distances[clipped])[:,None]
        before_cross=_face_cross(current,faces)
        attempts=[]
        accepted=current.copy()
        for halving in range(13):
            trial=np.float32(current+delta*(.5**halving))
            diagnostics={}
            if active_cache is None:
                candidate,order=asynchronous_first_step(
                    vertices=current,faces=faces,proposed=trial,ripped=ripped,
                    candidate_backend='snapshot',candidate_diagnostics=diagnostics)
            else:
                candidate,order,diagnostics=active_cache.step(current=current,proposal=trial,faces=faces)
            after_cross=_face_cross(candidate,faces)
            orientation=np.einsum('ij,ij->i',before_cross,after_cross)
            norms=np.linalg.norm(after_cross,axis=1)
            invalid=int(np.count_nonzero((orientation<=0)|(norms<=0)))
            attempts.append({'halving':halving,'scale':.5**halving,
                'invalid_or_zero_faces':invalid,'ordered_vertices':len(order),
                'candidate_diagnostics':diagnostics})
            if invalid==0 and np.isfinite(candidate).all():
                accepted=candidate
                break
        change=np.linalg.norm(accepted.astype(np.float64)-current,axis=1)
        trace.append({'iteration':iteration,'motion_limit_mm':motion_limit_mm,
            'clipped_vertices':int(clipped.sum()),'proposal_displacement_mm':_summary(distances),
            'accepted_displacement_mm':_summary(change),'accepted_changed_vertices':int(np.count_nonzero(change)),
            'attempts':attempts,'accepted_coordinates_sha256':hashlib.sha256(accepted.astype('<f4').tobytes()).hexdigest(),
            'proposal_coordinates_sha256':hashlib.sha256(proposal.astype('<f4').tobytes()).hexdigest(),
            'acceptance_wall_seconds':time.perf_counter()-started})
        return accepted

    return accept


def run_tissue_patch(*, input_surface: str | Path, intensity_volume: str | Path,
                     output_surface: str | Path, log_shape_prior: float = 0.0,
                     normal_offset_mm: float = 0.5, require_quality: bool = True,
                     report_path: str | Path | None = None,
                     collision_acceptance: str = 'ordered',
                     step_motion_limit_mm: float = 0.3) -> dict:
    """完整局部MRI拟合原语，返回自产候选与完整QC，非完整GA或生产默认。

    input_surface为已球面拓扑三角面surface RAS/mm，intensity_volume为
    同空间uint8 brain.mgz/MGH；output_surface新路径，保留有序面/尾部。
    log_shape_prior默认0，是MRI灰质/白质似然比选择的保留先验；offset
    默认0.5mm，须正且有限。require_quality默认True，失败报告保留但
    不写成功表面；False仅写独立失败诊断候选，不可生产使用。
    复用有序法向、完整三线性采样和直方图核；两次包含自身与一环的
    中值为本实验配方，不声称原GA采样流程逐位重现。选择log(p_gray/
    p_white)>prior的全部顶点，不按分区、mask、长度或坐标筛选。仅这些
    顶点复用已有完整40步defect_mri_match，其余固定。返回输入、模块、
    输出SHA、似然、组件、位移、顶点/边流形/闭合/Euler/自相交/面积QC
    和包含I/O/JIT墙钟。拓扑面序不变，无边重连、遗传搜索或体素删填。
    collision_acceptance默认ordered：复用完整顺序动态碰撞接受；每轮
    提案单顶点位移限制为step_motion_limit_mm（默认0.3mm，须≤0.3），
    发现面反转或零面积时整轮二分回退，最多12次，失败保持该轮输入。
    none仅复现未受碰撞控制的40轮原语，用于失败诊断。两种都执行
    独立最终结构QC，不能用局部接受记录代替最终相交和形状评价。
    """
    started=time.perf_counter()
    input_surface,intensity_volume,output_surface=map(Path,(input_surface,intensity_volume,output_surface))
    if len({p.resolve() for p in (input_surface,intensity_volume,output_surface)})!=3:
        raise ValueError('input/output paths must differ')
    if not np.isfinite(log_shape_prior) or not np.isfinite(normal_offset_mm) or normal_offset_mm<=0:
        raise ValueError('finite prior and positive finite offset required')
    if collision_acceptance not in {'ordered','ordered-active','none'}:
        raise ValueError('collision_acceptance must be ordered, ordered-active or none')
    if not np.isfinite(step_motion_limit_mm) or not 0<step_motion_limit_mm<=.3:
        raise ValueError('positive step_motion_limit_mm <= 0.3 required')
    if not isinstance(require_quality,bool): raise ValueError('require_quality must be bool')
    if report_path is not None and Path(report_path).resolve() in {p.resolve() for p in (input_surface,intensity_volume,output_surface)}:
        raise ValueError('report path must differ from input/output paths')
    xyz,faces,metadata=fsio.read_geometry(str(input_surface),read_metadata=True)
    if not metadata or not np.isfinite(xyz).all(): raise ValueError('finite surface with volume geometry required')
    xyz=np.ascontiguousarray(xyz,np.float32);faces=np.ascontiguousarray(faces,np.int32)
    image=nib.load(str(intensity_volume));volume=np.asarray(image.dataobj)
    if volume.ndim!=3 or volume.dtype!=np.uint8 or not hasattr(image.header,'get_vox2ras_tkr'):
        raise ValueError('requires 3D uint8 MGH/MGZ normalized MRI')
    matrix=np.asarray(surface_ras_to_voxel(image.header,metadata),np.float64)
    neighbors=ordered_neighbors(faces,len(xyz));normals=initial_vertex_normals(vertices=xyz,triangles=faces)
    white,gray=_sample_sides(xyz,normals,np.ascontiguousarray(volume),matrix,np.float32(normal_offset_mm))
    score,hist=_score(white,gray,neighbors,float(log_shape_prior));inside=np.flatnonzero(score>0).astype(np.int32)
    setup_done=time.perf_counter()
    trace=[]
    mask=np.zeros(len(xyz),bool);mask[inside]=True
    active_cache=None
    if collision_acceptance=='ordered-active':
        from .topology_collision_cache_python import FixedPatchCollisionCache
        active_cache=FixedPatchCollisionCache(faces=faces,selected=mask)
    if len(inside):
        corrected=defect_mri_match(xyz=xyz,faces=faces,inside_vertices=inside,
            runtime_neighbors=[neighbors[int(v)] for v in inside],volume=volume,ras_to_voxel=matrix,
            white_mean=float(hist['white_mean']),gray_mean=float(hist['gray_mean']),
            step_acceptor=_ordered_collision_acceptor(faces,mask,trace,float(step_motion_limit_mm),active_cache=active_cache) if collision_acceptance!='none' else None)
    else: corrected=xyz.copy()
    fit_done=time.perf_counter()
    final_normals=initial_vertex_normals(vertices=corrected,triangles=faces)
    final_white,final_gray=_sample_sides(corrected,final_normals,np.ascontiguousarray(volume),matrix,np.float32(normal_offset_mm))
    counts=topology_counts(faces,len(xyz));nonmanifold=int(_nonmanifold_vertex_links(faces,len(xyz)))
    _,intersections=mark_intersections(corrected,faces)
    area=np.linalg.norm(np.cross(corrected[faces[:,1]]-corrected[faces[:,0]],corrected[faces[:,2]]-corrected[faces[:,0]]),axis=1)/2
    all_edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    full_graph=coo_matrix((np.ones(len(all_edges)*2),(np.r_[all_edges[:,0],all_edges[:,1]],np.r_[all_edges[:,1],all_edges[:,0]])),shape=(len(xyz),len(xyz))).tocsr()
    full_components=int(connected_components(full_graph,directed=False)[0])
    signs=np.where(all_edges[:,0]<all_edges[:,1],1,-1)
    _,edge_inverse=np.unique(np.sort(all_edges,axis=1),axis=0,return_inverse=True)
    orientation_conflicts=int(np.count_nonzero(np.bincount(edge_inverse,weights=signs)))
    quality=(counts.euler==2 and full_components==1 and counts.boundary_edges==0 and counts.nonmanifold_edges==0 and nonmanifold==0 and orientation_conflicts==0 and intersections==0 and np.all(area>0) and np.isfinite(corrected).all())
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);edges=edges[mask[edges].all(axis=1)]
    mapping=np.full(len(xyz),-1,np.int64);mapping[inside]=np.arange(len(inside))
    if len(inside):
        graph=coo_matrix((np.ones(len(edges)*2),(np.r_[mapping[edges[:,0]],mapping[edges[:,1]]],np.r_[mapping[edges[:,1]],mapping[edges[:,0]]])),shape=(len(inside),len(inside))).tocsr()
        components,labels=connected_components(graph,directed=False);sizes=np.bincount(labels).tolist()
    else: components,sizes=0,[]
    displacement=np.linalg.norm(corrected.astype(np.float64)-xyz,axis=1)
    report={'scope':'MRI tissue-cost local patch primitive, not complete topology or GA solver','input_surface':str(input_surface),'intensity_volume':str(intensity_volume),
        'input_sha256':{'surface':_sha(input_surface),'MRI':_sha(intensity_volume)},'source_sha256':_sha(__file__),
        'reused_module_sha256':{name:_sha(Path(__file__).with_name(name+'.py')) for name in ['topology_intensity_histogram','topology_defect_mri_match','place_surface_normals','topology_volume_python','mris_remove_intersection_python','place_surface_collision','place_surface_snapshot','topology_collision_cache_python']},
        'output_surface':str(output_surface),'native_program_called':False,'device':'cpu','precision':'float32 geometry; source MRI uint8; float64 likelihood ratio',
        'log_shape_prior':float(log_shape_prior),'normal_offset_mm':float(normal_offset_mm),'patch_iterations':40 if len(inside) else 0,
        'collision_acceptance':collision_acceptance,'step_motion_limit_mm':float(step_motion_limit_mm),'ordered_step_trace':trace,
        'selected_vertices':len(inside),'selected_vertex_indices_sha256':hashlib.sha256(inside.astype('<i4').tobytes()).hexdigest(),'selected_components':int(components),'selected_component_sizes':sizes,
        'white_histogram_mean':float(hist['white_mean']),'gray_histogram_mean':float(hist['gray_mean']),
        'selection_score':_summary(score),'white_sample_before':_summary(white),'gray_sample_before':_summary(gray),
        'white_sample_after':_summary(final_white),'gray_sample_after':_summary(final_gray),
        'displacement_mm':_summary(displacement),'fixed_unselected_coordinates_equal':bool(np.array_equal(corrected[~mask],xyz[~mask])),
        'mesh':asdict(counts),'mesh_components':full_components,'nonmanifold_vertices':nonmanifold,'edge_orientation_conflicts':orientation_conflicts,'geometric_intersecting_faces':int(intersections),'zero_area_faces':int(np.count_nonzero(area==0)),
        'mesh_quality_status':'passed' if quality else 'failed','output_written':False,'execution_status':'complete_primitive' if quality else 'failed_quality',
        'setup_sampling_histogram_seconds':setup_done-started,'complete_40_step_fit_seconds':fit_done-setup_done,
        'quality_recheck_seconds':time.perf_counter()-fit_done,'overall_metric_equivalence':'not_assessed',
        'complete_dynamic_topology_solver':False}
    if quality or not require_quality:
        output_surface.parent.mkdir(parents=True,exist_ok=True);_write_preserving_surface(input_surface,output_surface,corrected)
        report.update(output_written=True,output_sha256=_sha(output_surface))
    report['total_seconds_including_IO_JIT_quality']=time.perf_counter()-started
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True,exist_ok=True);Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
    if require_quality and not quality: raise RuntimeError('tissue patch primitive failed structural QC; no production output written')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-surface',type=Path,required=True);parser.add_argument('--intensity-volume',type=Path,required=True)
    parser.add_argument('--output-surface',type=Path,required=True);parser.add_argument('--report-path',type=Path,required=True)
    parser.add_argument('--log-shape-prior',type=float,default=0.0);parser.add_argument('--normal-offset-mm',type=float,default=.5)
    parser.add_argument('--collision-acceptance',choices=('ordered','ordered-active','none'),default='ordered')
    parser.add_argument('--step-motion-limit-mm',type=float,default=.3)
    parser.add_argument('--allow-failed-diagnostic-output',action='store_true')
    args=parser.parse_args()
    run_tissue_patch(input_surface=args.input_surface,intensity_volume=args.intensity_volume,output_surface=args.output_surface,
        log_shape_prior=args.log_shape_prior,normal_offset_mm=args.normal_offset_mm,require_quality=not args.allow_failed_diagnostic_output,report_path=args.report_path,
        collision_acceptance=args.collision_acceptance,step_motion_limit_mm=args.step_motion_limit_mm)


if __name__=='__main__': main()
