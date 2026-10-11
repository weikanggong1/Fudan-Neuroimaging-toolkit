"""完整冻结有向距离→已有体积聚合的同输入回归；不运行GA。"""
from __future__ import annotations
import time
_START=time.perf_counter()
import argparse,hashlib,json,os,platform
from pathlib import Path
import numpy as np
from benchmark_recon_topology_distance_fields import load_distance_capture
from benchmark_recon_topology_unmri_fields import load_volume_capture
from benchmark_recon_topology_volume_likelihood import arguments as volume_arguments
from fnit.recon_all import topology_defect_signed_distance as implementation
from fnit.recon_all import topology_angle_pseudonormals as angle_implementation
from fnit.recon_all import topology_defect_volume_likelihood as volume_implementation
from fnit.recon_all.thread_budget import thread_budget
_IMPORT=time.perf_counter()-_START


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def difference(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:return dict(shape_equal=False,bitwise_equal=False)
    e=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(shape_equal=True,different_elements=int(np.count_nonzero(a!=b)),
        bitwise_equal=bool(a.dtype==b.dtype and a.tobytes()==b.tobytes()),
        max_abs=float(e.max(initial=0)),p99_abs=float(np.percentile(e,99)) if e.size else 0.)


def _profiled_evaluate(arguments):
    """只在隔离回归中计时成熟子项，恢复全部函数符号；算法输入不变。"""
    names=['_prepare_buffers','other_edge_faces','angle_weighted_vertex_pseudonormals','_distance']
    observed={};originals={name:getattr(implementation,name) for name in names}
    def wrap(name,function):
        def call(*args,**kwargs):
            tick=time.perf_counter()
            try:return function(*args,**kwargs)
            finally:observed[name]=observed.get(name,0.)+time.perf_counter()-tick
        return call
    for name,function in originals.items():setattr(implementation,name,wrap(name,function))
    start=time.perf_counter()
    try:result=implementation.defect_signed_distance(**arguments)
    finally:
        for name,function in originals.items():setattr(implementation,name,function)
    wall=time.perf_counter()-start
    observed['validation_copy_and_return']=max(0.,wall-sum(observed.values()))
    return result,wall,observed


def run(*,capture,native_repeat,volume_capture,observer_receipt,output,threads):
    """完整FNITDST1/2+聚合前FNITVOL1→独立自产距离和完整8字段/curvbak。

    capture/native_repeat是两个独立source输入/输出快照，仅供diagnostic。
    volume_capture提供预备MRI/PDF/feedback状态，distance参数改为自产结果，
    不使用参考距离。observer_receipt为独立构建绑定；output须新空路径。
    threads为总正预算，恢复父设置；不写生产目录。返回JSON和私有NPZ。
    不包含prepared MRI/PDF生成、完整GA或recon-all。严格门保留，不放宽。
    """
    if output.exists():raise FileExistsError('fresh diagnostic output required')
    if threads<1:raise ValueError('positive thread budget required')
    start=time.perf_counter();f=load_distance_capture(capture_path=capture);repeat=load_distance_capture(capture_path=native_repeat)
    v=load_volume_capture(capture_path=volume_capture)
    if 'algorithm_volume_geometry' in f:
        algorithm_geometry=f['algorithm_volume_geometry']
        algorithm_shape=tuple(int(x) for x in f['algorithm_volume_dimensions'][:3])
        if algorithm_shape!=v['intensity'].shape or algorithm_geometry.tobytes()!=v['intensity_geometry'].tobytes():
            raise ValueError('independent DST2 and same-candidate volume sampling geometry disagree')
        geometry_source='FNITDST2 actual mri_defect header, verified against same-candidate FNITVOL1'
    else:
        algorithm_geometry=v['intensity_geometry'];algorithm_shape=v['intensity'].shape
        geometry_source='same-candidate FNITVOL1 mri_defect intensity header; legacy DST1 lacks this input'
    args=dict(original_vertices=f['vertex_fields'][:,:3],faces=f['faces'],face_normals=f['face_normals'],
        incident_offsets=f['incident_offsets'],incident_faces=f['incident_faces'],incident_corners=f['incident_corners'],
        volume_shape=algorithm_shape,volume_origin_mm=algorithm_geometry[:3],volume_index_scale_per_mm=algorithm_geometry[3:6])
    with thread_budget(threads=threads):
        a,first,first_components=_profiled_evaluate(args)
        b,second,repeat_components=_profiled_evaluate(args)
        va=volume_arguments(v);va['signed_distance_mm']=a['distance_mm']
        tick=time.perf_counter();volume=volume_implementation.defect_volume_likelihood(**va);aggregation=time.perf_counter()-tick
    # Compare source buffer geometry by identity, retaining original source order hashes.
    own_order=np.argsort(a['buffer_bounds'][:,0]);ref_order=np.argsort(f['buffer_bounds'][:,0])
    own_query=np.argsort(a['query_vertices']);ref_query=np.argsort(f['query_vertices'])
    components=dict(distance=difference(a['distance_mm'],f['distance_after']),
        buffer_bounds_fno_aligned=difference(a['buffer_bounds'][own_order],f['buffer_bounds'][ref_order].astype(np.int64)),
        buffer_coordinates_fno_aligned=difference(a['buffer_coordinates'][own_order],f['buffer_coordinates'][ref_order]),
        other_edge_faces_fno_aligned=difference(a['other_edge_faces'][own_order],f['buffer_other_faces'][ref_order].astype(np.int64)),
        query_vertices_id_aligned=difference(a['query_vertices'][own_query],f['query_vertices'][ref_query].astype(np.int64)),
        query_pseudonormals_id_aligned=difference(a['query_pseudonormals'][own_query],f['query_pseudonormals'][ref_query]))
    native_comparisons={key:difference(value,repeat[key]) for key,value in f.items()}
    self_repeat={key:difference(value,b[key]) for key,value in a.items() if isinstance(value,np.ndarray)}
    names=['white_ll','gray_ll','white_mean_intensity','gray_mean_intensity','max_distance_mm','unmri_ll','nwhite','ngray']
    scores=np.asarray([volume[name] for name in names],np.float32)
    volume_comparisons=dict(reference_distance_same_current_candidate=difference(f['distance_after'],v['signed_distance']),
        eight_source_fields=difference(scores,v['reference_results']),vertex_curvbak=difference(volume['vertex_curvbak'],v['reference_curvbak']))
    output.mkdir(parents=True)
    np.savez_compressed(output/'candidate.private.npz',**{key:value for key,value in a.items() if isinstance(value,np.ndarray)},source_fields=scores,vertex_curvbak=volume['vertex_curvbak'])
    report=dict(status='complete_frozen_signed_distance_regression',
        strict_complete_distance_passed=all(row.get('bitwise_equal',False) for row in components.values()),
        strict_downstream_aggregation_passed=all(row.get('bitwise_equal',False) for row in volume_comparisons.values()),
        candidate_repeat_exact=all(row.get('bitwise_equal',False) for row in self_repeat.values()),
        native_distance_repeat_exact=native_comparisons['distance_after']['bitwise_equal'],
        native_full_capture_repeat_exact=sha(capture)==sha(native_repeat),
        native_buffer_order_repeat_exact=bool(f['buffer_bounds'].tobytes()==repeat['buffer_bounds'].tobytes()),
        reference_buffer_order_not_replaced=True,algorithm_volume_geometry=algorithm_geometry.tolist(),
        captured_distance_storage_header_not_sampling_geometry=f['volume_geometry'].tolist(),
        correct_algorithm_geometry_source=geometry_source,production_candidate_selection='all faces plus computed source bbox; no reference face list',
        captured_public_distance_before_not_used='active source thread-local maps start NPY, not captured shared zeros',
        comparisons=components,downstream_comparisons=volume_comparisons,candidate_repeat=self_repeat,native_repeat=native_comparisons,
        candidate_scores=dict(zip(names,[float(x) for x in scores])),
        capture_sha256=sha(capture),native_repeat_capture_sha256=sha(native_repeat),volume_capture_sha256=sha(volume_capture),
        native_build_sha256=sha(observer_receipt),candidate_npz_sha256=sha(output/'candidate.private.npz'),
        module_sha256=dict(implementation=sha(implementation.__file__),angle=sha(angle_implementation.__file__),volume=sha(volume_implementation.__file__),
            benchmark=sha(__file__),distance_parser=sha(__import__('benchmark_recon_topology_distance_fields').__file__)),
        input_fields={key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=array_sha(value)) for key,value in f.items()},
        input_vertex_count=len(f['vertex_fields']),input_face_count=len(f['faces']),buffer_faces=len(a['buffer_bounds']),query_vertices=len(a['query_vertices']),
        volume_shape=list(a['distance_mm'].shape),distance_unset_count=int(np.count_nonzero(a['distance_mm']==1000)),
        cpu_threads=threads,cpu_affinity=sorted(os.sched_getaffinity(0)),host=platform.node(),python=platform.python_version(),numpy=np.__version__,
        seconds=dict(imports=_IMPORT,first_complete_distance_api_available_jit_cache=first,repeat_complete_distance_api=second,first_component_observation=first_components,repeat_component_observation=repeat_components,downstream_aggregation_api=aggregation,
            read_validation_repeat_npz_io=time.perf_counter()-start),gpu_used=False,gpu_process_memory_bytes=None,native_distance_component_seconds=None,
        prepared_mri_pdf_generation='not implemented in this module',full_ga='not executed',whole_reconstruction='not executed',overall_equivalence='not_assessed')
    _plot(a['distance_mm'],f['distance_after'],output/'distance.png');report['figure_sha256']=sha(output/'distance.png')
    report['seconds']['tool_wall_including_imports_plot']=time.perf_counter()-_START
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def _plot(a,b,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    layer=a.shape[2]//2
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    av,bv=a[:,:,layer].copy(),b[:,:,layer].copy()
    error=av-bv
    absent=(av==1000)&(bv==1000)
    av[av==1000]=np.nan;bv[bv==1000]=np.nan;error[absent]=np.nan
    finite=np.concatenate([av[np.isfinite(av)],bv[np.isfinite(bv)]])
    scale=max(1.,float(np.percentile(np.abs(finite),99))) if finite.size else 1.
    for index,(ax,data,title) in enumerate(zip(axes,[av,bv,error],['FNIT signed distance (mm)','Source distance (mm)','FNIT minus source (mm)'])):
        image=ax.imshow(data.T,origin='lower',cmap='coolwarm',vmin=-scale if index<2 else None,vmax=scale if index<2 else None)
        ax.set_title(title);ax.set_xlabel('local x voxel');ax.set_ylabel('local y voxel');fig.colorbar(image,ax=ax,shrink=.8)
    fig.suptitle('Unprocessed NPY=1000 masked; source sampling geometry preserved')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['capture','native-repeat','volume-capture','observer-receipt','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=4);run(**vars(p.parse_args()))
