"""真实 pial 首轮首试步：完整有序接受回归与实际碰撞面对 GPU 回放。

只读取带哈希的冻结自产输入，不是完整 pial 或整例。捕获的真实面对和
接受状态保存在运行目录；报告不包含影像、许可证或原始三角对数据。
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import platform
import statistics
import time
from pathlib import Path
import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
import torch


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject',type=Path,required=True)
    parser.add_argument('--hemisphere',choices=('lh','rh'),default='lh')
    parser.add_argument('--candidate-directory',type=Path,required=True)
    parser.add_argument('--dependency-directory',type=Path,help='只读冻结自有依赖目录，不复制或修改')
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--code-commit',required=True)
    parser.add_argument('--capture',action='store_true',help='保存真实有序状态中完整候选面对，另计捕获耗时')
    args=parser.parse_args()
    import fnit.recon_all
    if args.dependency_directory:fnit.recon_all.__path__.insert(0,str(args.dependency_directory))
    fnit.recon_all.__path__.insert(0,str(args.candidate_directory))
    from fnit.recon_all import place_surface_collision as collision
    from fnit.recon_all.place_surface_collision_torch import triangle_pairs_intersect_torch
    from fnit.recon_all.place_surface_candidates_torch import conservative_face_candidates_torch
    from fnit.recon_all.place_surface_snapshot import snapshot_ordered_step
    from fnit.recon_all.place_surface_border import compute_border_values_first_pass
    from fnit.recon_all.place_surface_curvature import quadratic_curvature,tangent_basis,two_ring_neighbors
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_gradient_average import average_signed_gradients
    from fnit.recon_all.place_surface_intensity import intensity_gradient
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_repulsion import original_vertex_normals,surface_repulsion_gradient,vertex_buckets
    from fnit.recon_all.place_surface_rip import rip_outside_label
    from fnit.recon_all.place_surface_smoothing import average_marked_values,_ordered_neighbors
    from fnit.recon_all.place_surface_spring import spring_gradient
    from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets
    from fnit.recon_all.place_surface_volume import prepare_placement_volume
    output=args.output_directory;output.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    if device.type!='cuda' or device.index is None:raise ValueError('explicit CUDA target required')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    report=dict(scope='frozen_real_pial_first_pass_first_trial_ordered_acceptance',
        code_commit=args.code_commit,hostname=platform.node(),threads=args.threads,
        device=str(device),torch=torch.__version__,numpy=np.__version__,
        tf32_matmul=True,tf32_cudnn=True,autocast=False,
        precision_exception='float32 mesh and explicit float64 source collision arithmetic',
        reference='current Python/Numba same-input ordered collision',
        official_reference='not a complete stage; separate frozen-input complete pial report',
        tolerance_before_run=dict(coordinates_max_mm=0,accepted_offsets_max=0,decision_differences=0),
        default_admission='opt-in; complete pial and end-to-end not assessed')
    def persist(status):
        report['execution_status']=status
        temporary=output/'report.json.tmp'
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        temporary.replace(output/'report.json')
        print(json.dumps(dict(status=status,hostname=platform.node()),ensure_ascii=False),flush=True)
    source_functions=(collision.asynchronous_first_step,triangle_pairs_intersect_torch,
        conservative_face_candidates_torch,snapshot_ordered_step,quadratic_curvature,
        intensity_gradient,average_signed_gradients,unconstrained_step_with_offsets)
    report['source_sha256']={Path(inspect.getfile(inspect.unwrap(getattr(f,'py_func',f)))).name:
        sha256(inspect.getfile(inspect.unwrap(getattr(f,'py_func',f)))) for f in source_functions}
    started=time.perf_counter();hemi=args.hemisphere
    paths={name:args.subject/path for name,path in dict(white=f'surf/{hemi}.white',
        label=f'label/{hemi}.cortex+hipamyg.label',brain='mri/brain.finalsurfs.mgz',
        wm='mri/wm.mgz',aseg='mri/aseg.presurf.mgz',stats=f'surf/autodet.gw.stats.{hemi}.dat').items()}
    report['input_sha256']={name:sha256(path) for name,path in paths.items()}
    xyz,faces,metadata=fs.read_geometry(str(paths['white']),read_metadata=True)
    xyz,faces=xyz.astype(np.float32),faces.astype(np.int32)
    ripped=np.asarray(rip_outside_label(len(xyz),fs.read_label(str(paths['label']))),dtype=np.bool_)
    brain=nib.load(str(paths['brain']));wm=np.asarray(nib.load(str(paths['wm'])).dataobj)
    aseg=np.asarray(nib.load(str(paths['aseg'])).dataobj)
    stats=dict(line.split()[:2] for line in paths['stats'].read_text().splitlines() if len(line.split())>=2)
    volume,bright=prepare_placement_volume(np.asarray(brain.dataobj),wm,surface='pial',mid_gray=float(stats['MID_GRAY']))
    placement=volume.copy();placement[bright==130]=0
    affine=surface_ras_to_voxel(brain.header,metadata)
    normals=initial_vertex_normals(xyz,faces)
    thresholds=np.array([float(stats[f'pial_{name}']) for name in ('inside_hi','border_hi','border_low','outside_low','outside_hi')])
    border=compute_border_values_first_pass(volume,aseg,xyz,normals,xyz,ripped,
        np.full(len(xyz),-1.,np.float32),affine,thresholds,hemisphere=hemi,surface='pial')
    values=average_marked_values(border[0],border[4],ripped,faces,5)
    neighbors,valid,_=_ordered_neighbors(faces,len(xyz));ordered=(neighbors,valid)
    curve_offsets,curve_ids=two_ring_neighbors(faces,len(xyz),ordered_neighbors=ordered)
    intensity=intensity_gradient(placement,xyz,normals,ripped,values,border[5],affine,
        brain.header.get_zooms()[:3],weight=.2,sigma_global=2.)
    rep_offsets,rep_ids=vertex_buckets(xyz,xyz,ripped)
    repulsion=surface_repulsion_gradient(xyz,normals,xyz,original_vertex_normals(xyz,faces),
        ripped,rep_offsets,rep_ids,weight=5.,cropped=np.zeros(len(xyz),np.int32))
    averaged=average_signed_gradients(np.float32(intensity+repulsion),faces,ripped,16,ordered_neighbors=ordered)
    normal=spring_gradient(xyz,normals,faces,ripped,weight=.3,direction='normal',ordered_neighbors=ordered)
    curvature=quadratic_curvature(xyz,normals,tangent_basis(normals),ripped,curve_offsets,curve_ids)
    tangent=spring_gradient(xyz,normals,faces,ripped,weight=.3,direction='tangent',ordered_neighbors=ordered)
    momentum=np.float32(np.float32(np.float32(averaged+normal)+np.float32(curvature[:,None]*normals))+tangent)
    proposal,displacement=unconstrained_step_with_offsets(xyz,momentum,ripped,dt=.5,max_mm=.3)
    report['prepare_seconds']=time.perf_counter()-started
    state=output/'first_trial_input.npz'
    np.savez(state,vertices=xyz,faces=faces,proposal=proposal,ripped=ripped,
        momentum=momentum,offsets=displacement,neighbors=neighbors,valid=valid)
    report['checkpoint_sha256']={state.name:sha256(state)}
    persist('prepared')
    def run(backend):
        accepted=momentum.copy();diagnostic={}
        torch.cuda.synchronize(device)
        report.setdefault('gpu',torch.cuda.get_device_name(device))
        tick=time.perf_counter()
        result,order=collision.asynchronous_first_step(xyz,faces,proposal,ripped,
            offsets=displacement,accepted_offsets=accepted,ordered_neighbors=ordered,
            candidate_backend=backend,candidate_device=str(device) if backend=='torch_snapshot' else None,
            candidate_diagnostics=diagnostic)
        torch.cuda.synchronize(device)
        return result,order,accepted,time.perf_counter()-tick,diagnostic
    # 第一次包括JIT；后续ABBA均完整建索引、复制输入、接受和结果回传。
    baseline=run('tree');candidate=run('torch_snapshot')
    report['cold_seconds']=dict(tree=baseline[3],torch_snapshot=candidate[3])
    delta=np.abs(candidate[0].astype(np.float64)-baseline[0].astype(np.float64))
    report['comparison']=dict(different_coordinate_elements=int(np.count_nonzero(delta)),
        max_coordinate_mm=float(delta.max()),p99_coordinate_mm=float(np.percentile(delta,99)),
        different_accepted_offsets=int(np.count_nonzero(candidate[2]!=baseline[2])),
        order_identical=bool(np.array_equal(candidate[1],baseline[1])))
    reference_path=output/'first_trial_reference.npz'
    np.savez(reference_path,coordinates=baseline[0],order=baseline[1],accepted_offsets=baseline[2])
    report['checkpoint_sha256'][reference_path.name]=sha256(reference_path)
    persist('cold_pair_complete')
    torch.cuda.reset_peak_memory_stats(device);rows=[]
    for backend in ('tree','torch_snapshot','torch_snapshot','tree'):
        result=run(backend)
        rows.append(dict(backend=backend,seconds=result[3],diagnostics=result[4],
            coordinate_sha256=hashlib.sha256(result[0].tobytes()).hexdigest(),
            accepted_offsets_sha256=hashlib.sha256(result[2].tobytes()).hexdigest()))
    report['paired']=rows
    medians={backend:statistics.median(row['seconds'] for row in rows if row['backend']==backend)
             for backend in ('tree','torch_snapshot')}
    report['median_seconds']=medians;report['speed_ratio_tree_over_torch_snapshot']=medians['tree']/medians['torch_snapshot']
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved(device)
    persist('paired_complete')
    if args.capture:
        original=collision._vertex_collision_batch;chunks=[];pair_a=[];pair_b=[];pair_vertex=[];pair_count=0
        references=np.zeros(len(xyz),np.bool_);called=np.zeros(len(xyz),np.bool_)
        def flush():
            nonlocal pair_count
            if not pair_count:return
            path=output/f'live_pairs_{len(chunks):05d}.npz'
            np.savez(path,first=np.concatenate(pair_a),second=np.concatenate(pair_b),vertex=np.concatenate(pair_vertex))
            chunks.append(path);pair_a.clear();pair_b.clear();pair_vertex.clear();pair_count=0
            report['capture_checkpoint']=dict(chunks=len(chunks),last_chunk=path.name,last_sha256=sha256(path))
            persist('capture_in_progress')
        def capture(current,triangles,face_ids,vertex,endpoint,tree,maximum_radius):
            nonlocal pair_count
            moved,centers,radii,lows,highs=collision._incident_face_geometry(current,triangles,face_ids,vertex,endpoint)
            lists=tree.query_ball_point(centers,(radii+maximum_radius)+1.,return_sorted=False)
            for row,nearby in enumerate(lists):
                ids=np.asarray(nearby,np.int32);corners=triangles[face_ids[row]]
                ids=ids[~(triangles[ids,:,None]==corners[None,None,:]).any(axis=(1,2))]
                points=current[triangles[ids]]
                keep=(points.max(1)>=lows[row]).all(1)&(points.min(1)<=highs[row]).all(1)
                points=points[keep]
                if len(points):
                    pair_a.append(np.broadcast_to(moved[row],points.shape).copy());pair_b.append(points.copy())
                    pair_vertex.append(np.full(len(points),vertex,np.int32));pair_count+=len(points)
            references[vertex]=original(current,triangles,face_ids,vertex,endpoint,tree,maximum_radius)
            called[vertex]=True
            if pair_count>=65536:flush()
            return bool(references[vertex])
        collision._vertex_collision_batch=capture
        tick=time.perf_counter()
        try:captured=run('tree');flush()
        finally:collision._vertex_collision_batch=original
        capture_seconds=time.perf_counter()-tick
        report['capture_seconds']=capture_seconds
        persist('capture_complete')
        hits=np.zeros(len(xyz),np.bool_);predicate_rows=[]
        for path in chunks:
            with np.load(path) as saved:
                torch.cuda.synchronize(device);tick=time.perf_counter()
                result,diagnostic=triangle_pairs_intersect_torch(saved['first'],saved['second'],device=str(device),source_recheck=True)
                result=result.cpu().numpy();torch.cuda.synchronize(device)
                seconds=time.perf_counter()-tick
                np.logical_or.at(hits,saved['vertex'],result)
                predicate_rows.append(dict(pairs=len(result),seconds=seconds,diagnostics=diagnostic,sha256=sha256(path)))
                report['replay_checkpoint']=dict(chunks_completed=len(predicate_rows),chunks_total=len(chunks))
                persist('replay_in_progress')
        report['live_predicate_replay']=dict(capture_seconds=capture_seconds,called_vertices=int(called.sum()),
            pairs=sum(row['pairs'] for row in predicate_rows),source_collision_vertices=int(references.sum()),
            difference_vertices=int(np.count_nonzero(hits[called]!=references[called])),chunks=predicate_rows,
            capture_coordinates_identical=bool(np.array_equal(captured[0],baseline[0])),
            capture_acceptance_identical=bool(np.array_equal(captured[2],baseline[2])),
            meaning='GPU replay of complete eligible triangle pairs from actual sequential states; not live GPU acceptance')
        reference_bool=output/'live_pair_reference.npz';np.savez(reference_bool,called=called,collision=references)
        report['checkpoint_sha256'][reference_bool.name]=sha256(reference_bool)
    report['whole_process_gpu_memory']='not measured; allocator only, same-input subprocess'
    report['allocator_peak_scope']='ABBA complete first trial, after cold compilation/context setup'
    persist('complete')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
