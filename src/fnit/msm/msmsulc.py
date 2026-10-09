"""MSMSulc registration with independent HOCR reduction and FastPD fusion."""
from pathlib import Path
import json
import time
from functools import lru_cache

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from ._affine import _surface, _affine_initialization
from .config import MSMSulcConfig
from ._sphere_map import RadialSphereMap, _area_weights, _cross, _dot, _normalize
from .prepare import MSMSulcInputs
from ._execution import register_hemispheres, execution_report, record_statistics, current_statistics

def _ico(level,*,cached_area=False):
    a,b=0.8506508084,0.5257311121
    vertices=np.array([(a,b,0),(-a,b,0),(-a,-b,0),(a,-b,0),
                       (b,0,a),(b,0,-a),(-b,0,-a),(-b,0,a),
                       (0,a,b),(0,-a,b),(0,-a,-b),(0,a,-b)],np.float64)
    faces=np.array([(7,8,4),(9,7,4),(11,6,5),(6,10,5),
                    (3,4,0),(5,3,0),(1,7,2),(6,1,2),
                    (11,0,8),(1,11,8),(3,10,9),(10,2,9),
                    (0,4,8),(5,0,11),(3,9,4),(10,3,5),
                    (1,8,7),(11,1,6),(9,2,7),(2,10,6)],np.int64)[:,[0,2,1]]
    area=_vertex_area(vertices,faces) if cached_area and level==0 else None
    for step in range(level):
        points=vertices.tolist();edges={};next_faces=[]
        def midpoint(i,j):
            edge=(min(i,j),max(i,j))
            if edge not in edges:
                edges[edge]=len(points)
                points.append(((vertices[i]+vertices[j])/2).tolist())
            return edges[edge]
        for v0,v1,v2 in faces:
            p0=midpoint(v1,v2);p1=midpoint(v0,v2);p2=midpoint(v0,v1)
            next_faces.extend(((p2,p0,p1),(p1,v0,p2),(p0,v2,p1),(p2,v1,p0)))
        vertices=np.asarray(points,np.float64)
        faces=np.asarray(next_faces,np.int64)
        # Official Triangle caches its constructor area. retessellate builds
        # all four planar subtriangles before normalizing their midpoint
        # vertices; true_rescale later changes coordinates, not cached areas.
        if cached_area and step==level-1:area=_vertex_area(vertices,faces)
        vertices/=np.linalg.norm(vertices,axis=1,keepdims=True)
    # make_mesh_from_icosa is followed by true_rescale at every level.
    vertices/=np.linalg.norm(vertices,axis=1,keepdims=True)
    return (vertices*100,faces,area) if cached_area else (vertices*100,faces)


def _vertex_area(vertices,faces):
    triangles=vertices[faces]
    cross=np.cross(triangles[:,2]-triangles[:,0],triangles[:,1]-triangles[:,0])
    squared=(cross[:,0]*cross[:,0]+cross[:,1]*cross[:,1])+cross[:,2]*cross[:,2]
    area=np.sqrt(squared)*0.5
    total=np.bincount(faces.ravel(),weights=np.repeat(area,3),minlength=len(vertices))
    counts=np.bincount(faces.ravel(),minlength=len(vertices))
    return total/counts


def _adaptive_resample(vertices,faces,values,new_vertices,new_faces,device='cuda:0',execution='optimized',*,old_area=None,new_area=None,source_precision=False):
    """Resample a scalar metric using forward and reverse area corrected weights."""
    selected=torch.device(device)
    forward_map=RadialSphereMap(vertices,faces,selected,execution=execution,source_precision=source_precision)
    reverse_map=RadialSphereMap(new_vertices,new_faces,selected,execution=execution,source_precision=source_precision)
    forward_ids,forward_weight,_=forward_map.weights(torch.as_tensor(new_vertices,device=selected))
    reverse_ids,reverse_weight,_=reverse_map.weights(torch.as_tensor(vertices,device=selected))
    if selected.type == "cuda" and execution == "optimized":
        from ._ordered_sparse import adaptive_values, SparseLayoutTooLarge
        record_statistics(cuda_adaptive_resampling_calls=1)
        old_area=(_vertex_area(np.asarray(vertices,dtype=np.float64),np.asarray(faces,dtype=np.int64))
                  if old_area is None else np.asarray(old_area,dtype=np.float64))
        new_area=(_vertex_area(np.asarray(new_vertices,dtype=np.float64),np.asarray(new_faces,dtype=np.int64))
                  if new_area is None else np.asarray(new_area,dtype=np.float64))
        if (old_area.shape != (len(vertices),) or new_area.shape != (len(new_vertices),)
                or not np.isfinite(old_area).all() or not np.isfinite(new_area).all()):
            raise ValueError("adaptive resampling vertex-area arrays differ from their meshes")
        try:
            return adaptive_values(forward_ids,forward_weight,reverse_ids,reverse_weight,
                                   values,old_area,new_area).cpu().numpy()
        except SparseLayoutTooLarge:
            # Keep the original sparse algorithm for exceptional meshes;
            # never allocate a huge rectangular padding table on CUDA.
            record_statistics(adaptive_cpu_layout_fallback_calls=1)
    fi=forward_ids.cpu().numpy();fw=forward_weight.cpu().numpy()
    ri=reverse_ids.cpu().numpy();rw=reverse_weight.cpu().numpy()
    m=len(new_vertices);n=len(vertices)
    forward=sparse.coo_matrix((fw.ravel(),(np.repeat(np.arange(m),3),fi.ravel())),shape=(m,n)).tocsr()
    reverse=sparse.coo_matrix((rw.ravel(),(np.repeat(np.arange(n),3),ri.ravel())),shape=(n,m)).T.tocsr()
    choose=np.diff(reverse.indptr)>np.diff(forward.indptr)
    joined=sparse.diags(choose.astype(np.float64))@reverse + sparse.diags((~choose).astype(np.float64))@forward
    old_area=(_vertex_area(np.asarray(vertices,dtype=np.float64),np.asarray(faces,dtype=np.int64))
              if old_area is None else np.asarray(old_area,dtype=np.float64))
    new_area=(_vertex_area(np.asarray(new_vertices,dtype=np.float64),np.asarray(new_faces,dtype=np.int64))
              if new_area is None else np.asarray(new_area,dtype=np.float64))
    if old_area.shape!=(n,) or new_area.shape!=(m,) or not np.isfinite(old_area).all() or not np.isfinite(new_area).all():
        raise ValueError("adaptive resampling vertex-area arrays differ from their meshes")
    weighted=sparse.diags(new_area)@joined
    correction=np.asarray(weighted.sum(axis=0)).ravel()
    factors=np.divide(old_area,correction,out=np.zeros_like(old_area),where=correction>0)
    weighted=weighted@sparse.diags(factors)
    row_sum=np.asarray(weighted.sum(axis=1)).ravel()
    normalized=sparse.diags(np.divide(1.,row_sum,out=np.zeros_like(row_sum),where=row_sum>0))@weighted
    return np.asarray(normalized@values,dtype=np.float64)
def _sphere_warp(points,from_vertices,faces,to_vertices,device,execution="optimized",*,source_precision=False):
    mapper=RadialSphereMap(from_vertices,faces,device,execution=execution,source_precision=source_precision)
    ids,weights,patch=mapper.weights(points)
    if source_precision:
        from . import _fastpd_native
        query=points.detach().cpu().numpy().astype(np.float64,copy=False)
        destination=to_vertices.detach().cpu().numpy().astype(np.float64,copy=False)
        values=np.frombuffer(_fastpd_native.source_sphere_warp(
            mapper.vertex_bytes,mapper.face_bytes,destination.tobytes(),query.tobytes(),
            patch.detach().cpu().numpy().astype(np.int64,copy=False).tobytes(),
            len(from_vertices),len(faces),len(query)),dtype=np.float64).reshape(-1,3)
        return torch.tensor(values,device=device)
    order=ids.argsort(1)
    ids=ids.gather(1,order);weights=weights.gather(1,order)
    # sphere_project_warp iterates a std::map<int,double> in vertex-ID order.
    triangles=to_vertices[ids]
    result=(triangles[:,0]*weights[:,0,None]+triangles[:,1]*weights[:,1,None])+triangles[:,2]*weights[:,2,None]
    return _unit3(result)*100


def _unit3(points):
    from . import _point_cpu
    if _point_cpu.enabled(points):
        return _point_cpu.normalize(points)
    squared=(points[...,0]*points[...,0]+points[...,1]*points[...,1])+points[...,2]*points[...,2]
    norm=torch.sqrt(squared)
    denominator=torch.where(norm>1e-8,norm,torch.ones_like(norm))
    return points/denominator[...,None]


def _triplet_data_weights(prior,faces,patch,points):
    """Source likelihood projection in sorted triplet-corner order.

    The Octree chooses a face in mesh order; get_target_data then constructs
    that face again in sorted node-ID order. Reordering weights computed from
    the original corners preserves their formula but changes floating-point
    projection and area arithmetic. Recompute after the unchanged face lookup.
    """
    triangles=prior[faces[patch]]
    first,second,third=triangles.unbind(1)
    normal=_normalize(_cross(_normalize(third-first),_normalize(second-first)))
    from . import _point_cpu
    numerator,denominator=_dot(normal,first),_dot(normal,points)
    ratio=(_point_cpu.divide(numerator,denominator) if _point_cpu.enabled(numerator,denominator)
           else numerator/denominator)
    projected=points*ratio[:,None]
    return _area_weights(triangles,projected)


def _face_layout(faces, patch, data_weights, source, device):
    num_faces=len(faces)
    members=[[] for _ in range(num_faces)]
    for sample,face in enumerate(patch.cpu().numpy()):members[face].append(sample)
    max_points=max(map(len,members))
    index=np.zeros((num_faces,max_points),np.int64)
    mask=np.zeros((num_faces,max_points),bool)
    for face,items in enumerate(members):
        index[face,:len(items)]=items
        mask[face,:len(items)]=True
    packed=np.flatnonzero(mask.ravel())
    return (torch.as_tensor(index,device=device),torch.as_tensor(mask,device=device),
            data_weights,source,torch.as_tensor(packed,device=device))


def _face_costs(current,candidate,original,faces,layout,reference_map,reference_metric,lam,simval,config=None,energy_only=False,fold_reference=None):
    config=MSMSulcConfig() if config is None else config
    index,valid,weights,source,packed=layout
    bits=torch.as_tensor([[i>>2&1,i>>1&1,i&1] for i in range(1 if energy_only else 8)],
                         device=current.device,dtype=torch.bool)
    fixed=current[faces]
    moved=candidate[faces]
    proposed=torch.where(bits[None,:,:,None],moved[:,None,:,:],fixed[:,None,:,:])
    sample_idx=index
    sample_weights=weights[sample_idx]
    weighted=sample_weights[:,None,:,:,None]*proposed[:,:,None,:,:]
    xyz=_unit3((weighted[:,:,:,0]+weighted[:,:,:,1])+weighted[:,:,:,2])*100
    states=len(bits);width=index.shape[1]
    packed_expanded=((packed//width)[:,None]*(states*width)+
                     torch.arange(states,device=current.device)[None,:]*width+
                     (packed%width)[:,None]).reshape(-1)
    sampled=reference_map.sample(xyz.reshape(-1,3)[packed_expanded],reference_metric)
    target=torch.zeros(xyz.shape[:-1],dtype=sampled.dtype,device=current.device)
    target.reshape(-1)[packed_expanded]=sampled
    native=source[sample_idx][:,None,:]
    observed=valid[:,None,:].to(native.dtype)
    count=observed.sum(-1).clamp_min(1)
    mean_source=(native*observed).sum(-1)/count
    mean_target=(target*observed).sum(-1)/count
    centered_source=native-mean_source[:,:,None]
    centered_target=target-mean_target[:,:,None]
    cov=(centered_source*centered_target*observed).sum(-1)/count
    vx=(centered_source.square()*observed).sum(-1)/count
    vy=(centered_target.square()*observed).sum(-1)/count
    denom=torch.sqrt(vx)*torch.sqrt(vy)
    corr=torch.where((vx!=0)&(vy!=0),cov/torch.where(denom!=0,denom,1),0)
    if simval==1:
        similarity=torch.sqrt((((native-target)*observed).square()).sum(-1))/count
    else:
        similarity=1-(1+corr)*0.5

    cost=_regularized_triangle_cost(similarity,proposed,original,faces,lam,config,
                                    current,fold_reference)
    return cost.detach().cpu().numpy()


def _regularized_triangle_cost(similarity,proposed,original,faces,lam,config,current,fold_reference=None):
    """Shared strain/folding term; preserve the MSMSulc arithmetic order."""
    old=original[faces][:,None,:,:]
    u0=old[:,:,1,:]-old[:,:,0,:];v0=old[:,:,2,:]-old[:,:,0,:]
    u1=proposed[:,:,1,:]-proposed[:,:,0,:]
    v1=proposed[:,:,2,:]-proposed[:,:,0,:]
    a=(u0*u0).sum(-1);b=(u0*v0).sum(-1);c=(v0*v0).sum(-1)
    d=(u1*u1).sum(-1);e=(u1*v1).sum(-1);f=(v1*v1).sum(-1)
    det0=(a*c-b*b).clamp_min(1e-8)
    det1=(d*f-e*e).clamp_min(1e-8)
    j=torch.sqrt(det1/det0)
    trace=(d*c+f*a-2*e*b)/det0
    invariant=trace/j
    ratio=torch.where(invariant<=2,torch.ones_like(invariant),
                      0.5*(invariant+torch.sqrt((invariant.square()-4).clamp_min(0))))
    shape_power=ratio.pow(config.strain_exponent)
    area_power=j.pow(config.strain_exponent)
    strain=0.5*(config.shear_modulus*(shape_power+shape_power.reciprocal()-2)+
                config.bulk_modulus*(area_power+area_power.reciprocal()-2))
    regularization=strain.pow(config.regularization_exponent)
    current_triangles=(current if fold_reference is None else fold_reference)[faces]
    current_normal=torch.cross(current_triangles[:,1]-current_triangles[:,0],
                               current_triangles[:,2]-current_triangles[:,0],dim=-1)
    folded=(torch.cross(u1,v1,dim=-1)*current_normal[:,None,:]).sum(-1)<0
    cost=torch.where(folded,torch.full_like(similarity,1e7*lam),similarity+lam*regularization)
    return cost


def _label_samples(grid,faces,max_distance):
    neighbours=[[] for _ in grid]
    for a,b,c in faces:
        for vertex,others in ((a,(b,c)),(b,(a,c)),(c,(a,b))):
            for other in others:
                if other not in neighbours[vertex]:neighbours[vertex].append(other)
    centroid=next(i for i,n in enumerate(neighbours) if len(n)==6)
    centre=grid[centroid]
    frontier=[centroid];seen=set();samples={}
    # std::map<double,Point> replaces equal-distance entries. Sorting vertex
    # IDs by distance retains both and changes the expansion proposal order.
    while frontier:
        next_frontier=[]
        for vertex in frontier:
            for neighbour in neighbours[vertex]:
                if neighbour==centroid or neighbour in seen:continue
                delta=grid[neighbour]-centre
                distance=float(np.sqrt(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]))
                if distance<=max_distance:
                    samples[distance]=grid[neighbour]
                    seen.add(neighbour);next_frontier.append(neighbour)
        frontier=next_frontier
    return centre,np.asarray([samples[d] for d in sorted(samples)],dtype=np.float64).reshape(-1,3)


def _rescaled_labels(centre,samples,scale):
    if scale>=0.25:
        raw=centre+(centre-samples)*scale
        labels=raw/np.linalg.norm(raw,axis=1,keepdims=True)*100
    else:
        scale=1.0
        labels=samples.copy()
    return labels,scale*0.8


def _rotation_matrices(positions,centre,device):
    """Cache the source Point/libm Rodrigues matrices once per iteration."""
    from . import _fastpd_native
    points=np.ascontiguousarray(positions,dtype=np.float64)
    origin=np.ascontiguousarray(centre,dtype=np.float64)
    data=_fastpd_native.source_rotation_matrices(points,origin,len(points))
    matrices=np.frombuffer(data,dtype=np.float64).reshape(-1,3,3).copy()
    return torch.as_tensor(matrices,device=device)


def _rotated_label(rotations,sample):
    """Apply the cached matrices in source three-term Point order."""
    label=torch.as_tensor(sample,dtype=rotations.dtype,device=rotations.device)
    return (rotations[:,:,0]*label[0]+rotations[:,:,1]*label[1])+rotations[:,:,2]*label[2]


def _normalize_sphere(vertices):
    """newMSM's four-point sphere-origin estimate, then radius normalization."""
    points=np.asarray(vertices,dtype=np.float64).copy()
    samples=points[[len(points)//i-1 for i in range(1,5)]]
    differences=samples[1:]-samples[0]
    try:
        center=np.linalg.solve(2*differences,np.square(samples[1:]).sum(1)-np.square(samples[0]).sum())
    except np.linalg.LinAlgError:
        center=np.zeros(3)
    if np.linalg.norm(center)>1e-2:
        nonzero=np.linalg.norm(points,axis=1)!=0
        points[nonzero]-=center
    norms=np.linalg.norm(points,axis=1,keepdims=True)
    if np.any(norms<=1e-8):raise ValueError("sphere contains a zero-radius vertex")
    return points/norms*100


def _variance_normalize(values):
    """The official sample-variance Welford update in original vertex order."""
    values=np.asarray(values,dtype=np.float64)
    mean=0.0;variance=0.0
    for i,value in enumerate(values):
        delta=value-mean
        mean+=delta/(i+1)
        variance+=delta*(value-mean)
    variance/=len(values)-1
    centered=values-mean
    return centered/np.sqrt(variance) if variance>0 else centered


@lru_cache(maxsize=12)
def _unfold_incident(face_bytes,count):
    faces=np.frombuffer(face_bytes,dtype=np.int64).reshape(-1,3).copy()
    incident=[[] for _ in range(count)]
    for i,triangle in enumerate(faces):
        for vertex in triangle:incident[vertex].append(i)
    width=max(map(len,incident))
    table=np.asarray([ids+[ids[0]]*(width-len(ids)) for ids in incident],np.int64)
    return faces,incident,table


def _unfold(vertices,faces):
    """The newMSM sequential area-gradient/step-halving unfolding operation."""
    faces_np=np.asarray(faces,dtype=np.int64)
    faces_np,incident,table=_unfold_incident(faces_np.tobytes(),len(vertices))
    ids=torch.as_tensor(faces_np,device=vertices.device)
    local=torch.as_tensor(table,device=vertices.device)
    triangles=vertices[ids]
    normals=F.normalize(torch.cross(triangles[:,2]-triangles[:,0],
                                    triangles[:,1]-triangles[:,0],dim=-1),dim=-1)
    intersections=(normals[local[:,0]][:,None,:]*normals[local]).sum(-1)<=0.5
    if not bool(intersections.any()):return vertices,0
    points=vertices.detach().cpu().numpy().copy()
    moved=0
    def normals_for(vertex):
        triangle=points[faces_np[incident[vertex]]]
        normal=np.cross(triangle[:,2]-triangle[:,0],triangle[:,1]-triangle[:,0])
        normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        return normal
    def folded(vertex):
        normal=normals_for(vertex)
        return np.any((normal[0]*normal).sum(1)<=0.5)
    for _ in range(1000):
        triangle=points[faces_np]
        normal=np.cross(triangle[:,2]-triangle[:,0],triangle[:,1]-triangle[:,0])
        normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        selected=np.flatnonzero(((normal[table[:,0]][:,None,:]*normal[table]).sum(-1)<=0.5).any(1))
        if not len(selected):break
        gradients=[]
        for vertex in selected:
            grad=np.zeros(3)
            for face in incident[vertex]:
                triangle_ids=faces_np[face]
                corner=int(np.flatnonzero(triangle_ids==vertex)[0])
                a,b,c=points[triangle_ids[[((corner+1)%3),((corner+2)%3),corner]]]
                first=c-a;second=b-a
                length=np.linalg.norm(second)
                first=first/max(np.linalg.norm(first),1e-10)
                second=second/max(length,1e-10)
                n=np.cross(first,second);n/=max(np.linalg.norm(n),1e-10)
                edge=np.cross(second,n)
                if np.dot(first,edge)<0:edge=-edge
                grad+=edge*(0.5*length)
            gradients.append(grad)
        # The gradients are frozen for this sweep; trial moves are applied in
        # original vertex order and each checks the already moved neighbours.
        for vertex,grad in zip(selected,gradients):
            start=points[vertex].copy();step=1.0
            while True:
                proposed=start-grad*step
                proposed/=np.linalg.norm(proposed)
                points[vertex]=proposed*100
                step*=0.5
                if not folded(vertex) or step<=1e-3:break
            moved+=1
    return torch.as_tensor(points,dtype=vertices.dtype,device=vertices.device),moved


def _native_output_qc(vertices,faces,original):
    """Measure native orientation at solver and written GIFTI precision.

    The official final transform interpolates the native sphere and saves it
    without another unfolding operation. Preserve those coordinates and report
    their orientation; DATA/control meshes retain the per-iteration unfolding.
    """
    points=np.asarray(vertices,dtype=np.float64)
    if not np.isfinite(points).all():raise RuntimeError("output sphere contains nonfinite coordinates")
    faces=np.asarray(faces,dtype=np.int64)
    before=np.asarray(original,dtype=np.float64)[faces]
    baseline=(np.cross(before[:,1]-before[:,0],before[:,2]-before[:,0])*before[:,0]).sum(1)
    usable=baseline!=0
    def orientation(coordinates):
        triangles=np.asarray(coordinates,dtype=np.float64)[faces]
        signed=(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])*triangles[:,0]).sum(1)
        ratios=np.divide(signed,baseline,out=np.full_like(signed,np.nan),where=usable)
        return int(np.count_nonzero(ratios<=0)),float(ratios[usable].min()) if usable.any() else None
    folded_solver,min_solver=orientation(points)
    folded_written,min_written=orientation(points.astype(np.float32))
    return {"folded_output_faces":folded_written,"folded_solver_faces":folded_solver,
            "minimum_output_orientation_ratio":min_written,
            "minimum_solver_orientation_ratio":min_solver,
            "degenerate_input_faces":int(np.count_nonzero(~usable))}


def run_msmsulc(
    inputs: dict[str, MSMSulcInputs], output_dir: str | Path, *,
    device: str = "cuda:0", config: MSMSulcConfig | str | Path | None = None,
    execution: str = "optimized", parallel: bool = True, cpu_threads: int | None = None,
    qc_policy: str = "report",
) -> dict[str, Path]:
    """Register both sulcal spheres with the explicit official MSMSulc schedule.

    ``qc_policy`` controls the native-sphere output boundary. ``report`` keeps
    the source-compatible interpolation and records any folded faces,
    ``repair`` applies sequential unfolding to a folded native output, and
    ``error`` refuses to write a folded sphere. The standalone function and
    the surface pipeline both keep ``report`` as the compatibility default;
    repair is an explicit production option.
    """
    from . import _fastpd_native
    if set(inputs) != {"L", "R"}:
        raise ValueError("inputs must contain L and R MSMSulc inputs")
    if config is None:config=MSMSulcConfig()
    elif isinstance(config,(str,Path)):config=MSMSulcConfig.from_file(config)
    elif not isinstance(config,MSMSulcConfig):raise TypeError("config must be MSMSulcConfig or a config path")
    if execution not in ("optimized","reference"):raise ValueError("execution must be optimized or reference")
    if qc_policy not in ("report", "repair", "error"):
        raise ValueError("qc_policy must be 'report', 'repair' or 'error'")
    selected=torch.device(device)
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.cuda.init()
    output=Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    def register(hemi, threads):
        path, report = _register_msmsulc_one(inputs[hemi],output,hemi=hemi,
                                           device=device,config=config,execution=execution,
                                           qc_policy=qc_policy)
        report["cpu_threads"] = threads
        report["execution_counts"] = current_statistics()
        return path, report
    results = register_hemispheres(register,selected,parallel=parallel,cpu_threads=cpu_threads)
    report = {hemi:results[index][1] for index,hemi in enumerate("LR")}
    overall = execution_report(selected,parallel=parallel,cpu_threads=cpu_threads)
    for hemisphere in "LR":
        report[hemisphere]["peak_allocated_gb"] = overall["peak_allocated_gb"]
        report[hemisphere]["peak_scope"] = overall["peak_scope"]
    report["execution"] = overall
    (output/"registration_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return {hemi:output/f"{hemi}.sphere.MSMSulc.native.surf.gii" for hemi in "LR"}


def _register_msmsulc_one(entry, output, *, hemi, device, config, execution, qc_policy="report"):
    from . import _fastpd_native
    selected=torch.device(device)
    started=time.perf_counter()
    native,native_faces=_surface(entry.rotated_sphere)
    reference_xyz,reference_faces=_surface(entry.reference_sphere)
    # File-loaded Triangle areas are cached before recentre/true_rescale.
    native_area=_vertex_area(native,native_faces)
    reference_area=_vertex_area(reference_xyz,reference_faces)
    native=_normalize_sphere(native)
    reference_xyz=_normalize_sphere(reference_xyz)
    source_metric=np.asarray(nib.load(str(entry.native_sulc)).darrays[0].data,np.float64)
    reference_metric=np.asarray(nib.load(str(entry.reference_sulc)).darrays[0].data,np.float64)
    if (source_metric.shape!=(len(native),) or reference_metric.shape!=(len(reference_xyz),)
            or not np.isfinite(source_metric).all() or not np.isfinite(reference_metric).all()):
        raise ValueError(f"{hemi} sphere/metric dimensions differ or metric is not finite")
    affine_started=time.perf_counter()
    affine_grid,affine_faces,affine_area=_ico(config.data_grid[0],cached_area=True)
    affine_source=_variance_normalize(_adaptive_resample(native,native_faces,source_metric,
                                                          affine_grid,affine_faces,device=device,execution=execution,
                                                          old_area=native_area,new_area=affine_area))
    affine_target=_variance_normalize(_adaptive_resample(reference_xyz,reference_faces,reference_metric,
                                                          affine_grid,affine_faces,device=device,execution=execution,
                                                          old_area=reference_area,new_area=affine_area))
    affine,angles,affine_report,previous_positions=_affine_initialization(affine_grid,affine_faces,
                                                      affine_source,affine_target,selected,config,execution=execution)
    previous_grid=affine_grid
    previous_faces=affine_faces
    affine_seconds=time.perf_counter()-affine_started
    stages=[]
    for stage_index in range(1,4):
        stage_started=time.perf_counter()
        level=config.control_grid[stage_index]
        data_level=config.data_grid[stage_index]
        lam=config.regularization[stage_index]
        regular_np,faces_np=_ico(level)
        data_np,data_faces,data_area=_ico(data_level,cached_area=True)
        label_grid,label_faces=_ico(config.sampling_grid[stage_index])
        regular=torch.as_tensor(regular_np,dtype=torch.float64,device=selected)
        strain_original=torch.as_tensor(data_np[:len(regular_np)],dtype=torch.float64,device=selected)
        # Official transfer: previous data grid -> native mesh -> the new
        # data/control grids. Direct CP-to-CP transfer changes trajectories.
        native_positions=_sphere_warp(torch.as_tensor(native,device=selected),previous_grid,
                                      previous_faces,previous_positions,selected,execution=execution)
        source_positions=_sphere_warp(torch.as_tensor(data_np,device=selected),native,
                                      native_faces,native_positions,selected,execution=execution)
        cp_positions=_sphere_warp(regular,native,native_faces,native_positions,selected,execution=execution)
        source_positions,source_unfold=_unfold(source_positions,data_faces)
        cp_positions,control_unfold=_unfold(cp_positions,faces_np)
        src=_variance_normalize(_adaptive_resample(native,native_faces,source_metric,
                                                    data_np,data_faces,device=device,execution=execution,
                                                    old_area=native_area,new_area=data_area))
        tgt=_variance_normalize(_adaptive_resample(reference_xyz,reference_faces,reference_metric,
                                                    data_np,data_faces,device=device,execution=execution,
                                                    old_area=reference_area,new_area=data_area))
        source_values=torch.as_tensor(src,device=selected)
        target_values=torch.as_tensor(tgt,device=selected)
        target_map=RadialSphereMap(data_np,data_faces,selected,execution=execution)
        edges=np.unique(np.sort(np.concatenate((faces_np[:,[0,1]],
            faces_np[:,[1,2]],faces_np[:,[2,0]])),axis=1),axis=0)
        chord=np.linalg.norm(regular_np[edges[:,0]]-regular_np[edges[:,1]],axis=1)
        spacing=float((200*np.arcsin(chord/200)).max())
        centre,samples=_label_samples(label_grid,label_faces,0.5*spacing)
        sorted_faces=np.sort(faces_np,axis=1).astype(np.int32)
        face_tensor=torch.as_tensor(sorted_faces.astype(np.int64),device=selected)
        face_bytes=sorted_faces.tobytes()
        iterations=[];scale=1.0;previous_energy=0.0;converged=False
        for iteration in range(config.iterations[stage_index]):
            iteration_started=time.perf_counter()
            prior=cp_positions.clone()
            prior_np=prior.detach().cpu().numpy()
            rotations=_rotation_matrices(prior_np,centre,selected)
            current_map=RadialSphereMap(prior_np,faces_np,selected,execution=execution)
            _,_,patch=current_map.weights(source_positions)
            weights=_triplet_data_weights(prior,face_tensor,patch,source_positions)
            layout=_face_layout(sorted_faces,patch,weights,source_values,selected)
            labels=np.zeros(len(regular_np),np.int16)
            changed=0
            label_positions,scale=_rescaled_labels(centre,np.vstack((centre,samples)),scale)
            # Source costs and applyLabeling rotate every label, including
            # label zero. R(prior)*centre differs from prior by rounding;
            # retaining prior for zero labels alters near-tie proposals.
            cp_positions=_rotated_label(rotations,label_positions[0])
            for _ in range(2):
                for label,sample in enumerate(label_positions):
                    if np.all(labels==label):continue
                    candidate=_rotated_label(rotations,sample)
                    costs=_face_costs(cp_positions,candidate,strain_original,face_tensor,
                                      layout,target_map,target_values,lam,
                                      simval=config.simval[stage_index],config=config,fold_reference=prior)
                    ordered=costs.astype(np.float64)
                    choice=np.frombuffer(_fastpd_native.optimize(
                        face_bytes,ordered.tobytes(),len(regular_np)),dtype=np.uint8)
                    update=(choice==1)&(labels!=label)
                    if update.any():
                        mask=torch.as_tensor(update,device=selected)
                        cp_positions[mask]=candidate[mask]
                        labels[update]=label
                        changed+=int(update.sum())
            energy_costs=_face_costs(cp_positions,cp_positions,strain_original,face_tensor,
                                     layout,target_map,target_values,lam,
                                     simval=config.simval[stage_index],config=config,
                                     energy_only=True,fold_reference=prior)
            # evaluateTotalCostSum adds triplets sequentially in face order.
            energy=sum(float(value) for value in energy_costs[:,0])
            # Source checks convergence before applyLabeling and before the
            # control/data warp. Revert this tentative fusion on stopping.
            stopping=iteration>2 and (iteration-1)%2==0 and previous_energy-energy<0.001
            if stopping:
                cp_positions=prior;converged=True
                iterations.append({"changed":changed,"energy":energy,"applied":False,
                                   "seconds":time.perf_counter()-iteration_started})
                break
            source_positions=_sphere_warp(source_positions,prior_np,
                                          faces_np,cp_positions,selected,execution=execution)
            cp_positions,moved=_unfold(cp_positions,faces_np);control_unfold+=moved
            source_positions,moved=_unfold(source_positions,data_faces);source_unfold+=moved
            previous_energy=energy
            iterations.append({"changed":changed,"energy":energy,"applied":True,
                               "seconds":time.perf_counter()-iteration_started})
        previous_grid=data_np;previous_faces=data_faces;previous_positions=source_positions
        stages.append({"control_points":len(regular_np),"data_points":len(data_np),
                       "labels":len(samples)+1,"similarity":config.simval[stage_index],
                       "maximum_iterations":config.iterations[stage_index],"converged":converged,
                       "source_unfold_updates":source_unfold,"control_unfold_updates":control_unfold,
                       "iterations":iterations,"seconds":time.perf_counter()-stage_started})
    vertices=_sphere_warp(torch.as_tensor(native,device=selected),previous_grid,
                          previous_faces,previous_positions,selected,execution=execution).detach().cpu().numpy()
    output_qc_before_repair=_native_output_qc(vertices,native_faces,native)
    repair = {"policy": qc_policy, "applied": False, "moved_vertices": 0}
    if output_qc_before_repair["folded_output_faces"]:
        if qc_policy == "error":
            raise RuntimeError(
                f"{hemi} MSMSulc native sphere has "
                f"{output_qc_before_repair['folded_output_faces']} folded faces"
            )
        if qc_policy == "repair":
            repaired, moved = _unfold(
                torch.as_tensor(vertices, dtype=torch.float64, device=selected), native_faces
            )
            vertices = repaired.detach().cpu().numpy()
            repair.update(applied=True, moved_vertices=int(moved))
    output_qc=_native_output_qc(vertices,native_faces,native)
    repair["success"] = output_qc["folded_output_faces"] == 0
    if qc_policy == "repair" and not repair["success"]:
        raise RuntimeError(f"{hemi} MSMSulc native sphere fold repair did not pass QC")
    # ``report`` preserves the official final interpolation for exact
    # source-compatible comparisons. ``repair`` is explicit and changes
    # native coordinates; the pre-repair QC remains available for audit.
    path=output/f"{hemi}.sphere.MSMSulc.native.surf.gii"
    nib.save(nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(vertices.astype(np.float32),intent="NIFTI_INTENT_POINTSET"),
        nib.gifti.GiftiDataArray(native_faces.astype(np.int32),intent="NIFTI_INTENT_TRIANGLE")]),str(path))
    report={"seconds":time.perf_counter()-started,
                  "affine_angles_deg":angles,"affine_seconds":affine_seconds,
                  "affine":affine_report,"config":config.to_dict(),"execution":execution,
                  **output_qc,
                  "native_output_qc_before_repair": output_qc_before_repair,
                  "fold_repair": repair,
                  "orientation_qc": "pass" if repair["success"] else "warning",
                  "stages":stages}
    return path, report
