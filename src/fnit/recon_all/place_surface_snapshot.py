"""Trial-local broadphase CSR and compiled ordered placement; no Jacobi updates.

Modified FNIT implementation of FreeSurfer placement formulas, source d932c45;
FreeSurfer Software License: licenses/FreeSurfer.txt.
"""
from itertools import chain
import numpy as np
from numba import njit
import time
from .place_surface_collision import _moved_face_geometry,_candidate_collision,_project_close_neighbors


@njit(cache=True)
def _query_radius_candidates(candidates, centers, query_center, query_radius):
    result=np.empty(len(candidates),np.int32)
    count=0
    squared_radius=query_radius*query_radius
    for face in candidates:
        dx=centers[face,0]-query_center[0]
        dy=centers[face,1]-query_center[1]
        dz=centers[face,2]-query_center[2]
        if dx*dx+dy*dy+dz*dz<=squared_radius:
            result[count]=face
            count+=1
    return result[:count]


@njit(cache=True)
def _ordered_snapshot_step(initial,triangles,proposal,order,incident,incident_offsets,
                           neighbors,neighbor_valid,offsets,has_offsets,accepted_offsets,
                           update_offsets,geometry,vertex_svi,min_neighbor_mm,
                           initial_centers,maximum_radius,candidate_offsets,candidates,motion_bound):
    current=initial.copy()
    for vertex in order:
        unchanged=True
        for axis in range(3):
            if proposal[vertex,axis]!=initial[vertex,axis]:unchanged=False
        if unchanged:continue
        endpoint=proposal[vertex].copy()
        final_offset=offsets[vertex].copy()
        if has_offsets:
            projected,valid=_project_close_neighbors(current,int(vertex),neighbors,neighbor_valid,
                         offsets[vertex],geometry,int(vertex_svi[vertex]),min_neighbor_mm)
            if not valid:continue
            endpoint=(initial[vertex]+projected).astype(np.float32)
            final_offset=projected
        squared=0.0
        for axis in range(3):
            displacement=float(endpoint[axis])-float(initial[vertex,axis])
            squared+=displacement*displacement
        if squared>motion_bound*motion_bound:
            raise ValueError('projected endpoint exceeds trial snapshot motion bound')
        collision=False
        for slot in range(incident_offsets[vertex],incident_offsets[vertex+1]):
            face=incident[slot]
            moved,center,radius,low,high=_moved_face_geometry(current,triangles,face,int(vertex),endpoint)
            nearby=_query_radius_candidates(candidates[candidate_offsets[face]:candidate_offsets[face+1]],
                            initial_centers,center,(radius+maximum_radius)+1.0)
            if _candidate_collision(current,triangles,moved,triangles[face],low,high,nearby):
                collision=True
                break
        if collision:
            if update_offsets:accepted_offsets[vertex]=0.0
        else:
            current[vertex]=endpoint
            if update_offsets:accepted_offsets[vertex]=final_offset
    return current


def snapshot_ordered_step(xyz,triangles,proposal,order,incident,incident_offsets,
                          neighbors,neighbor_valid,offsets,accepted_offsets,geometry,
                          vertex_svi,min_neighbor_mm,tree,centers,radii,maximum_radius,
                          *, candidate_device=None, candidate_diagnostics=None,
                          candidate_grid_cells_per_axis=2):
    """One first trial; build a conservative union then filter exact query radii.

    Every vertex moves only after preceding accepted updates. Current candidate
    triangle coordinates are read by the existing compiled intersection test.
    The union is rebuilt for each input/trial. Rejected retained-MHT trials use
    the original tree path in the caller, never this CSR.
    candidate_device=None preserves cKDTree; indexed CUDA selects the complete
    Torch grid and conservative initial AABB filter. candidate_diagnostics is
    an optional mutable dictionary for counts and complete wall times. Runtime
    motion-bound failure raises ValueError; CUDA and candidate budget failures
    propagate. Dynamic triangle predicates and close-neighbor projection remain
    the original ordered Numba kernels, rather than fixed-state GPU decisions.
    """
    displacement=np.asarray(proposal,dtype=np.float64)-np.asarray(xyz,dtype=np.float64)
    maximum=np.linalg.norm(displacement,axis=1).max(initial=0)
    if offsets is not None:maximum=max(maximum,np.linalg.norm(np.asarray(offsets,dtype=np.float64),axis=1).max(initial=0))
    # Orthogonal close-neighbor projection does not grow displacement; doubling
    # plus 0.01 mm provides an explicit guarded allowance for FP32 rounding.
    bound=2.0*float(maximum)+0.01
    # Center can move by bound, radius by 2*bound. All live tree queries are
    # therefore contained in the union sphere expanded by 3*bound.
    build_started=time.perf_counter()
    if candidate_device is None:
        counts=np.empty(len(triangles),np.int64);chunks=[]
        for start in range(0,len(triangles),4096):
            end=min(start+4096,len(triangles))
            lists=tree.query_ball_point(centers[start:end],((radii[start:end]+maximum_radius)+1.0)+3.0*bound,
                                       return_sorted=False,workers=1)
            sizes=np.fromiter((len(row) for row in lists),np.int64,count=len(lists))
            counts[start:end]=sizes
            chunks.append(np.fromiter(chain.from_iterable(lists),np.int32,count=int(sizes.sum())))
        candidate_offsets=np.empty(len(triangles)+1,np.int64);candidate_offsets[0]=0
        np.cumsum(counts,out=candidate_offsets[1:]);candidates=np.concatenate(chunks) if chunks else np.empty(0,np.int32)
    else:
        from .place_surface_candidates_torch import conservative_face_candidates_torch
        initial_points=xyz[triangles]
        low,high=initial_points.min(axis=1),initial_points.max(axis=1)
        candidate_offsets,candidates,details=conservative_face_candidates_torch(
            centers,centers,((radii+maximum_radius)+1.0)+3.0*bound,
            source_low=low,source_high=high,query_low=low,query_high=high,
            motion_bound=bound,source_faces=triangles,query_faces=triangles,
            device=candidate_device, grid_cells_per_axis=candidate_grid_cells_per_axis)
        if candidate_diagnostics is not None:candidate_diagnostics.update(details)
    if candidate_diagnostics is not None:
        candidate_diagnostics.update(candidate_build_seconds=time.perf_counter()-build_started,
            motion_bound_mm=bound,candidate_pairs=len(candidates),
            effective_candidate_backend="torch_snapshot" if candidate_device is not None else "snapshot")
    has_offsets=offsets is not None
    if not has_offsets:
        offsets=np.zeros_like(xyz);neighbors=np.empty((len(xyz),0),np.int32);neighbor_valid=np.empty((len(xyz),0),np.bool_)
    update_offsets=accepted_offsets is not None
    if not update_offsets:accepted_offsets=np.zeros_like(xyz)
    ordered_started=time.perf_counter()
    result=_ordered_snapshot_step(xyz,triangles,proposal,order,incident,incident_offsets,
               neighbors,neighbor_valid,offsets,has_offsets,accepted_offsets,update_offsets,
               geometry,vertex_svi,np.float32(min_neighbor_mm),centers,maximum_radius,
               candidate_offsets,candidates,bound)
    if candidate_diagnostics is not None:candidate_diagnostics["ordered_acceptance_seconds"]=time.perf_counter()-ordered_started
    return result
