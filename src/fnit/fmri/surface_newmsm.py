"""Experimental FNIT implementation of multilevel MSMSulc registration.

The solver is independent of the newMSM binary. Its current spherical output
is not numerically equivalent to newMSM; see docs/fmri/surface.md.
"""
from pathlib import Path
import json
import time

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull, cKDTree
import torch
import torch.nn.functional as F

from .surface_msmsulc import _surface, _smoothing_graph, _affine_initialization
from .surface_registration import MSMSulcInputs

def _ico(level):
    a, b = 0.8506508084, 0.5257311121
    vertices = np.array([(a,b,0),(-a,b,0),(-a,-b,0),(a,-b,0),
                         (b,0,a),(b,0,-a),(-b,0,-a),(-b,0,a),
                         (0,a,b),(0,-a,b),(0,-a,-b),(0,a,-b)], np.float32)
    faces = ConvexHull(vertices).simplices.astype(np.int64)
    for _ in range(level):
        edges = np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
        edge_sorted = np.sort(edges,axis=1)
        unique, inverse = np.unique(edge_sorted,axis=0,return_inverse=True)
        midpoint = F.normalize(torch.from_numpy(vertices[unique].mean(axis=1)),dim=1).numpy()
        midpoint_id = inverse.reshape(3,-1).T + len(vertices)
        a0,b0,c0 = faces.T
        ab,bc,ca = midpoint_id.T
        faces = np.concatenate([np.stack([a0,ab,ca],1),
                                np.stack([b0,bc,ab],1),
                                np.stack([c0,ca,bc],1),
                                np.stack([ab,bc,ca],1)])
        vertices = np.concatenate([vertices,midpoint])
    tri=vertices[faces]
    flip=(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])*tri[:,0]).sum(1)<0
    faces[flip]=faces[flip][:,[0,2,1]]
    return vertices*100,faces


class SphereMap:
    def __init__(self, vertices, faces, device):
        self.vertices=torch.as_tensor(vertices,dtype=torch.float32,device=device)
        self.faces=torch.as_tensor(faces,dtype=torch.long,device=device)
        self.tree=cKDTree(vertices)
        incident=[[] for _ in range(len(vertices))]
        for f,triangle in enumerate(faces):
            for vertex in triangle: incident[vertex].append(f)
        count=max(map(len,incident))
        adjacent=np.empty((len(vertices),count),np.int64)
        for vertex,items in enumerate(incident):
            adjacent[vertex]=items+[items[0]]*(count-len(items))
        self.adjacent=torch.as_tensor(adjacent,device=device)

    def weights(self, points):
        nearest=self.tree.query(points.detach().cpu().numpy(),k=1,workers=4)[1]
        candidates=self.adjacent[torch.as_tensor(nearest,device=points.device)]
        triangles=self.vertices[self.faces[candidates]]
        e0=triangles[:,:,1]-triangles[:,:,0]
        e1=triangles[:,:,2]-triangles[:,:,0]
        q=points[:,None,:]-triangles[:,:,0]
        aa=(e0*e0).sum(-1); ab=(e0*e1).sum(-1); bb=(e1*e1).sum(-1)
        qa=(q*e0).sum(-1); qb=(q*e1).sum(-1)
        det=(aa*bb-ab*ab).clamp_min(1e-8)
        w1=(bb*qa-ab*qb)/det
        w2=(aa*qb-ab*qa)/det
        weights=torch.stack([1-w1-w2,w1,w2],-1)
        projection=(triangles*weights[:,:,:,None]).sum(2)
        score=(projection-points[:,None,:]).square().sum(-1)+1e4*F.relu(-weights.min(-1).values).square()
        choice=score.argmin(1)
        row=torch.arange(len(points),device=points.device)
        selected=self.faces[candidates[row,choice]]
        chosen_weights=weights[row,choice].clamp_min(0)
        chosen_weights=chosen_weights/chosen_weights.sum(-1,keepdim=True)
        return selected,chosen_weights,candidates[row,choice]

    def sample(self, points, metric):
        ids,weights,_=self.weights(points)
        return (metric[ids]*weights).sum(-1)


def _vertex_area(vertices,faces):
    triangles=vertices[faces]
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                 triangles[:,2]-triangles[:,0]),axis=1)*0.5
    total=np.bincount(faces.ravel(),weights=np.repeat(area,3),minlength=len(vertices))
    counts=np.bincount(faces.ravel(),minlength=len(vertices))
    return total/counts


def _adaptive_resample(vertices,faces,values,new_vertices,new_faces,device='cuda:0'):
    """Resample a scalar metric using forward and reverse area corrected weights."""
    selected=torch.device(device)
    forward_map=SphereMap(vertices,faces,selected)
    reverse_map=SphereMap(new_vertices,new_faces,selected)
    forward_ids,forward_weight,_=forward_map.weights(torch.as_tensor(new_vertices,device=selected))
    reverse_ids,reverse_weight,_=reverse_map.weights(torch.as_tensor(vertices,device=selected))
    fi=forward_ids.cpu().numpy();fw=forward_weight.cpu().numpy()
    ri=reverse_ids.cpu().numpy();rw=reverse_weight.cpu().numpy()
    m=len(new_vertices);n=len(vertices)
    forward=sparse.coo_matrix((fw.ravel(),(np.repeat(np.arange(m),3),fi.ravel())),shape=(m,n)).tocsr()
    reverse=sparse.coo_matrix((rw.ravel(),(np.repeat(np.arange(n),3),ri.ravel())),shape=(n,m)).T.tocsr()
    choose=np.diff(reverse.indptr)>np.diff(forward.indptr)
    joined=sparse.diags(choose.astype(np.float64))@reverse + sparse.diags((~choose).astype(np.float64))@forward
    old_area=_vertex_area(np.asarray(vertices,dtype=np.float64),np.asarray(faces,dtype=np.int64))
    new_area=_vertex_area(np.asarray(new_vertices,dtype=np.float64),np.asarray(new_faces,dtype=np.int64))
    weighted=sparse.diags(new_area)@joined
    correction=np.asarray(weighted.sum(axis=0)).ravel()
    factors=np.divide(old_area,correction,out=np.zeros_like(old_area),where=correction>0)
    weighted=weighted@sparse.diags(factors)
    row_sum=np.asarray(weighted.sum(axis=1)).ravel()
    normalized=sparse.diags(np.divide(1.,row_sum,out=np.zeros_like(row_sum),where=row_sum>0))@weighted
    return np.asarray(normalized@values,dtype=np.float32)
def _colors(faces, count):
    neighbours=[set() for _ in range(count)]
    for a,b,c in faces:
        neighbours[a].update((b,c));neighbours[b].update((a,c));neighbours[c].update((a,b))
    color=np.full(count,-1,np.int64)
    for node in range(count):
        used={color[other] for other in neighbours[node] if color[other]>=0}
        color[node]=next(k for k in range(8) if k not in used)
    return [np.flatnonzero(color==k) for k in range(color.max()+1)]


def _face_layout(faces, patch, data_weights, source, device):
    num_faces=len(faces);num_nodes=int(faces.max())+1
    members=[[] for _ in range(num_faces)]
    for sample,face in enumerate(patch.cpu().numpy()):members[face].append(sample)
    max_points=max(map(len,members))
    index=np.zeros((num_faces,max_points),np.int64)
    mask=np.zeros((num_faces,max_points),bool)
    for face,items in enumerate(members):
        index[face,:len(items)]=items
        mask[face,:len(items)]=True
    incident=[[] for _ in range(num_nodes)]
    for face,triangle in enumerate(faces):
        for node in triangle:incident[node].append(face)
    max_inc=max(map(len,incident))
    face_for_node=np.zeros((num_nodes,max_inc),np.int64)
    valid=np.zeros((num_nodes,max_inc),bool)
    for node,items in enumerate(incident):
        face_for_node[node,:len(items)]=items
        valid[node,:len(items)]=True
    return (torch.as_tensor(index,device=device),torch.as_tensor(mask,device=device),
            torch.as_tensor(face_for_node,device=device),torch.as_tensor(valid,device=device),
            data_weights,source)


def _local_energy(nodes,positions,original,faces,layout,reference_map,reference_metric,
                  lam,radius,ring_scale):
    sample_idx,sample_valid,incident,face_valid,data_weight,source=layout
    n=len(nodes)
    adjacent=incident[nodes]
    valid=face_valid[nodes]
    triangle_ids=faces[adjacent]
    old=positions[triangle_ids]
    radial=F.normalize(positions[nodes],dim=-1)
    axis=torch.zeros_like(radial);axis[:,2]=1
    axis[radial[:,2].abs()>0.9]=torch.tensor([1.,0.,0.],device=radial.device)
    tangent=F.normalize(torch.cross(radial,axis,dim=-1),dim=-1)
    tangent2=torch.cross(radial,tangent,dim=-1)
    directions=torch.stack([tangent*np.cos(k*np.pi/3)+tangent2*np.sin(k*np.pi/3)
                            for k in range(6)],dim=1)
    zeros=torch.zeros((n,1,3),device=radial.device)
    offsets=torch.cat((zeros,directions*0.03*ring_scale,
                       directions*0.1*ring_scale),dim=1)
    candidates=F.normalize(positions[nodes,None,:]+offsets,dim=-1)*radius
    label_count=candidates.shape[1]
    replace=triangle_ids==nodes[:,None,None]
    proposed=torch.where(replace[:,None,:,:,None],
                         candidates[:,:,None,None,:],old[:,None,:,:,:])
    indices=sample_idx[adjacent]
    sample_w=data_weight[indices]
    xyz=F.normalize((sample_w[:,None,:,:,:,None]*proposed[:,:,:,None,:,:]).sum(-2),dim=-1)*radius
    reference=reference_map.sample(xyz.reshape(-1,3),reference_metric).reshape(xyz.shape[:-1])
    native=source[indices][:,None,:,:]
    observed=sample_valid[adjacent][:,None,:,:].float()
    count=observed.sum(-1).clamp_min(1)
    sx=(native*observed).sum(-1);sy=(reference*observed).sum(-1)
    cov=(native*reference*observed).sum(-1)-sx*sy/count
    vx=((native.square()*observed).sum(-1)-sx.square()/count).clamp_min(0)
    vy=((reference.square()*observed).sum(-1)-sy.square()/count).clamp_min(0)
    correlation=cov/(torch.sqrt(vx*vy)+1e-6)
    use=(observed.sum(-1)>=4)&(vx>1e-5)&(vy>1e-5)&valid[:,None,:]
    similarity=((1-correlation)/2)*use

    original_tri=original[triangle_ids][:,None,:,:,:]
    u0=original_tri[:,:,:,1]-original_tri[:,:,:,0]
    v0=original_tri[:,:,:,2]-original_tri[:,:,:,0]
    u1=proposed[:,:,:,1]-proposed[:,:,:,0]
    v1=proposed[:,:,:,2]-proposed[:,:,:,0]
    a=(u0*u0).sum(-1);b=(u0*v0).sum(-1);c=(v0*v0).sum(-1)
    d=(u1*u1).sum(-1);e=(u1*v1).sum(-1);f=(v1*v1).sum(-1)
    det0=(a*c-b*b).clamp_min(1e-8)
    det1=(d*f-e*e).clamp_min(1e-8)
    j=torch.sqrt(det1/det0)
    trace=(d*c+f*a-2*e*b)/det0
    shape=((trace/j).square()-4).clamp_min(0)
    strain=0.5*(0.4*shape+1.6*(j.square()+j.reciprocal().square()-2))
    signed=(torch.cross(u1,v1,dim=-1)*proposed[:,:,:,0]).sum(-1)
    signed0=(torch.cross(u0,v0,dim=-1)*original_tri[:,:,:,0]).sum(-1)
    fold=F.relu(0.2-signed/signed0).square()
    energy=(similarity+lam*strain.square()+20*fold)*valid[:,None,:]
    return energy.sum(-1),candidates


def _repair_folds(original, deformed, faces, device):
    """Move vertices of inverted output faces until their local orientation is valid."""
    before=original[faces]
    after=deformed[faces]
    signed_before=np.einsum('ij,ij->i',np.cross(before[:,1]-before[:,0],
                                               before[:,2]-before[:,0]),before[:,0])
    signed_after=np.einsum('ij,ij->i',np.cross(after[:,1]-after[:,0],
                                              after[:,2]-after[:,0]),after[:,0])
    inverted=np.flatnonzero(signed_after/signed_before<=0)
    if not len(inverted):
        return deformed,0,0.0
    moving=np.unique(faces[inverted].ravel())
    affected=np.flatnonzero(np.isin(faces,moving).any(axis=1))
    selected=torch.device(device)
    full=torch.as_tensor(deformed,device=selected)
    ids=torch.as_tensor(moving,device=selected,dtype=torch.long)
    local_faces=torch.as_tensor(faces[affected].astype(np.int64),device=selected)
    reference=torch.as_tensor(signed_before[affected],device=selected)
    starting=full[ids].detach()
    parameters=torch.nn.Parameter(starting.clone())
    optimizer=torch.optim.Adam([parameters],lr=0.02)
    radius=float(np.linalg.norm(original,axis=1).mean())
    for _ in range(250):
        positions=F.normalize(parameters,dim=1)*radius
        triangles=full.index_copy(0,ids,positions)[local_faces]
        signed=(torch.cross(triangles[:,1]-triangles[:,0],
                            triangles[:,2]-triangles[:,0],dim=1)*triangles[:,0]).sum(1)
        ratio=signed/reference
        if bool((ratio>0.05).all()):
            break
        loss=F.relu(0.1-ratio).square().sum()+0.0001*(positions-starting).square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    corrected=full.index_copy(0,ids,F.normalize(parameters,dim=1)*radius).detach().cpu().numpy()
    check=corrected[faces[affected]]
    signed=np.einsum('ij,ij->i',np.cross(check[:,1]-check[:,0],
                                        check[:,2]-check[:,0]),check[:,0])
    if np.any(signed/signed_before[affected]<=0):
        raise RuntimeError(f"sphere still has folded triangles after local repair")
    movement=np.linalg.norm(corrected[moving]-deformed[moving],axis=1)
    return corrected.astype(np.float32),len(inverted),float(movement.max())


def run_newmsm_msmsulc(
    inputs: dict[str, MSMSulcInputs], output_dir: str | Path, *,
    device: str = "cuda:0",
) -> dict[str, Path]:
    """Register left and right sulcal spheres with the experimental solver.

    ``inputs`` is the L/R mapping from ``prepare_msmsulc_inputs``;
    ``output_dir`` receives native-order L/R GIFTI spheres and a JSON report.
    ``device`` selects the PyTorch device. The returned mapping contains the
    two sphere paths. The current result is not newMSM-equivalent.
    """
    if set(inputs) != {"L", "R"}:
        raise ValueError("inputs must contain L and R MSMSulc inputs")
    selected=torch.device(device)
    if selected.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    output=Path(output_dir).expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    report={}
    for hemi in 'LR':
        start=time.perf_counter(); entry=inputs[hemi]
        native_xyz,native_faces=_surface(entry.rotated_sphere)
        reference_xyz,reference_faces=_surface(entry.reference_sphere)
        source=np.asarray(nib.load(str(entry.native_sulc)).darrays[0].data,np.float32)
        reference=np.asarray(nib.load(str(entry.reference_sulc)).darrays[0].data,np.float32)
        source_graph,_=_smoothing_graph(native_faces,len(native_xyz))
        reference_graph,_=_smoothing_graph(reference_faces,len(reference_xyz))
        reference_xyz,affine,angles=_affine_initialization(native_xyz,reference_xyz,
                                         source,reference,source_graph,reference_graph,selected)
        prior_map=None;prior_deformed=None
        stages=[]
        for level,data_level,lam,sweeps in ((2,4,10.,12),(3,5,7.5,14),(4,6,7.5,16)):
            cp_np,faces_np=_ico(level);data_np,data_faces=_ico(data_level)
            control=torch.as_tensor(cp_np,device=selected)
            faces=torch.as_tensor(faces_np,device=selected)
            data=torch.as_tensor(data_np,device=selected)
            cp_map=SphereMap(cp_np,faces_np,selected)
            _,data_weights,patch=cp_map.weights(data)
            if prior_map is None:
                positions=control.clone()
            else:
                previous_ids,previous_weights,_=prior_map.weights(control)
                positions=F.normalize((prior_deformed[previous_ids]*previous_weights[:,:,None]).sum(1),dim=1)*100
            with torch.no_grad():
                source_grid=_adaptive_resample(native_xyz,native_faces,source,data_np,data_faces,device=device)
                reference_grid=_adaptive_resample(
                    np.asarray(_surface(entry.reference_sphere)[0],np.float32),
                    reference_faces,reference,data_np,data_faces,device=device)
                values=torch.as_tensor(source_grid,device=selected)
                values=(values-values.mean())/values.std()
                ref=torch.as_tensor(reference_grid,device=selected)
                ref=(ref-ref.mean())/ref.std()
                reference_map=SphereMap(data_np@affine,data_faces,selected)
            layout=_face_layout(faces_np,patch,data_weights,values,selected)
            colors=[torch.as_tensor(ids,device=selected) for ids in _colors(faces_np,len(cp_np))]
            edge=np.unique(np.sort(np.concatenate([faces_np[:,[0,1]],faces_np[:,[1,2]],faces_np[:,[2,0]]]),axis=1),axis=0)
            spacing=float(np.linalg.norm(cp_np[edge[:,0]]-cp_np[edge[:,1]],axis=1).max())
            for sweep in range(sweeps):
                changes=0
                for nodes in colors:
                    if len(nodes)==0:continue
                    energy,candidates=_local_energy(nodes,positions,control,faces,layout,
                         reference_map,ref,lam,100.,spacing*(0.8**sweep))
                    best=energy.argmin(1)
                    changes+=int((best!=0).sum())
                    positions[nodes]=candidates[torch.arange(len(nodes),device=selected),best]
                if changes==0 and sweep >= 5:break
            prior_deformed=positions
            prior_map=cp_map
            stages.append({'control_points':len(cp_np),'data_points':len(data_np),
                           'sweeps':sweep+1,'last_changes':changes})
        native=torch.as_tensor(native_xyz,device=selected)
        ids,weights,_=prior_map.weights(native)
        vertices=F.normalize((prior_deformed[ids]*weights[:,:,None]).sum(1),dim=1)*100
        vertices=vertices.cpu().numpy().astype(np.float32)@affine.T
        vertices,folded_before,max_repair_mm=_repair_folds(
            native_xyz,vertices,native_faces,device)
        path=output/f'{hemi}.sphere.discrete.native.surf.gii'
        nib.save(nib.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(vertices,intent='NIFTI_INTENT_POINTSET'),
            nib.gifti.GiftiDataArray(native_faces,intent='NIFTI_INTENT_TRIANGLE')]),str(path))
        report[hemi]={'seconds':time.perf_counter()-start,'affine_angles_deg':angles,
                      'folded_before_repair':folded_before,'folded_after_repair':0,
                      'maximum_repair_displacement_mm':max_repair_mm,'stages':stages}
    (output/'registration_report.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    return {hemi: output/f'{hemi}.sphere.discrete.native.surf.gii' for hemi in 'LR'}
