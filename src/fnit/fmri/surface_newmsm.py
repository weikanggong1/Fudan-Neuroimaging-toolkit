"""MSMSulc registration with independent HOCR reduction and FastPD fusion."""
from pathlib import Path
import json
import time

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from .surface_msmsulc import _surface, _smoothing_graph, _affine_initialization
from .surface_registration import MSMSulcInputs

def _ico(level):
    a,b=0.8506508084,0.5257311121
    vertices=np.array([(a,b,0),(-a,b,0),(-a,-b,0),(a,-b,0),
                       (b,0,a),(b,0,-a),(-b,0,-a),(-b,0,a),
                       (0,a,b),(0,-a,b),(0,-a,-b),(0,a,-b)],np.float64)
    faces=np.array([(7,8,4),(9,7,4),(11,6,5),(6,10,5),
                    (3,4,0),(5,3,0),(1,7,2),(6,1,2),
                    (11,0,8),(1,11,8),(3,10,9),(10,2,9),
                    (0,4,8),(5,0,11),(3,9,4),(10,3,5),
                    (1,8,7),(11,1,6),(9,2,7),(2,10,6)],np.int64)[:,[0,2,1]]
    for _ in range(level):
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
        vertices/=np.linalg.norm(vertices,axis=1,keepdims=True)
        faces=np.asarray(next_faces,np.int64)
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
class RadialSphereMap:
    def __init__(self, vertices, faces, device):
        self.device=torch.device(device)
        self.vertices=torch.as_tensor(vertices,dtype=torch.float32,device=self.device)
        self.faces=torch.as_tensor(faces,dtype=torch.long,device=self.device)
        self.tree=cKDTree(np.asarray(vertices))
        incident=[[] for _ in range(len(vertices))]
        for face_id,triangle in enumerate(faces):
            for vertex in triangle:incident[vertex].append(face_id)
        width=max(map(len,incident))
        table=np.empty((len(vertices),width),np.int64)
        for vertex,items in enumerate(incident):
            table[vertex]=items+[items[0]]*(width-len(items))
        self.incident=torch.as_tensor(table,device=self.device)

    def weights(self,points,batch_size=4096):
        query=points.detach().cpu().numpy()
        nearest=self.tree.query(query,k=1,workers=4)[1][:,None]
        chosen_faces=[];chosen_weights=[];chosen_patches=[];no_inside=0
        for start in range(0,len(query),batch_size):
            stop=min(start+batch_size,len(query))
            p=points[start:stop]
            near=torch.as_tensor(nearest[start:stop],device=p.device)
            candidates=self.incident[near].reshape(len(p),-1)
            triangles=self.vertices[self.faces[candidates]]
            a=triangles[:,:,0];b=triangles[:,:,1];c=triangles[:,:,2]
            u=b-a;v=c-a
            normal=torch.cross(u,v,dim=-1)
            top=(normal*a).sum(-1)
            bottom=(normal*p[:,None,:]).sum(-1)
            bottom=torch.where(bottom.abs()<1e-8,torch.full_like(bottom,1e-8),bottom)
            q=p[:,None,:]*(top/bottom)[:,:,None]
            d=q-a
            uu=(u*u).sum(-1);uv=(u*v).sum(-1);vv=(v*v).sum(-1)
            du=(d*u).sum(-1);dv=(d*v).sum(-1)
            det=(uu*vv-uv*uv).clamp_min(1e-8)
            w1=(vv*du-uv*dv)/det
            w2=(uu*dv-uv*du)/det
            weights=torch.stack([1-w1-w2,w1,w2],-1)
            outside=torch.relu(-weights.min(-1).values)
            no_inside+=int((outside.min(1).values>1e-4).sum())
            residual=(p[:,None,:]-q).square().sum(-1)
            score=residual+1e6*outside.square()
            selected=score.argmin(1)
            row=torch.arange(len(p),device=p.device)
            face=candidates[row,selected]
            w=weights[row,selected]
            w=w.clamp_min(0)
            w=w/w.sum(-1,keepdim=True)
            chosen_faces.append(self.faces[face]);chosen_weights.append(w)
            chosen_patches.append(face)
        self.no_inside=no_inside
        return (torch.cat(chosen_faces),torch.cat(chosen_weights),
                torch.cat(chosen_patches))


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


def _face_costs(current,candidate,original,faces,layout,reference_map,reference_metric,lam,simval,components=False):
    index,valid,_,_,weights,source=layout
    bits=torch.as_tensor([[i>>2&1,i>>1&1,i&1] for i in range(8)],
                         device=current.device,dtype=torch.bool)
    fixed=current[faces]
    moved=candidate[faces]
    proposed=torch.where(bits[None,:,:,None],moved[:,None,:,:],fixed[:,None,:,:])
    sample_idx=index
    sample_weights=weights[sample_idx]
    xyz=F.normalize((sample_weights[:,None,:,:,None]*proposed[:,:,None,:,:]).sum(-2),dim=-1)*100
    target=reference_map.sample(xyz.reshape(-1,3),reference_metric).reshape(xyz.shape[:-1])
    native=source[sample_idx][:,None,:]
    observed=valid[:,None,:].float()
    count=observed.sum(-1).clamp_min(1)
    sx=(native*observed).sum(-1);sy=(target*observed).sum(-1)
    cov=(native*target*observed).sum(-1)-sx*sy/count
    vx=((native.square()*observed).sum(-1)-sx.square()/count).clamp_min(0)
    vy=((target.square()*observed).sum(-1)-sy.square()/count).clamp_min(0)
    corr=cov/(torch.sqrt(vx*vy)+1e-6)
    use=(valid.sum(-1)[:,None]>=4)&(vx>1e-5)&(vy>1e-5)
    if simval==1:
        similarity=torch.sqrt((((native-target)*observed).square()).sum(-1))/count
    else:
        similarity=torch.where(use,(1-corr)/2,0.5)

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
    shape=((trace/j).square()-4).clamp_min(0)
    strain=0.5*(0.4*shape+1.6*(j.square()+j.reciprocal().square()-2))
    signed=(torch.cross(u1,v1,dim=-1)*proposed[:,:,0,:]).sum(-1)
    signed0=(torch.cross(u0,v0,dim=-1)*old[:,:,0,:]).sum(-1)
    cost=similarity+lam*strain.square()+torch.where(signed/signed0<=0,1e4,0.)
    if components:
        return (similarity.detach().cpu().numpy(),
                strain.square().detach().cpu().numpy(),
                (signed/signed0<=0).detach().cpu().numpy())
    return cost.detach().cpu().numpy()


def _label_samples(grid,faces,max_distance):
    degree=np.bincount(faces.ravel(),minlength=len(grid))
    centre=grid[np.flatnonzero(degree==6)[0]]
    distance=np.linalg.norm(grid-centre,axis=1)
    selected=np.flatnonzero((distance>0)&(distance<=max_distance))
    selected=selected[np.argsort(distance[selected])]
    return centre,grid[selected]


def _rotated_label(positions,centre,sample,scale):
    selected=positions.device
    source=F.normalize(torch.as_tensor(centre,device=selected),dim=0)
    raw=torch.as_tensor(centre+(centre-sample)*scale,device=selected)
    label=F.normalize(raw,dim=0)
    target=F.normalize(positions,dim=1)
    crossing=torch.cross(source.expand_as(target),target,dim=1)
    cosine=(target*source).sum(1,keepdim=True)
    mapped=(label+torch.cross(crossing,label.expand_as(target),dim=1)+
            torch.cross(crossing,torch.cross(crossing,label.expand_as(target),dim=1),dim=1)/
            (1+cosine).clamp_min(1e-4))
    axis=F.normalize(torch.cross(source,torch.tensor([0.,0.,1.],device=selected),dim=0),dim=0)
    antipodal=2*(axis*label).sum()*axis-label
    mapped=torch.where(cosine< -0.999,antipodal,mapped)
    return F.normalize(mapped,dim=1)*100


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
    """Register both sulcal spheres and return native-order GIFTI paths."""
    from . import _fastpd_native
    if set(inputs) != {"L", "R"}:
        raise ValueError("inputs must contain L and R MSMSulc inputs")
    selected=torch.device(device)
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    output=Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    report={}
    for hemi in "LR":
        started=time.perf_counter()
        entry=inputs[hemi]
        native,native_faces=_surface(entry.rotated_sphere)
        reference_xyz,reference_faces=_surface(entry.reference_sphere)
        source_metric=np.asarray(nib.load(str(entry.native_sulc)).darrays[0].data,np.float32)
        reference_metric=np.asarray(nib.load(str(entry.reference_sulc)).darrays[0].data,np.float32)
        native_graph,_=_smoothing_graph(native_faces,len(native))
        reference_graph,_=_smoothing_graph(reference_faces,len(reference_xyz))
        _,affine,angles=_affine_initialization(
            native,reference_xyz,source_metric,reference_metric,
            native_graph,reference_graph,selected)
        native_positions=torch.as_tensor(native,device=selected)
        if selected.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected)
        prior_map=None
        prior_deformed=None
        stages=[]
        for level,data_level,lam in ((2,4,10.),(3,5,7.5),(4,6,7.5)):
            stage_started=time.perf_counter()
            regular_double,faces_np=_ico(level)
            data_double,data_faces=_ico(data_level)
            regular_np=regular_double.astype(np.float32)
            data_np=data_double.astype(np.float32)
            regular=torch.as_tensor(regular_np,device=selected)
            face_tensor=torch.as_tensor(faces_np,device=selected)
            if prior_map is None:
                cp_positions=regular.clone()
                source_positions=torch.as_tensor(data_np,device=selected)
            else:
                ids,w,_=prior_map.weights(torch.as_tensor(data_np,device=selected))
                source_positions=F.normalize((prior_deformed[ids]*w[:,:,None]).sum(1),dim=1)*100
                ids,w,_=prior_map.weights(regular)
                cp_positions=F.normalize((prior_deformed[ids]*w[:,:,None]).sum(1),dim=1)*100

            src=_adaptive_resample(native,native_faces,source_metric,data_np,data_faces,device=device)
            tgt=_adaptive_resample(reference_xyz,reference_faces,reference_metric,
                                   data_np,data_faces,device=device)
            source_values=torch.as_tensor(src,device=selected)
            source_values=(source_values-source_values.mean())/source_values.std()
            target_values=torch.as_tensor(tgt,device=selected)
            target_values=(target_values-target_values.mean())/target_values.std()
            target_map=SphereMap(data_np@affine,data_faces,selected)
            edges=np.unique(np.sort(np.concatenate((faces_np[:,[0,1]],
                faces_np[:,[1,2]],faces_np[:,[2,0]])),axis=1),axis=0)
            spacing=float(np.linalg.norm(regular_np[edges[:,0]]-regular_np[edges[:,1]],axis=1).max())
            centre,samples=_label_samples(data_double,data_faces,0.5*spacing)
            centre=centre.astype(np.float32)
            samples=samples.astype(np.float32)
            sorted_faces=np.sort(faces_np,axis=1).astype(np.int32)
            face_bytes=sorted_faces.tobytes()
            bits=np.asarray([[value>>2&1,value>>1&1,value&1] for value in range(8)],np.int8)
            order=np.empty((len(faces_np),8),np.int8)
            for face_id,triangle in enumerate(faces_np):
                order[face_id]=bits[:,np.argsort(np.argsort(triangle))]@np.array([4,2,1],np.int8)
            iterations=[]
            for iteration in range(8):
                iteration_started=time.perf_counter()
                prior=cp_positions.clone()
                current_map=RadialSphereMap(prior.cpu().numpy(),faces_np,selected)
                _,weights,patch=current_map.weights(source_positions)
                layout=_face_layout(faces_np,patch,weights,source_values,selected)
                labels=np.zeros(len(regular_np),np.int16)
                changed=0
                for _ in range(2):
                    for label,sample in enumerate([centre,*samples]):
                        if np.all(labels==label):
                            continue
                        candidate=_rotated_label(prior,centre,sample,0.8**iteration)
                        costs=_face_costs(cp_positions,candidate,regular,face_tensor,
                                          layout,target_map,target_values,lam,
                                          simval=1 if level==2 else 2)
                        ordered=np.take_along_axis(costs,order,axis=1).astype(np.float64)
                        choice=np.frombuffer(_fastpd_native.optimize(
                            face_bytes,ordered.tobytes(),len(regular_np)),dtype=np.uint8)
                        update=(choice==1)&(labels!=label)
                        if update.any():
                            mask=torch.as_tensor(update,device=selected)
                            cp_positions[mask]=candidate[mask]
                            labels[update]=label
                            changed+=int(update.sum())
                for key,positions in (("source",source_positions),("native",native_positions)):
                    ids,w,_=current_map.weights(positions)
                    warped=F.normalize((cp_positions[ids]*w[:,:,None]).sum(1),dim=1)*100
                    if key=="source":
                        source_positions=warped
                    else:
                        native_positions=warped
                iterations.append({"changed":changed,"seconds":time.perf_counter()-iteration_started})
            prior_map=RadialSphereMap(regular_np,faces_np,selected)
            prior_deformed=cp_positions
            stages.append({"control_points":len(regular_np),"data_points":len(data_np),
                           "labels":len(samples)+1,"iterations":iterations,
                           "seconds":time.perf_counter()-stage_started})

        vertices=native_positions.cpu().numpy().astype(np.float32)@affine.T
        vertices,folded_before,max_repair_mm=_repair_folds(
            native,vertices,native_faces,device)
        path=output/f"{hemi}.sphere.MSMSulc.native.surf.gii"
        nib.save(nib.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(vertices,intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(native_faces,intent="NIFTI_INTENT_TRIANGLE")]),str(path))
        report[hemi]={"seconds":time.perf_counter()-started,
                      "affine_angles_deg":angles,"folded_before_repair":folded_before,
                      "folded_after_repair":0,"maximum_repair_displacement_mm":max_repair_mm,
                      "peak_allocated_gb":(torch.cuda.max_memory_allocated(selected)/1e9
                                           if selected.type=="cuda" else None),"stages":stages}
    (output/"registration_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return {hemi:output/f"{hemi}.sphere.MSMSulc.native.surf.gii" for hemi in "LR"}
