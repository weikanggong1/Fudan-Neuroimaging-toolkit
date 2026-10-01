"""Source-derived newMSM rigid initialization for a single sulcal feature."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def _surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    points = np.asarray(next(a.data for a in image.darrays if a.intent == 1008),dtype=np.float64)
    faces = np.asarray(next(a.data for a in image.darrays if a.intent == 1009),dtype=np.int64)
    if (points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2
            or faces.shape[1] != 3 or not np.isfinite(points).all()
            or faces.min()<0 or faces.max()>=len(points)):
        raise ValueError(f"invalid sphere: {path}")
    return points, faces


def _euler_matrix(angles):
    """Row-coordinate convention of newresampler::euler_rotate."""
    a,b,c=angles.unbind(-1)
    ca,cb,cc=torch.cos(a),torch.cos(b),torch.cos(c)
    sa,sb,sc=torch.sin(a),torch.sin(b),torch.sin(c)
    return torch.stack((cb*cc,-ca*sc+sa*sb*cc,sa*sc+ca*sb*cc,
                        cb*sc,ca*cc+sa*sb*sc,-sa*cc+ca*sb*sc,
                        -sb,sa*cb,ca*cb),-1).reshape(*angles.shape[:-1],3,3)


def _local_normals(vertices,faces):
    triangles=vertices[faces]
    normals=F.normalize(torch.cross(triangles[:,2]-triangles[:,0],
                                    triangles[:,1]-triangles[:,0],dim=-1),dim=-1)
    total=torch.zeros_like(vertices)
    for corner in range(3):
        total.index_add_(0,faces[:,corner],normals)
    total=F.normalize(total,dim=-1)
    return torch.where((total*vertices).sum(-1,keepdim=True)<0,-total,total)


class _RigidCost:
    def __init__(self,vertices,faces,source,reference,device,simval=1,execution="optimized"):
        from .msmsulc import RadialSphereMap
        self.vertices=torch.as_tensor(vertices,dtype=torch.float64,device=device)
        self.faces=torch.as_tensor(faces,dtype=torch.long,device=device)
        self.source=torch.as_tensor(source,dtype=torch.float64,device=device)
        self.reference=torch.as_tensor(reference,dtype=torch.float64,device=device)
        self.simval=2 if simval==3 else simval
        self.centered_source=self.source-self.source.mean()
        self.centered_reference=self.reference-self.reference.mean()
        self.mapper=RadialSphereMap(vertices,faces,device,execution=execution)
        self.normals=_local_normals(self.vertices,self.faces)
        incident=[[] for _ in vertices]
        for triangle in faces:
            for vertex in triangle:
                for other in triangle:
                    if other not in incident[vertex]:incident[vertex].append(other)
        patches=[]
        for triangle in faces:
            members=[]
            for vertex in triangle:
                for other in incident[vertex]:
                    if other not in members:members.append(other)
            patches.append(members)
        width=max(map(len,patches))
        ids=np.zeros((len(faces),width),np.int64)
        mask=np.zeros_like(ids,dtype=bool)
        for i,patch in enumerate(patches):ids[i,:len(patch)]=patch;mask[i,:len(patch)]=True
        self.ids=torch.as_tensor(ids,device=device)
        self.valid=torch.as_tensor(mask,device=device)
        edges=np.concatenate((faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]]))
        edges=np.unique(np.sort(edges,axis=1),axis=0)
        self.sigma=float(np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1).mean())

    def __call__(self,rotation):
        positions=self.vertices@rotation
        normals=self.normals@rotation
        _,_,face=self.mapper.weights(positions)
        ids=self.ids[face]
        delta=self.vertices[ids]-positions[:,None,:]
        tangent_sq=(delta.square().sum(-1)-(delta*normals[:,None,:]).sum(-1).square()).clamp_min(0)
        weight=torch.exp(-tangent_sq/(2*self.sigma*self.sigma))*self.valid[face]*(tangent_sq>0)
        # Sparse similarity generation skips reference vertex ID zero.
        if self.simval==1:
            similarity=-(self.reference[ids]-self.source[:,None]).abs()
        else:
            similarity=torch.sign(self.centered_reference[ids]*self.centered_source[:,None])
        similarity=torch.where(ids!=0,similarity,0)
        result=(weight*similarity).sum(-1)/weight.sum(-1)
        return float(result.sum().detach().cpu())


def _affine_initialization(vertices,faces,source,reference,device,config,execution="optimized"):
    """Run the original finite-difference update and rejection sequence.

    newMSM's historical NMI option 3 becomes pairwise Pearson option 2. A
    single sulcal feature is centered by its spatial mean before pairwise
    Pearson evaluation, so the pair term is sign(centered_a*centered_b).
    """
    rotation=torch.eye(3,dtype=torch.float64,device=device)
    cost=_RigidCost(vertices,faces,source,reference,device,config.simval[0],execution)
    zero=cost(rotation);initial=zero;best=zero;min_iter=0;loop=0;evaluations=1
    spacing=config.affine_gradient_spacing
    while spacing>0.05:
        step=config.affine_step_size
        for iteration in range(1,config.iterations[0]+1):
            finite=torch.eye(3,dtype=torch.float64,device=device)*spacing
            trial=_euler_matrix(finite)
            values=np.asarray([cost(rotation@candidate) for candidate in trial])
            evaluations+=3
            gradient=torch.as_tensor((values-zero)/spacing,dtype=torch.float64,device=device)
            norm=torch.linalg.vector_norm(gradient)
            gradient=gradient/torch.where(norm>1e-8,norm,torch.ones_like(norm))
            delta=_euler_matrix(step*gradient)
            previous=rotation
            rotation=rotation@delta
            # The source evaluates another identical Euler increment after
            # temporarily rotating SOURCE. Preserve this original behavior.
            zero=cost(rotation@delta);evaluations+=1
            global_iteration=loop*config.iterations[0]+iteration
            if zero>best:
                best=zero;min_iter=global_iteration
            if global_iteration-min_iter>0:
                step*=0.5;rotation=previous
            if step<1e-3:break
        loop+=1;spacing*=0.5
    matrix=rotation.detach().cpu().numpy()
    angles=[np.arctan2(matrix[2,1],matrix[2,2]),
            np.arcsin(np.clip(-matrix[2,0],-1,1)),np.arctan2(matrix[1,0],matrix[0,0])]
    return matrix,(np.asarray(angles)*180/np.pi).tolist(),{
        "cost_evaluations":evaluations,"initial_similarity":initial,
        "final_best_similarity":best,"constant_similarity":False}
