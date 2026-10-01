"""Source-derived newMSM rigid initialization for a single sulcal feature."""

from pathlib import Path
import math

import nibabel as nib
import numpy as np
import torch

from ._sphere_map import _dot, _cross, _normalize


def _surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    points = np.asarray(next(a.data for a in image.darrays if a.intent == 1008),dtype=np.float64)
    faces = np.asarray(next(a.data for a in image.darrays if a.intent == 1009),dtype=np.int64)
    if (points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2
            or faces.shape[1] != 3 or not np.isfinite(points).all()
            or faces.min()<0 or faces.max()>=len(points)):
        raise ValueError(f"invalid sphere: {path}")
    return points, faces


def _source_euler_matrix(angles,device):
    """The source CPU/libm Euler constants, transferred once per trial."""
    a,b,c=angles
    ca,cb,cc=math.cos(a),math.cos(b),math.cos(c)
    sa,sb,sc=math.sin(a),math.sin(b),math.sin(c)
    return torch.tensor(((cb*cc,-ca*sc+sa*sb*cc,sa*sc+ca*sb*cc),
                         (cb*sc,ca*cc+sa*sb*sc,-sa*cc+ca*sb*sc),
                         (-sb,sa*cb,ca*cb)),dtype=torch.float64,device=device)


def _point_matmul(points,matrix):
    """Three products and two additions in newresampler Point order."""
    first=points[...,0,None]*matrix[0]
    second=points[...,1,None]*matrix[1]
    third=points[...,2,None]*matrix[2]
    return (first+second)+third


def _normal_incident(faces,count,device):
    incident=[[] for _ in range(count)]
    for face_id,triangle in enumerate(faces):
        for vertex in triangle:incident[vertex].append(face_id)
    width=max(map(len,incident))
    table=np.zeros((count,width),dtype=np.int64)
    valid=np.zeros((count,width),dtype=bool)
    for vertex,items in enumerate(incident):
        table[vertex,:len(items)]=items;valid[vertex,:len(items)]=True
    return torch.as_tensor(table,device=device),torch.as_tensor(valid,device=device)


def _local_normals(vertices,faces,incident=None):
    triangles=vertices[faces]
    normals=_normalize(_cross(triangles[:,2]-triangles[:,0],triangles[:,1]-triangles[:,0]))
    if incident is None:
        incident=_normal_incident(faces.detach().cpu().numpy(),len(vertices),vertices.device)
    table,valid=incident
    total=torch.zeros_like(vertices)
    for index in range(table.shape[1]):
        total=total+torch.where(valid[:,index,None],normals[table[:,index]],0)
    total=_normalize(total)
    return torch.where(_dot(total,vertices)[:,None]<0,-total,total)


def _tangent_basis(normals):
    """Source calculate_tangs branch order and Point normalization."""
    x,y,z=normals.unbind(-1)
    first=(x.abs()>=y.abs())&(x.abs()>=z.abs())
    second=(~first)&(y.abs()>=x.abs())&(y.abs()>=z.abs())
    third=~(first|second)
    yz=torch.sqrt(z*z+y*y);xz=torch.sqrt(z*z+x*x);xy=torch.sqrt(y*y+x*x)
    yz_safe=torch.where(yz==0,1,yz)
    xz_safe=torch.where(xz==0,1,xz)
    xy_safe=torch.where(xy==0,1,xy)
    zero=torch.zeros_like(x)
    e1=torch.stack((torch.where(second,-z/xz_safe,torch.where(third,-y/xy_safe,zero)),
                    torch.where(first,-z/yz_safe,torch.where(third,x/xy_safe,zero)),
                    torch.where(first,y/yz_safe,torch.where(second,x/xz_safe,zero))),-1)
    zero_mag=(first&(yz==0))|(second&(xz==0))
    e1=torch.where(zero_mag[:,None],torch.tensor([0.,0.,1.],device=normals.device),e1)
    e1=torch.where((third&(xy==0))[:,None],torch.tensor([1.,0.,0.],device=normals.device),e1)
    return e1,_normalize(_cross(normals,e1))


def _mean_neighbor_distance(vertices,faces):
    """Mesh.calculate_MeanVD visits directed neighbors in insertion order."""
    neighbors=[[] for _ in vertices]
    for a,b,c in faces:
        for vertex,others in ((a,(b,c)),(b,(a,c)),(c,(a,b))):
            for other in others:
                if other not in neighbors[vertex]:neighbors[vertex].append(other)
    first=np.repeat(np.arange(len(vertices)),[len(items) for items in neighbors])
    second=np.array([other for items in neighbors for other in items])
    delta=vertices[second]-vertices[first]
    squares=delta*delta
    distance=np.sqrt((squares[:,0]+squares[:,1])+squares[:,2])
    return float(np.add.accumulate(distance)[-1]/len(distance))


class _RigidCost:
    def __init__(self,vertices,faces,source,reference,device,simval=1,execution="optimized"):
        from .msmsulc import RadialSphereMap
        from . import _fastpd_native
        self.wls_cost=_fastpd_native.source_wls_cost
        self.vertices=torch.as_tensor(vertices,dtype=torch.float64,device=device)
        self.faces=torch.as_tensor(faces,dtype=torch.long,device=device)
        self.source=torch.as_tensor(source,dtype=torch.float64,device=device)
        self.reference=torch.as_tensor(reference,dtype=torch.float64,device=device)
        self.simval=2 if simval==3 else simval
        # Source meanvector accumulates scalar features in original vertex order.
        source_mean=np.add.accumulate(np.asarray(source,dtype=np.float64))[-1]/len(source)
        reference_mean=np.add.accumulate(np.asarray(reference,dtype=np.float64))[-1]/len(reference)
        self.centered_source=self.source-source_mean
        self.centered_reference=self.reference-reference_mean
        self.mapper=RadialSphereMap(vertices,faces,device,execution=execution)
        self.normal_incident=_normal_incident(faces,len(vertices),device)
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
        self.sigma=_mean_neighbor_distance(vertices,faces)

    def __call__(self,rotation):
        return self.evaluate_positions(_point_matmul(self.vertices,rotation))

    def evaluate_positions(self,positions):
        normals=_local_normals(positions,self.faces,self.normal_incident)
        _,_,face=self.mapper.weights(positions)
        ids=self.ids[face]
        e1,e2=_tangent_basis(normals)
        origin=_normalize(_cross(e1,e2))*100
        source_delta=positions-origin
        target_delta=self.vertices[ids]-origin[:,None,:]
        first=_dot(target_delta,e1[:,None,:])-_dot(source_delta,e1)[:,None]
        second=_dot(target_delta,e2[:,None,:])-_dot(source_delta,e2)[:,None]
        tangent_sq=first*first+second*second
        # Sparse similarity generation skips reference vertex ID zero.
        if self.simval==1:
            similarity=-(self.reference[ids]-self.source[:,None]).abs()
        else:
            similarity=torch.sign(self.centered_reference[ids]*self.centered_source[:,None])
        similarity=torch.where(ids!=0,similarity,0)
        # The source combines literal double division, CPU/libm exp, and
        # ordered neighbor/vertex sums. Their coupled last-bit differences
        # change later shared-edge face assignment in the rigid optimizer.
        # Geometry stays on the device; this packed buffer crosses the one
        # host boundary already required to return the optimizer's cost.
        payload=torch.stack((tangent_sq,similarity,self.valid[face].to(tangent_sq.dtype)),dim=-1)
        return self.wls_cost(payload.detach().cpu().numpy(),len(positions),ids.shape[1],self.sigma)


def _affine_initialization(vertices,faces,source,reference,device,config,execution="optimized",*,return_positions=False):
    """Run the original finite-difference update and rejection sequence.

    newMSM's historical NMI option 3 becomes pairwise Pearson option 2. A
    single sulcal feature is centered by its spatial mean before pairwise
    Pearson evaluation, so the pair term is sign(centered_a*centered_b).
    """
    rotation=torch.eye(3,dtype=torch.float64,device=device)
    cost=_RigidCost(vertices,faces,source,reference,device,config.simval[0],execution)
    positions=cost.vertices.clone()
    zero=cost.evaluate_positions(positions);initial=zero;best=zero;min_iter=0;loop=0;evaluations=1
    spacing=config.affine_gradient_spacing
    while spacing>0.05:
        step=config.affine_step_size
        for iteration in range(1,config.iterations[0]+1):
            trial=[_source_euler_matrix(angles,device) for angles in
                   ((spacing,0.,0.),(0.,spacing,0.),(0.,0.,spacing))]
            values=np.asarray([cost.evaluate_positions(_point_matmul(positions,candidate))
                               for candidate in trial])
            evaluations+=3
            gradient=(values-zero)/spacing
            norm=math.sqrt((gradient[0]*gradient[0]+gradient[1]*gradient[1])+gradient[2]*gradient[2])
            if norm>1e-8:gradient/=norm
            delta=_source_euler_matrix(step*gradient,device)
            previous_positions=positions
            previous_rotation=rotation
            positions=_point_matmul(positions,delta)
            rotation=_point_matmul(rotation,delta)
            # The source evaluates another identical Euler increment after
            # temporarily rotating SOURCE. Preserve this original behavior.
            zero=cost.evaluate_positions(_point_matmul(positions,delta));evaluations+=1
            global_iteration=loop*config.iterations[0]+iteration
            if zero>best:
                best=zero;min_iter=global_iteration
            if global_iteration-min_iter>0:
                step*=0.5;positions=previous_positions;rotation=previous_rotation
            if step<1e-3:break
        loop+=1;spacing*=0.5
    matrix=rotation.detach().cpu().numpy()
    angles=[np.arctan2(matrix[2,1],matrix[2,2]),
            np.arcsin(np.clip(-matrix[2,0],-1,1)),np.arctan2(matrix[1,0],matrix[0,0])]
    result=(matrix,(np.asarray(angles)*180/np.pi).tolist(),{
        "cost_evaluations":evaluations,"initial_similarity":initial,
        "final_best_similarity":best,"constant_similarity":False})
    # Source rotates mesh coordinates after each accepted update. Reapplying
    # one accumulated matrix loses that rounding history at shared mesh edges.
    return (*result,positions) if return_positions else result
