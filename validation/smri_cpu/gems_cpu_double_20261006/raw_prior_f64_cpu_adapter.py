"""Private bounded CPU precision controls; no production/GPU source edits.

Coordinate/gradient storage is Double. Ownership alone follows the captured
FP32 geometry and compact lookup. Projection and Gaussian precision are separate
declared controls. All raw mass, epsilon, priors and source definitions are fixed.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import torch


path=Path(os.environ['FNIT_GEMS_RAW_ADAPTER'])
if hashlib.sha256(path.read_bytes()).hexdigest()!='fd6a6cd80c0cfe81df9c77c3694f328d75b0e75c5978280fb1efd106d846f675':
    raise RuntimeError('actual raw-prior closure required')
spec=importlib.util.spec_from_file_location('fixed_raw_cpu_parent',path)
raw=importlib.util.module_from_spec(spec);sys.modules[spec.name]=raw;spec.loader.exec_module(raw)
core=raw.core
rasterize=raw.rasterize
base=raw._base


def raw_f64_priors(points,tetrahedra,alphas,shape,*,valid_mask,block_index,
                   background_channel,current_geometry,owner_geometry):
    if points.device.type!='cpu' or points.dtype!=torch.float64:
        raise ValueError('private Double CPU coordinate storage required')
    # Owner scan and fixed integer voxel grid use precisely the old FP32 path;
    # all differentiable origins/inverses still refer to the Double leaf.
    owners=points.detach().float()
    v0,inverse,selected,voxel_points,covered,reorder=rasterize._compact_lookup(
        owners,tetrahedra,valid_mask,block_index,current_geometry,
        owner_geometry=owner_geometry)
    if selected.numel():
        cells=tetrahedra[selected]
        reduction=base.prepare_vertex_reduction(selected,len(v0))
        origins=rasterize.ordered_row_gather(v0,selected,reduction)
        inverses=rasterize.ordered_row_gather(inverse,selected,reduction)
        w123=torch.einsum('pij,pj->pi',inverses,voxel_points-origins)
        weights=torch.cat([1.-w123.sum(-1,keepdim=True),w123],dim=-1)
        values=(alphas[cells]*weights[...,None]).sum(1).clamp_min(0)
        values=torch.where(covered[...,None],values,0)
    else:
        values=torch.zeros((0,alphas.shape[1]),dtype=torch.float64)
    out=torch.cat([values,torch.zeros((1,alphas.shape[1]),dtype=values.dtype)])[reorder]
    mask=torch.cat([covered,torch.zeros(1,dtype=torch.bool)])[reorder]
    out[:,background_channel]=torch.where(~mask,torch.ones_like(out[:,background_channel]),out[:,background_channel])
    return out.T.contiguous(),mask


class SharedClosure(raw.SharedClosure):
    def __init__(self,arrays,background):
        super().__init__(arrays,background)
        self.start=self.start.double();self.anchor=self.start.clone()
        projection_mode=os.environ.get('FNIT_GEMS_F64_PROJECTION','0')
        gaussian_mode=os.environ.get('FNIT_GEMS_F64_GAUSSIAN','0')
        if projection_mode not in {'0','1'} or gaussian_mode not in {'0','1'}:
            raise ValueError('explicit precision control 0/1 required')
        if projection_mode=='1':
            self.projection=base.sliding_boundary_projectors(self.can_move,self.boundary.double())
        else:self.projection=self.projection.double()
        if gaussian_mode=='1':
            self.parameters=base.GaussianParameters(self.parameters.means.double(),self.parameters.covariances.double())
            self.likelihood=base.gaussian_log_likelihood(
                self.image[self.valid].double().reshape(-1,1,1),self.parameters).reshape(self.alphas.shape[1],-1).detach()
        self.precision_policy={'points':str(self.start.dtype),'gradient_leaf':str(self.start.dtype),
            'owner_geometry_and_scan':'torch.float32','projection_matrix_storage':str(self.projection.dtype),
            'projection_computed_in':'torch.float64' if projection_mode=='1' else 'torch.float32_then_promoted',
            'Gaussian_likelihood':str(self.likelihood.dtype),'optimizer_state':'torch.float64',
            'raw_mass':True,'mixture_epsilon':1e-15,'GPU_source_or_default_changed':False}

    def evaluate(self,supplied,return_priors=False):
        self.evaluations+=1
        points=supplied.detach().clone().requires_grad_(True)
        if points.device.type!='cpu' or points.dtype!=torch.float64:
            raise RuntimeError('bounded Double CPU coordinate storage required')
        if (points.detach()-self.anchor).abs().amax().item()>1.5:
            self.index=rasterize.build_block_index(points.detach().numpy(),self.tetra.numpy(),self.shape,8,margin=3.)
            self.anchor=points.detach().clone();self.rebuilds+=1
        geometry=base.prepare_current_geometry(points,self.tetra,
            deterministic_gradient=True,vertex_reduction=self.reduction)
        owner=base.prepare_current_geometry(points.detach().float(),self.tetra)
        priors,coverage=raw_f64_priors(points,self.tetra,self.alphas,self.shape,
            valid_mask=self.valid,block_index=self.index,background_channel=self.background,
            current_geometry=geometry,owner_geometry=owner)
        data_cost=core._compact_mesh_data_cost(priors,self.likelihood,double_accumulation=True)
        prior_cost,jacobian=base.ashburner_prior(points,self.reference,self.tetra,self.stiffness,
            reference_geometry=self.reference_geometry,current_geometry=geometry,
            analytic_gradient=True,double_accumulation=True)
        objective=torch.where(torch.isfinite(jacobian).all() & (jacobian>0).all(),
                              data_cost+prior_cost,data_cost.new_tensor(float('inf')))
        objective.backward()
        points.grad.copy_(torch.bmm(self.projection,points.grad[...,None]).squeeze(-1))
        self.last={'data_cost':float(data_cost),'prior_cost':float(prior_cost),
            'min_jacobian':float(jacobian.min()),'covered_voxels':int(coverage.sum()),'index_rebuilds':self.rebuilds}
        result=objective.detach(),points.grad.detach().clone()
        return (*result,priors.detach(),coverage.detach()) if return_priors else result
