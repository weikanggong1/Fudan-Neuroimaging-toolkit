"""Private CPU mesh-likelihood closure with native raw atlas mass semantics.

This diagnostic changes only the interpolated prior normalization inside the
synthetic CPU mesh objective. Public rasterization, EM, probabilities and every
GPU code path retain the frozen implementation.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import torch


_base_path = Path(os.environ['FNIT_GEMS_FROZEN_ADAPTER'])
if hashlib.sha256(_base_path.read_bytes()).hexdigest() != 'a996995feed38f49b53498a8a6e10e49cfba0ee86ceec176cb9267a67ff841f4':
    raise RuntimeError('exact captured closure adapter required')
_spec = importlib.util.spec_from_file_location('raw_prior_frozen_base', _base_path)
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
core = _base.core
rasterize = _base.rasterize


def raw_cpu_priors(points, tetrahedra, alphas, shape, *, valid_mask, block_index,
                   background_channel, current_geometry, owner_geometry):
    if points.device.type != 'cpu' or points.dtype != torch.float32:
        raise ValueError('private raw CPU mesh priors require CPU FP32 points')
    all_v0, all_inv, selected_ids, selected_points, covered, reorder = rasterize._compact_lookup(
        points, tetrahedra, valid_mask, block_index, current_geometry,
        owner_geometry=owner_geometry)
    if selected_ids.numel():
        cells = tetrahedra[selected_ids]
        reduction = _base.prepare_vertex_reduction(selected_ids, len(all_v0))
        v0 = rasterize.ordered_row_gather(all_v0, selected_ids, reduction)
        inverse = rasterize.ordered_row_gather(all_inv, selected_ids, reduction)
        w123 = torch.einsum('pij,pj->pi', inverse, selected_points-v0)
        weights = torch.cat([1.-w123.sum(-1, keepdim=True), w123], dim=-1)
        # Preserve atlas mass. KVL sums mixture*interpolatedAlpha directly;
        # division by sum changes its derivative around zero-alpha support.
        values = (alphas[cells]*weights[..., None]).sum(1).clamp_min(0)
        values = torch.where(covered[..., None], values, 0)
    else:
        values = torch.zeros((0, alphas.shape[1]), dtype=alphas.dtype)
    out = torch.cat([values, torch.zeros((1, alphas.shape[1]), dtype=values.dtype)])[reorder]
    mask = torch.cat([covered, torch.zeros(1, dtype=torch.bool)])[reorder]
    out[:, background_channel] = torch.where(~mask, torch.ones_like(out[:, background_channel]), out[:, background_channel])
    return out.T.contiguous(), mask


class SharedClosure(_base.SharedClosure):
    """Same image/alphas/reference, ownership, epsilon and projection as capture."""
    def evaluate(self, supplied, return_priors=False):
        self.evaluations += 1
        points = supplied.detach().clone().requires_grad_(True)
        if points.device.type != 'cpu' or points.dtype != torch.float32:
            raise RuntimeError('private CPU FP32 point storage required')
        if (points.detach()-self.anchor).abs().amax().item() > 1.5:
            self.index = rasterize.build_block_index(points.detach().numpy(), self.tetra.numpy(), self.shape, 8, margin=3.)
            self.anchor = points.detach().clone()
            self.rebuilds += 1
        geometry = _base.prepare_current_geometry(points.double(), self.tetra,
            deterministic_gradient=True, vertex_reduction=self.reduction)
        owner = _base.prepare_current_geometry(points.detach(), self.tetra)
        priors, coverage = raw_cpu_priors(points, self.tetra, self.alphas, self.shape,
            valid_mask=self.valid, block_index=self.index, background_channel=self.background,
            current_geometry=geometry, owner_geometry=owner)
        data_cost = core._compact_mesh_data_cost(priors, self.likelihood, double_accumulation=True)
        prior_cost, jacobian = _base.ashburner_prior(points, self.reference, self.tetra, self.stiffness,
            reference_geometry=self.reference_geometry, current_geometry=geometry,
            analytic_gradient=True, double_accumulation=True)
        objective = data_cost+prior_cost
        objective = torch.where(torch.isfinite(jacobian).all() & (jacobian>0).all(),
                                objective, objective.new_tensor(float('inf')))
        objective.backward()
        points.grad.copy_(torch.bmm(self.projection, points.grad[..., None]).squeeze(-1))
        self.last = {'data_cost':float(data_cost), 'prior_cost':float(prior_cost),
            'min_jacobian':float(jacobian.min()), 'covered_voxels':int(coverage.sum()),
            'index_rebuilds':self.rebuilds}
        result = objective.detach(), points.grad.detach().clone()
        return (*result, priors.detach(), coverage.detach()) if return_priors else result
