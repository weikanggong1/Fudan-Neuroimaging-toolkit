"""Same-point CPU decomposition; local diagnostic variants never enter FNIT.

All full images, priors, owner IDs and gradients stay in the private output.
The public result contains only hashes, errors, counts and source bindings.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
from numba import njit, prange, set_num_threads
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    delta = left-right
    norm = np.linalg.norm(right)
    return {'max_abs': float(np.abs(delta).max()), 'rmse': float(np.sqrt(np.mean(delta*delta))),
            'relative_l2': float(np.linalg.norm(delta)/norm) if norm else None}


@njit(cache=True, parallel=True, fastmath=False)
def diagnostic_lookup64(points, blocks, offsets, candidates, default, origins, inverses, singular, tolerance, order):
    selected = np.empty(len(points), dtype=np.int64)
    covered = np.zeros(len(points), dtype=np.bool_)
    for work in prange(len(points)):
        row = order[work]
        block = blocks[row]
        owner, best = default[block], -np.inf
        for index in range(offsets[block], offsets[block+1]):
            cell = candidates[index]
            if singular[cell]:
                continue
            dx, dy, dz = points[row,0]-origins[cell,0], points[row,1]-origins[cell,1], points[row,2]-origins[cell,2]
            w1 = (inverses[cell,0,0]*dx+inverses[cell,0,1]*dy)+inverses[cell,0,2]*dz
            w2 = (inverses[cell,1,0]*dx+inverses[cell,1,1]*dy)+inverses[cell,1,2]*dz
            w3 = (inverses[cell,2,0]*dx+inverses[cell,2,1]*dy)+inverses[cell,2,2]*dz
            w0 = 1.-((w1+w2)+w3)
            score = min(w0,w1,w2,w3)
            if score > best:
                best, owner = score, cell
        selected[row], covered[row] = owner, best >= -tolerance
    return selected, covered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--baseline-python', type=Path, required=True)
    parser.add_argument('--candidate-python', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    if stat.S_IMODE(args.output.stat().st_mode) != 0o700:
        raise RuntimeError('private arrays require actual0700')
    torch.set_num_threads(8)
    set_num_threads(8)
    if digest(args.adapter) != 'a996995feed38f49b53498a8a6e10e49cfba0ee86ceec176cb9267a67ff841f4':
        raise RuntimeError('frozen real closure adapter changed')
    spec = importlib.util.spec_from_file_location('frozen_real_closure', args.adapter)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    from fnit.gems import _raster_cpu_compact, rasterize
    from fnit.gems.deformation import prepare_current_geometry, prepare_vertex_reduction, ordered_row_gather, ashburner_prior
    from fnit.gems.gaussian import GaussianParameters, gaussian_log_likelihood
    capture = json.loads((args.capture/'report.public.json').read_text())
    for name, expected in capture['outputs'].items():
        if digest(args.capture/name) != expected:
            raise RuntimeError('bound capture changed')
    source = Path(adapter.core.__file__).parent
    if {str(p.relative_to(source)):digest(p) for p in source.rglob('*.py')} != capture['source']:
        raise RuntimeError('actual frozen source changed')
    with np.load(args.capture/'shared_input.npz', allow_pickle=False) as loaded:
        arrays = {key:loaded[key].copy() for key in loaded.files}
    results = {}
    states = [('existing_armijo_3',args.baseline_python,'existing_armijo'),
              ('native_definition_cpu_v2_3',args.candidate_python,'native_definition_cpu')]
    for name, run, mode in states:
        python_report = json.loads((run/'summary.public.json').read_text())
        info = python_report['modes'][mode]
        accepted = run/mode/'accepted-003.private.npz'
        with np.load(accepted, allow_pickle=False) as loaded:
            point_array, expected_gradient = loaded['points'].copy(), loaded['gradient'].copy()
        with np.load(args.native/(name+'.private.npz'), allow_pickle=False) as loaded:
            native = {key:loaded[key].copy() for key in loaded.files}
        if not np.array_equal(point_array,native['points']):
            raise RuntimeError('native decomposition point differs')
        native_report = json.loads((args.native/'summary.public.json').read_text())['same_points'][name]
        if (native_report['python_state_file_sha256'] != digest(accepted)
                or native_report['native_private_state_file_sha256'] != digest(args.native/(name+'.private.npz'))):
            raise RuntimeError('native/accepted point hashes changed')
        baseline_owner = baseline_coverage = None
        variants = {}
        for variant in ['baseline','owner64','tolerance_zero','no_prior_normalization','gaussian64','epsilon_zero']:
            closure = adapter.SharedClosure(arrays,capture['background_class'])
            points = torch.from_numpy(point_array.copy()).requires_grad_(True)
            if (points.detach()-closure.anchor).abs().amax().item()>1.5:
                closure.index=rasterize.build_block_index(points.detach().numpy(),closure.tetra.numpy(),closure.shape,8,margin=3.)
            geometry = prepare_current_geometry(points.double(),closure.tetra,
                deterministic_gradient=True,vertex_reduction=closure.reduction)
            owner = prepare_current_geometry(points.detach(),closure.tetra)
            original_lookup = _raster_cpu_compact.lookup_compact_cpu
            if variant=='owner64':
                def lookup64(batches,origins,inverses,singular,cache,*,tolerance=2e-5):
                    plan = _raster_cpu_compact._build_plan(batches,len(singular))
                    selected, covered = diagnostic_lookup64(plan.points.numpy().astype(np.float64),
                        plan.point_blocks,plan.offsets,plan.candidates,plan.default_owners,
                        origins.detach().numpy(),inverses.detach().numpy(),singular.numpy(),float(tolerance),plan.work_order(8))
                    return torch.from_numpy(selected),plan.points,torch.from_numpy(covered)
                _raster_cpu_compact.lookup_compact_cpu=lookup64
            try:
                all_v0,all_inv,selected_ids,selected_points,covered,reorder=rasterize._compact_lookup(
                    points,closure.tetra,closure.valid,closure.index,geometry,
                    tolerance=0. if variant=='tolerance_zero' else 2e-5,
                    owner_geometry=None if variant=='owner64' else owner)
            finally:
                _raster_cpu_compact.lookup_compact_cpu=original_lookup
            cells=closure.tetra[selected_ids]
            reduction=prepare_vertex_reduction(selected_ids,len(all_v0))
            v0=ordered_row_gather(all_v0,selected_ids,reduction)
            inverse=ordered_row_gather(all_inv,selected_ids,reduction)
            w123=torch.einsum('pij,pj->pi',inverse,selected_points-v0)
            weights=torch.cat([1.-w123.sum(-1,keepdim=True),w123],dim=-1)
            raw=(closure.alphas[cells]*weights[...,None]).sum(1)
            values=raw.clamp_min(0)
            if variant!='no_prior_normalization':
                values=values/values.sum(-1,keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
            values=torch.where(covered[...,None],values,0)
            out=torch.cat([values,torch.zeros((1,values.shape[1]),dtype=values.dtype)])[reorder]
            mask=torch.cat([covered,torch.zeros(1,dtype=torch.bool)])[reorder]
            out[:,closure.background]=torch.where(~mask,torch.ones_like(out[:,closure.background]),out[:,closure.background])
            priors=out.T.contiguous()
            likelihood=closure.likelihood
            if variant=='gaussian64':
                parameters=GaussianParameters(closure.parameters.means.double(),closure.parameters.covariances.double())
                likelihood=gaussian_log_likelihood(closure.image[closure.valid].double().reshape(-1,1,1),parameters).reshape(closure.alphas.shape[1],-1)
            if variant=='epsilon_zero':
                density=(priors.clamp_min(torch.finfo(priors.dtype).tiny).log()+likelihood).logsumexp(dim=0)
                data_cost=-density.sum(dtype=torch.float64)
            else:
                data_cost=adapter.core._compact_mesh_data_cost(priors,likelihood,double_accumulation=True)
            prior_cost,jacobian=ashburner_prior(points,closure.reference,closure.tetra,closure.stiffness,
                reference_geometry=closure.reference_geometry,current_geometry=geometry,
                analytic_gradient=True,double_accumulation=True)
            data_gradient=torch.autograd.grad(data_cost,points,retain_graph=True)[0]
            prior_gradient=torch.autograd.grad(prior_cost,points,retain_graph=True)[0]
            total_gradient=torch.autograd.grad(data_cost+prior_cost,points)[0]
            projected=lambda gradient:torch.bmm(closure.projection,gradient[...,None]).squeeze(-1)
            total_projected,data_projected,prior_projected=map(projected,[total_gradient,data_gradient,prior_gradient])
            owner_mask=torch.cat([selected_ids,torch.full((1,),-1,dtype=torch.long)])[reorder]
            if baseline_owner is None:
                baseline_owner,baseline_coverage=owner_mask.clone(),mask.clone()
            baseline_cost=info['rows'][2]['cost' if mode=='existing_armijo' else 'accepted_cost']
            delta=total_projected.detach().numpy().astype(np.float64)-native['native_total_gradient']
            node_energy=np.sum(delta*delta,axis=1);ordered=np.sort(node_energy)[::-1];energy=float(node_energy.sum())
            native_projected=projected(torch.from_numpy(native['native_total_gradient_unprojected']).float()).numpy()
            native_priors=native['native_priors']
            prior_delta=priors.detach().numpy()-native_priors
            covered_together=mask.numpy() & native['native_coverage']
            record={'cost':float(data_cost+prior_cost),'data_cost':float(data_cost),'prior_cost':float(prior_cost),
                'baseline_accepted_cost_exact':float(data_cost+prior_cost)==baseline_cost,
                'baseline_accepted_gradient_exact':bool(np.array_equal(total_projected.detach().numpy(),expected_gradient)),
                'complete_projected_gradient':difference(total_projected.detach().numpy(),native['native_total_gradient']),
                'complete_unprojected_gradient':difference(total_gradient.detach().numpy(),native['native_total_gradient_unprojected']),
                'data_projected_gradient':difference(data_projected.detach().numpy(),native['native_data_gradient']),
                'data_unprojected_gradient':difference(data_gradient.detach().numpy(),native['native_data_gradient_unprojected']),
                'prior_projected_gradient':difference(prior_projected.detach().numpy(),native['native_prior_gradient']),
                'prior_unprojected_gradient':difference(prior_gradient.detach().numpy(),native['native_prior_gradient_unprojected']),
                'native_sliding_vs_FNIT_projection_of_native_unprojected':difference(native_projected,native['native_total_gradient']),
                'changed_FNIT_owner_rows_vs_baseline':int((owner_mask!=baseline_owner).sum()),
                'changed_FNIT_coverage_vs_baseline':int((mask!=baseline_coverage).sum()),
                'FNIT_native_coverage_mismatch':int(np.count_nonzero(mask.numpy()!=native['native_coverage'])),
                'priors_vs_native_float_drawer_on_both_covered':difference(priors.detach().numpy()[:,covered_together],native_priors[:,covered_together]),
                'covered_voxels_with_prior_error_above_1e_5':int(np.count_nonzero(np.max(np.abs(prior_delta[:,covered_together]),axis=0)>1e-5)),
                'vertex_gradient_error_L2_above_1e_4':int(np.count_nonzero(np.sqrt(node_energy)>1e-4)),
                'gradient_error_energy_top1_fraction':float(ordered[:1].sum()/energy) if energy else 0.,
                'gradient_error_energy_top4_fraction':float(ordered[:4].sum()/energy) if energy else 0.,
                'min_jacobian':float(jacobian.min()),
                'gate':{'gradient_relative_l2_pass':bool(np.linalg.norm(delta)/np.linalg.norm(native['native_total_gradient'])<=1e-5),
                        'absolute_cost_pass':abs(float(data_cost+prior_cost)-native_report['native_total_cost'])<=.01}}
            private=args.output/(name+'_'+variant+'.private.npz')
            np.savez_compressed(private,points=points.detach().numpy(),priors=priors.detach().numpy(),
                owner=owner_mask.numpy(),coverage=mask.numpy(),gradient=total_projected.detach().numpy(),
                gradient_unprojected=total_gradient.detach().numpy(),data_gradient=data_projected.detach().numpy(),
                prior_gradient=prior_projected.detach().numpy(),weights=weights.detach().numpy(),selected_cells=cells.numpy())
            record['private_arrays_sha256']=digest(private)
            variants[variant]=record
        if not variants['baseline']['baseline_accepted_cost_exact'] or not variants['baseline']['baseline_accepted_gradient_exact']:
            raise RuntimeError('manual baseline formula is not exact to actual accepted state')
        results[name]=variants
    report={'status':'completed_bounded_third_point_gradient_variants','same_points':results,
        'scope':'Diagnostic variants only; no production/source/GPU/default edits or extra optimizer updates',
        'source_sha256':capture['source'],'capture_sha256':digest(args.capture/'shared_input.npz'),
        'program_sha256':digest(__file__),'native_summary_sha256':digest(args.native/'summary.public.json'),
        'affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads(),
        'output_directory_mode':oct(stat.S_IMODE(args.output.stat().st_mode))}
    (args.output/'summary.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'gates':{name:{v:r['gate'] for v,r in variants.items()} for name,variants in results.items()}}))


if __name__=='__main__':
    main()
