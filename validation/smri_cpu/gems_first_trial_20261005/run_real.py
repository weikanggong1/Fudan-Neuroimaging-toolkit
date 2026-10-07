"""Private CPU synthetic first-step replay from an existing real capture.

No alignment, EM update, recipe fit, installed software production call, or
production modification. A strict full-array initial-state gate precedes trials.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
from time import perf_counter

import numpy as np
import torch

from fnit.gems import core, rasterize
from fnit.gems.deformation import (ashburner_prior, prepare_current_geometry,
    prepare_deformation_reference, prepare_vertex_reduction, sliding_boundary_projectors)
from fnit.gems.gaussian import GaussianParameters, gaussian_log_likelihood


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    array = value.detach().numpy() if torch.is_tensor(value) else np.asarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


class SharedClosure:
    """The actual core CPU formula, with privately owned index/cache/counters.

    All static state is snapshotted by value and checked after each diagnostic.
    The sole mutable cache is BlockIndex._device_cache; refresh counters, anchor,
    last scalar observation and that cache are restored explicitly.
    """
    def __init__(self, arrays, background):
        self.image = torch.from_numpy(arrays['image'].copy())
        self.start = torch.from_numpy(arrays['vertices'].copy())
        self.reference = torch.from_numpy(arrays['reference'].copy())
        self.tetra = torch.from_numpy(arrays['tetrahedra'].copy()).long()
        self.alphas = torch.from_numpy(arrays['alphas'].copy())
        self.can_move = torch.from_numpy(arrays['can_move'].copy())
        self.boundary = torch.from_numpy(arrays['boundary_transform'].copy()).float()
        self.stiffness = float(arrays['stiffness'])
        if self.start.dtype != torch.float32 or self.image.dtype != torch.float32:
            raise RuntimeError('captured FP32 image and points required')
        self.valid = self.image.isfinite() & (self.image != 0)
        if not self.valid.any():
            raise RuntimeError('nonempty captured finite/nonzero mask required')
        self.shape = tuple(self.image.shape)
        self.background = int(background)
        self.parameters = GaussianParameters(torch.from_numpy(arrays['means'].copy()).float(),
            torch.from_numpy(arrays['variances'].copy()).float())
        self.likelihood = gaussian_log_likelihood(
            self.image[self.valid].reshape(-1, 1, 1), self.parameters).reshape(self.alphas.shape[1], -1).detach()
        self.projection = sliding_boundary_projectors(self.can_move, self.boundary)
        self.reference_geometry = prepare_deformation_reference(self.reference.double(), self.tetra)
        self.reduction = prepare_vertex_reduction(self.tetra.reshape(-1), len(self.start))
        self.index = rasterize.build_block_index(self.start.numpy(), self.tetra.numpy(), self.shape, 8, margin=3.)
        self.anchor = self.start.clone()
        self.evaluations = self.rebuilds = 0
        self.last = None

    def static_signature(self):
        tensors = {name: value for name, value in vars(self).items()
                   if torch.is_tensor(value) and name != 'anchor'}
        tensors.update(means=self.parameters.means, variances=self.parameters.covariances,
            reference_inverse=self.reference_geometry.inverse_edges,
            reference_volumes=self.reference_geometry.volumes,
            reduction_order=self.reduction.order, reduction_offsets=self.reduction.offsets)
        return {name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'sha256': array_sha(tensor)}
                for name, tensor in tensors.items()}

    def checkpoint(self):
        return {'index': self.index, 'cache': copy.deepcopy(self.index._device_cache),
                'anchor': self.anchor.clone(), 'evaluations': self.evaluations,
                'rebuilds': self.rebuilds, 'last': copy.deepcopy(self.last),
                'static': self.static_signature()}

    def restore(self, snapshot):
        if self.static_signature() != snapshot['static']:
            raise RuntimeError('closure mutated static image/mesh/likelihood/projection/reference state')
        self.index = snapshot['index']
        self.index._device_cache.clear()
        self.index._device_cache.update(snapshot['cache'])
        self.anchor = snapshot['anchor'].clone()
        self.evaluations = snapshot['evaluations']
        self.rebuilds = snapshot['rebuilds']
        self.last = copy.deepcopy(snapshot['last'])

    def evaluate(self, supplied, return_priors=False):
        self.evaluations += 1
        points = supplied.detach().clone().requires_grad_(True)
        if points.device.type != 'cpu' or points.dtype != torch.float32:
            raise RuntimeError('actual mixed candidate stores CPU FP32 points')
        if (points.detach() - self.anchor).abs().amax().item() > 1.5:
            self.index = rasterize.build_block_index(points.detach().numpy(), self.tetra.numpy(), self.shape, 8, margin=3.)
            self.anchor = points.detach().clone()
            self.rebuilds += 1
        geometry = prepare_current_geometry(points.double(), self.tetra,
            deterministic_gradient=True, vertex_reduction=self.reduction)
        owner = prepare_current_geometry(points.detach(), self.tetra)
        fused = rasterize.compact_data_cost(points, self.tetra, self.alphas, self.shape,
            valid_mask=self.valid, block_index=self.index, background_channel=self.background,
            current_geometry=geometry, likelihood=self.likelihood,
            cache_owner_hints=False, owner_hints=False, hint_tolerance=2e-4,
            hint_stats=None, double_accumulation=True, deterministic_gradient=True)
        if fused is not None:
            raise RuntimeError('frozen CPU branch unexpectedly used fused objective')
        priors, coverage = rasterize.rasterize_priors_compact(points, self.tetra,
            self.alphas, self.shape, valid_mask=self.valid, block_index=self.index,
            background_channel=self.background, current_geometry=geometry, owner_geometry=owner)
        data_cost = core._compact_mesh_data_cost(priors, self.likelihood, double_accumulation=True)
        prior_cost, jacobian = ashburner_prior(points, self.reference, self.tetra,
            self.stiffness, reference_geometry=self.reference_geometry,
            current_geometry=geometry, analytic_gradient=True, double_accumulation=True)
        objective = data_cost + 1. * prior_cost
        objective = torch.where(torch.isfinite(jacobian).all() & (jacobian > 0).all(),
            objective, objective.new_tensor(float('inf')))
        objective.backward()
        points.grad.copy_(torch.bmm(self.projection, points.grad[..., None]).squeeze(-1))
        self.last = {'data_cost': float(data_cost), 'prior_cost': float(prior_cost),
            'min_jacobian': float(jacobian.min()), 'covered_voxels': int(coverage.sum()),
            'index_rebuilds': self.rebuilds}
        result = (objective.detach(), points.grad.detach().clone())
        return (*result, priors.detach(), coverage.detach()) if return_priors else result

    def __call__(self, supplied):
        return self.evaluate(supplied)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--prototype', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=64)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    if stat.S_IMODE(args.output.stat().st_mode) != 0o700:
        raise RuntimeError('real-data output must actually have mode0700')
    torch.set_num_threads(8)
    import numba
    numba.set_num_threads(8)
    report = json.loads((args.capture / 'report.public.json').read_text())
    for name, expected in report['outputs'].items():
        if digest(args.capture / name) != expected:
            raise RuntimeError('captured array SHA mismatch: ' + name)
    source_dir = Path(core.__file__).parent
    actual_sources = {str(path.relative_to(source_dir)): digest(path)
                      for path in source_dir.rglob('*.py')}
    if actual_sources != report['source']:
        raise RuntimeError('GEMS source differs from actual candidate capture')
    with np.load(args.capture / 'shared_input.npz', allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    closure = SharedClosure(arrays, report['background_class'])
    before = closure.static_signature()
    begin = perf_counter()
    cost, gradient, priors, covered = closure.evaluate(closure.start, return_priors=True)
    gate = {'cost_exact': float(cost) == report['first_cost'],
        'gradient_exact': np.array_equal(gradient.numpy(), np.load(args.capture/'fnit_gradient.npy')),
        'priors_exact': np.array_equal(priors.numpy(), np.load(args.capture/'fnit_priors.npy')),
        'coverage_exact': np.array_equal(covered.numpy(), np.load(args.capture/'fnit_coverage.npy')),
        'points_exact': np.array_equal(closure.start.numpy(), np.load(args.capture/'fnit_vertices.npy'))}
    (args.output/'initial_gate.public.json').write_text(json.dumps({'gate': gate,
        'cost': float(cost), 'capture_cost': report['first_cost'], 'observation': closure.last,
        'single_evaluation_observation_seconds': perf_counter()-begin}, indent=2)+'\n')
    if not all(gate.values()):
        raise RuntimeError('strict actual closure-equivalence gate failed; no searches executed')
    closure.index._device_cache.clear()
    closure.evaluations = closure.rebuilds = 0
    closure.last = None
    spec = importlib.util.spec_from_file_location('first_step_diagnostic', args.prototype)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    optimizer_source = source_dir/'optim.py'
    summary = {}
    for name, function in [('armijo', module.existing_armijo_first_step),
                           ('reference', module.reference_first_step)]:
        snapshot = closure.checkpoint()
        started = perf_counter()
        if name == 'armijo':
            _, trace = function(closure.start, closure, args.output/name, optimizer_source, budget=args.budget)
        else:
            _, trace = function(closure.start, closure, args.output/name, budget=args.budget,
                zoom_interval_stop=1e-10)
        restored = (closure.static_signature() == before and closure.evaluations == snapshot['evaluations']
            and closure.rebuilds == snapshot['rebuilds'] and torch.equal(closure.anchor, snapshot['anchor'])
            and closure.last == snapshot['last'] and set(closure.index._device_cache) == set(snapshot['cache']))
        if not restored:
            raise RuntimeError('diagnostic did not restore complete closure state')
        summary[name] = {key: trace[key] for key in ('method','reason','evaluations',
            'initial_cost','selected_cost','selected_alpha','selected_max_node_displacement',
            'selected_source_armijo_satisfied','selected_strong_wolfe_satisfied')}
        summary[name].update(caller_static_state_restored=True,
            bounded_search_observation_seconds=perf_counter()-started)
    public = {'status': 'completed_bounded_same_state_python_first_steps',
        'scope': 'Real synthetic stage1 only; reference search is source-informed Python, not actual native optimizer; no complete fit or default adoption',
        'structure': report['structure'], 'gate': gate, 'searches': summary,
        'host': socket.gethostname(), 'affinity': sorted(os.sched_getaffinity(0)),
        'torch_threads': torch.get_num_threads(), 'numba_threads': numba.get_num_threads(),
        'capture_inputs_sha256': {name:digest(args.capture/name) for name in ['report.public.json','shared_input.npz']},
        'program_sha256': digest(__file__), 'prototype_sha256': digest(args.prototype),
        'source_sha256': actual_sources,
        'native_options_for_source_informed_search': {'LineSearchMaximalDeformationIntervalStopCriterion':1e-10},
        'output_directory_mode': oct(stat.S_IMODE(args.output.stat().st_mode))}
    (args.output/'summary.public.json').write_text(json.dumps(public, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':public['status'],'gate':gate,'searches':summary}))


if __name__ == '__main__':
    main()
