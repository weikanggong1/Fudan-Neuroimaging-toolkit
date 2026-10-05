"""Private reconstructable CPU index checkpoint, no historical cost evaluation.

The original short process saved optimizer state and trial points, not its
Python cache object. We explicitly rebuild that deterministic cache from the
saved point sequence and verify the resume point separately. This does not
claim byte-identical Python objects or a fresh uninterrupted benchmark.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    array = value.detach().numpy() if torch.is_tensor(value) else np.asarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def fingerprint(closure):
    candidates = closure.index.candidates
    offsets = np.cumsum([0] + [len(x) for x in candidates], dtype=np.int64)
    packed = np.concatenate(candidates) if candidates else np.empty(0, dtype=np.int64)
    return {'anchor_sha256': array_sha(closure.anchor),
            'candidate_offsets_sha256': array_sha(offsets),
            'candidate_ids_sha256': array_sha(packed),
            'shape': list(closure.index.shape), 'block_size': closure.index.block_size,
            'index_rebuilds': closure.rebuilds, 'closure_evaluations': closure.evaluations,
            'cache_keys': sorted(repr(k) for k in closure.index._device_cache),
            'static_signature': closure.static_signature(), 'last': closure.last}


def reconstruct(closure, rasterize, prior_directory, prior):
    """Rebuild the final index anchor by following saved trials without fitting."""
    directory = Path(prior_directory) / 'native_definition_cpu'
    observations = []
    closure.evaluations = 1  # saved raw initial objective evaluation
    for row in prior['modes']['native_definition_cpu']['rows']:
        saved_initial = (closure.index, closure.anchor.clone(), closure.rebuilds,
                         closure.evaluations)
        selected = None
        for number, trial in enumerate(row['records'], 1):
            path = directory / ('trial-%03d-%03d.private.npz' % (row['step'], number))
            with np.load(path, allow_pickle=False) as loaded:
                points = torch.from_numpy(loaded['points'].copy())
            refreshed = bool((points-closure.anchor).abs().amax().item() > 1.5)
            if refreshed:
                closure.index = rasterize.build_block_index(
                    points.numpy(), closure.tetra.numpy(), closure.shape, 8, margin=3.)
                closure.anchor = points.clone()
                closure.rebuilds += 1
            closure.evaluations += 1
            if trial['alpha'] == row['alpha']:
                selected = (closure.index, closure.anchor.clone(), closure.rebuilds,
                            closure.evaluations)
            observations.append({'step': row['step'], 'trial': number,
                                 'private_trial_sha256': digest(path),
                                 'anchor_refreshed': refreshed})
        if row['alpha'] == 0:
            selected = saved_initial
        if selected is None:
            raise RuntimeError('accepted alpha is absent from saved trial sequence')
        closure.index, closure.anchor, closure.rebuilds, closure.evaluations = selected
        accepted = directory / ('accepted-%03d.private.npz' % row['step'])
        if digest(accepted) != row['private_state_sha256']:
            raise RuntimeError('saved accepted point changed')
    return {'method': 'rebuild deterministic anchor/index from saved trial points; no historical objective calls',
            'original_mutable_Python_cache_object_saved': False,
            'cache_rebuilt_and_resume_objective_rechecked': True,
            'accepted_closure_evaluations_reconstructed': closure.evaluations,
            'original_optimizer_evaluations': prior['modes']['native_definition_cpu']['rows'][-1]['total_evaluations'],
            'trial_provenance': observations, 'before_resume_cost_gate': fingerprint(closure)}


def save_portable(closure, path):
    """Persist anchor/index/counters; dynamic cache remains explicitly derivable."""
    candidates = closure.index.candidates
    np.savez_compressed(path, anchor=closure.anchor.numpy(),
                        candidate_offsets=np.cumsum([0] + [len(x) for x in candidates], dtype=np.int64),
                        candidate_ids=np.concatenate(candidates),
                        evaluations=closure.evaluations, rebuilds=closure.rebuilds)
    return {'private_checkpoint_sha256': digest(path),
            'state_fingerprint': fingerprint(closure),
            'cache_storage': 'derived immutable compact plans; explicitly rebuildable, no byte-object claim'}
