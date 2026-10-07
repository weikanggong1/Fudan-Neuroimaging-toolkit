"""Private portable CPU index checkpoint, no historical cost evaluation.

The accepted4 process saved optimizer state and a portable anchor/index. Its
dynamic Python plans are rebuilt from the exact saved candidate list and the
accepted point is rechecked separately. This does not
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
    """Restore saved portable checkpoint; rebuild only immutable dynamic plans."""
    import copy
    prior_directory=Path(prior_directory)
    path=prior_directory/'closure_state.private.npz'
    saved=prior['closure_checkpoint']['state_fingerprint']
    if digest(path)!=prior['closure_checkpoint']['private_checkpoint_sha256']:
        raise RuntimeError('saved portable closure checkpoint changed')
    if closure.static_signature()!=saved['static_signature']:
        raise RuntimeError('restored static source/precision/input differs')
    if prior['actual_private_CPU_precision_policy']!=closure.precision_policy:
        raise RuntimeError('restored CPU precision policy differs')
    with np.load(path,allow_pickle=False) as loaded:
        if set(loaded.files)!={'anchor','candidate_offsets','candidate_ids','evaluations','rebuilds'}:
            raise RuntimeError('unexpected portable checkpoint schema')
        arrays={k:loaded[k].copy() for k in loaded.files}
    anchor,offsets,ids=arrays['anchor'],arrays['candidate_offsets'],arrays['candidate_ids']
    shape=tuple(saved['shape']);block=int(saved['block_size'])
    nblocks=int(np.prod([(n+block-1)//block for n in shape]))
    if (shape!=closure.shape or block!=8 or anchor.shape!=tuple(closure.start.shape)
            or anchor.dtype!=closure.start.numpy().dtype or not np.isfinite(anchor).all()
            or offsets.dtype!=np.int64 or offsets.shape!=(nblocks+1,)
            or offsets[0]!=0 or offsets[-1]!=len(ids) or np.any(np.diff(offsets)<0)
            or ids.ndim!=1 or ids.dtype.kind not in 'iu'
            or np.any(ids<0) or np.any(ids>=len(closure.tetra))):
        raise RuntimeError('portable geometry/index schema is invalid')
    for key,value in [('anchor_sha256',anchor),('candidate_offsets_sha256',offsets),('candidate_ids_sha256',ids)]:
        if array_sha(value)!=saved[key]:raise RuntimeError('portable array digest differs: '+key)
    counts={name:int(arrays[name]) for name in ('evaluations','rebuilds')}
    if (counts['evaluations']!=saved['closure_evaluations']
            or counts['rebuilds']!=saved['index_rebuilds'] or min(counts.values())<0):
        raise RuntimeError('saved mutable counters differ')
    closure.index=rasterize.BlockIndex(shape,block,tuple(ids[a:b].copy() for a,b in zip(offsets[:-1],offsets[1:])))
    closure.anchor=torch.from_numpy(anchor)
    closure.evaluations=counts['evaluations'];closure.rebuilds=counts['rebuilds']
    closure.last=copy.deepcopy(saved['last'])
    actual=fingerprint(closure)
    if {k:v for k,v in actual.items() if k!='cache_keys'}!={k:v for k,v in saved.items() if k!='cache_keys'}:
        raise RuntimeError('portable restored closure fingerprint differs')
    accepted_row=prior['modes']['native_definition_cpu']['rows'][-1]
    accepted=prior_directory/'native_definition_cpu'/('accepted-%03d.private.npz'%accepted_row['step'])
    if digest(accepted)!=accepted_row['private_state_sha256']:
        raise RuntimeError('resume accepted point changed')
    return {'method':'restore hashed portable anchor/index/counters/last; rebuild immutable plans at accepted-point gate',
            'original_mutable_Python_cache_object_saved':False,
            'dynamic_cache_initially_empty':len(closure.index._device_cache)==0,
            'saved_dynamic_cache_keys':saved['cache_keys'],
            'cache_rebuilt_and_resume_objective_rechecked':False,
            'portable_checkpoint_sha256':digest(path),
            'accepted_point_sha256':digest(accepted),
            'original_optimizer_evaluations':accepted_row['total_evaluations'],
            'before_resume_cost_gate':actual}

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
