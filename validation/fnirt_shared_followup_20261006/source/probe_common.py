"""Small private validation helpers; public reports contain scalars and hashes."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError('different array shapes')
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    result = {'shape':list(reference.shape),'dtype_reference':str(reference.dtype),
              'dtype_candidate':str(candidate.dtype),'different':int(np.count_nonzero(delta)),
              'all_finite':bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
              'max_abs':float(np.max(np.abs(delta))),
              'rmse':float(np.sqrt(np.mean(delta*delta))),
              'relative_l2':float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(reference.ravel()),np.finfo(float).tiny))}
    if reference.dtype == candidate.dtype == np.float64:
        result['different_bits'] = int(np.count_nonzero(reference.view(np.uint64) != candidate.view(np.uint64)))
    return result


def load_vector(oracle, name, dimension=1177):
    values = np.fromfile(Path(oracle)/name,dtype='<f8')
    if values.size != dimension or not np.isfinite(values).all():
        raise ValueError('invalid bounded vector '+name)
    return values


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
