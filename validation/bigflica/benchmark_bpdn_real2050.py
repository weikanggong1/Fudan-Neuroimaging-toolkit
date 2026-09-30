"""Compare batched ADMM/BPDN with production LARS on real projected data.

Usage: python benchmark_bpdn_real2050.py PROJECTED_DIR DICTIONARY_DIR OUTPUT_JSON
The equations follow SPORCO BPDN (BSD-3); implementation is independent PyTorch.
Only sparse encoding is tested; this is not an end-to-end benchmark.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from fnit.bigflica.dicl_torch import _LarsInverseSolver


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


class BPDNSolver:
    """Reuse a twenty-step graph; factorization is refreshed for each dictionary."""

    def __init__(self, batch_size, atoms, device, dtype):
        self.inverse = torch.eye(atoms, device=device, dtype=dtype)
        self.response = torch.zeros((batch_size, atoms), device=device, dtype=dtype)
        self.code = torch.zeros_like(self.response)
        self.dual = torch.zeros_like(self.code)
        self.rho = torch.ones((), device=device, dtype=dtype)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            for _ in range(20):
                self.step()

    def step(self):
        estimate = (self.response + self.rho * (self.code - self.dual)) @ self.inverse
        relaxed = 1.8 * estimate + (1 - 1.8) * self.code
        value = relaxed + self.dual
        updated = value.sign() * (value.abs() - 1 / self.rho).clamp_min(0)
        self.dual.add_(relaxed - updated)
        self.code.copy_(updated)

    def __call__(self, batch, dictionary, iterations, rho):
        gram = dictionary @ dictionary.T
        self.inverse.copy_(torch.cholesky_inverse(torch.linalg.cholesky(
            gram + rho * torch.eye(len(dictionary), device=batch.device, dtype=batch.dtype))))
        self.response.copy_(batch @ dictionary.T)
        self.rho.fill_(rho)
        self.code.zero_(); self.dual.zero_()
        for _ in range(iterations // 20):
            self.graph.replay()
        return self.code.clone()


def kkt(code, batch, dictionary):
    gradient = (code @ dictionary - batch) @ dictionary.T
    violation = torch.where(code.abs() > 1e-9,
                            (gradient + code.sign()).abs(),
                            (gradient.abs() - 1).clamp_min(0))
    return float(violation.max())


def main():
    projected, dictionaries, output = map(Path, sys.argv[1:])
    torch.backends.cuda.matmul.allow_tf32 = True
    reports = []
    for name in ('vbm', 'fa', 'md'):
        dictionary = torch.as_tensor(np.load(dictionaries / f'{name}_gpu_dictionary.npy'),
                                     device='cuda', dtype=torch.float64)
        dictionary /= torch.linalg.vector_norm(dictionary, dim=1, keepdim=True).clamp_min(1)
        with h5py.File(projected / f'{name}_projected.h5') as file:
            data = file['data']
            sums = np.zeros(data.shape[1]); squares = np.zeros_like(sums)
            for start in range(0, len(data), 4096):
                block = data[start:start + 4096].astype(np.float64)
                sums += block.sum(0); squares += np.square(block).sum(0)
            mean = sums / len(data)
            std = np.sqrt(np.maximum(squares / len(data) - mean**2, 0)); std[std == 0] = .1
            batch = torch.as_tensor((data[:32].astype(np.float64) - mean) / std,
                                    device='cuda', dtype=torch.float64)
        # Published dictionaries are standardized after learning and normalized
        # to the training unit-ball constraint for this probe. This test
        # examines these exact inputs, not the online training trajectory.
        solver = _LarsInverseSolver(32, len(dictionary), 'cuda', torch.float64)
        reference, reference_s = timed(lambda: solver(batch, dictionary, 1., 1000))
        candidate, setup_s = timed(lambda: BPDNSolver(32, len(dictionary), 'cuda', torch.float64))
        row = {'admm_graph_setup_seconds': setup_s, 'modality': name, 'lars_seconds': reference_s,
               'lars_kkt': kkt(reference, batch, dictionary), 'candidates': []}
        for rho in (.1, 1., 10.):
            for iterations in (200, 1000):
                code, seconds = timed(lambda: candidate(batch, dictionary, iterations, rho))
                row['candidates'].append({'rho': rho, 'iterations': iterations,
                    'seconds_including_factorization_excluding_graph_setup': seconds,
                    'relative_code_error': float(torch.linalg.vector_norm(code - reference) /
                                                 torch.linalg.vector_norm(reference)),
                    'kkt_max_abs': kkt(code, batch, dictionary)})
        reports.append(row)
        print(json.dumps(row), flush=True)
    output.write_text(json.dumps({'scope': 'Real 2050-subject full-mask projections; first 32 voxels per modality; final dictionaries normalized to unit-ball constraint, not online trajectory',
        'upstream_equations': 'https://sporco.readthedocs.io/en/latest/modules/sporco.admm.bpdn.html',
        'dtype': 'float64 existing DicL precision', 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(), 'shared_gpu': True, 'results': reports}, indent=2))


if __name__ == '__main__':
    main()
