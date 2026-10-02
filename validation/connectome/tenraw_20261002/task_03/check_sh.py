"""冻结基线同输入SH逐值回归和GPU同步AB/BA测量（非真实pipeline benchmark）。"""
import argparse
import importlib.util
import json
import hashlib
import os
import time
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)


parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True)
parser.add_argument('--candidate', required=True)
parser.add_argument('--fixture', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('baseline_fod', args.baseline)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
baseline = module.tracking_sh_precomputed
spec = importlib.util.spec_from_file_location('candidate_fod', args.candidate)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
candidate = module.tracking_sh_precomputed
reports = []
for device in ['cpu', 'cuda']:
    if device == 'cuda' and not torch.cuda.is_available():
        continue
    gen = torch.Generator().manual_seed(23)
    for lmax in [0, 2, 4, 6, 8, 10, 12]:
        random = torch.randn((2, 257, 3), generator=gen).to(device)
        edge = torch.tensor([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, -1, 0],
                             [1e-30, 0, 1], [0, 0, 0]], dtype=torch.float32, device=device)
        for data in [random, edge]:
            a, b = baseline(data, lmax), candidate(data, lmax)
            assert torch.equal(a, b), (device, lmax, (a-b).abs().max())
        reports.append({'device': device, 'lmax': lmax, 'neq': 0})
    with np.load(args.fixture) as fixture:
        directions = torch.from_numpy(fixture['directions'].copy()).to(device)
    assert torch.equal(baseline(directions, 8), candidate(directions, 8))
    if device == 'cuda':
        timings = []
        # Real direction fixture replicated only for operator throughput.
        for n in [128, 8192, 131072]:
            data = directions.reshape(-1,3).repeat(((n + directions.reshape(-1, 3).shape[0] - 1) // directions.reshape(-1, 3).shape[0]),1)[:n]
            for label, fn in [('A', baseline), ('B', candidate), ('B', candidate), ('A', baseline)]:
                for _ in range(3): fn(data, 8)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(30): fn(data, 8)
                torch.cuda.synchronize()
                timings.append({'n': len(data), 'variant': label, 'seconds_per_call': (time.perf_counter()-start)/30})
        reports.append({'gpu_timings': timings, 'allocated_peak': torch.cuda.max_memory_allocated(), 'reserved_peak': torch.cuda.max_memory_reserved()})
report = {'source_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in [('baseline',args.baseline),('candidate',args.candidate),('fixture',args.fixture)]}, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch_version': torch.__version__, 'scope': 'operator diagnostic; replicated old fixture is not newpilot benchmark', 'results': reports}
Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(reports), flush=True)
