"""One fresh-cache 256 call, import/JIT separate; no repeated CNN or oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from fnit.synthmorph import _cpu_preprocessing, spatial


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--completed-report', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(8); torch.set_num_interop_threads(1)
reference = json.loads(args.completed_report.read_text())
data_file = args.reference / 'source_data_0.npy'
matrix_file = args.reference / 'source_to_network_0.npy'
volume = torch.from_numpy(np.array(np.load(data_file, mmap_mode='r'), dtype=np.float32, copy=True))[None, None]
matrix = np.load(matrix_file)
row = reference['cases']['256']['images']['0']
assert hashlib.sha256(data_file.read_bytes()).hexdigest() == row['source_file_sha256']
assert hashlib.sha256(matrix_file.read_bytes()).hexdigest() == row['matrix_file_sha256']
report = {'scope': __doc__, 'affinity': sorted(os.sched_getaffinity(0)),
          'source_sha256': reference['source_sha256'], 'input_file_sha256': row['source_file_sha256'],
          'matrix_file_sha256': row['matrix_file_sha256'], 'load_before': list(os.getloadavg()),
          'compiled_cache_initially_empty': not any(Path(os.environ['NUMBA_CACHE_DIR']).iterdir())}
with torch.inference_mode():
    started = time.perf_counter(); old = _cpu_preprocessing.network_transform(volume, matrix, shape=(256,) * 3)
    report['v29_first_call_seconds'] = time.perf_counter() - started
    old_sha = array_sha(old.numpy()); del old
    started = time.perf_counter(); import raw_sampler_numba as candidate
    report['lazy_NumBa_module_import_seconds'] = time.perf_counter() - started
    started = time.perf_counter(); result = candidate.network_transform(volume, matrix, shape=(256,) * 3)
    report['candidate_first_call_including_JIT_seconds'] = time.perf_counter() - started
    report['candidate_import_and_first_call_seconds'] = report['lazy_NumBa_module_import_seconds'] + report['candidate_first_call_including_JIT_seconds']
    report['result_sha256'] = array_sha(result.numpy())
    assert report['result_sha256'] == old_sha == row['raw_vs_v29']['expected_sha256']
    coords = spatial.grid((256,) * 3, 'cpu', torch.float32)
    locations = coords + spatial._dense_from_grid(torch.as_tensor(matrix, dtype=torch.float32), coords)
    report['unchanged_einsum_locations_sha256'] = array_sha(locations.numpy())
report['worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report['load_after'] = list(os.getloadavg())
report['production_adopted'] = False
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
