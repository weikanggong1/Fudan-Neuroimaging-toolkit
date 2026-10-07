"""Initialize CUDA before importing the unchanged registration benchmark worker."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--expected-uuid')
    args, remaining = parser.parse_known_args()
    started = time.perf_counter()
    torch.cuda.init()
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(getattr(properties, 'uuid', ''))
    if args.expected_uuid and actual_uuid.lower().removeprefix('gpu-') != args.expected_uuid.lower().removeprefix('gpu-'):
        raise RuntimeError('selected CUDA GPU UUID differs: ' + actual_uuid)
    total = properties.total_memory
    torch.cuda.set_per_process_memory_fraction(19_000_000_000 / total, 0)
    initialization = time.perf_counter() - started
    sys.argv = [args.worker, *remaining]
    runpy.run_path(args.worker, run_name='__main__')
    helpers = [name for name in ('fnit.synthmorph._cpu_eigen', 'fnit.synthmorph._cpu_preprocessing',
                                 'fnit.synthmorph._cpu_features') if name in sys.modules]
    if helpers:
        raise RuntimeError('GPU benchmark imported CPU-only helpers: ' + ', '.join(helpers))
    output = Path(remaining[remaining.index('--output') + 1])
    record = {'scope': 'benchmark initialization order only; production calculations unchanged',
              'initialization_seconds': initialization, 'logical_cuda_device': 0,
              'quota_bytes': 19_000_000_000,
              'actual_cuda_uuid': actual_uuid, 'CPU_helpers_loaded': helpers,
              'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'worker_sha256': hashlib.sha256(Path(args.worker).read_bytes()).hexdigest()}
    (output / 'initialization.private.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
