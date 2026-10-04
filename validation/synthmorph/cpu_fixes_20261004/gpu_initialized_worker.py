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
    args, remaining = parser.parse_known_args()
    started = time.perf_counter()
    torch.cuda.init()
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(19_000_000_000 / total, 0)
    initialization = time.perf_counter() - started
    sys.argv = [args.worker, *remaining]
    runpy.run_path(args.worker, run_name='__main__')
    output = Path(remaining[remaining.index('--output') + 1])
    record = {'scope': 'benchmark initialization order only; production calculations unchanged',
              'initialization_seconds': initialization, 'logical_cuda_device': 0,
              'quota_bytes': 19_000_000_000,
              'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'worker_sha256': hashlib.sha256(Path(args.worker).read_bytes()).hexdigest()}
    (output / 'initialization.private.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
