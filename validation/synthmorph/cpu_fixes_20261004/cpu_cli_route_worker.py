"""Run the unchanged CLI and record the actual optional CPU sampling route."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True)
    parser.add_argument('--expected-backend')
    args, remaining = parser.parse_known_args()
    started = time.monotonic()
    sys.argv = ['fnit', *remaining]
    runpy.run_module('fnit.cli', run_name='__main__')
    cli_seconds = time.monotonic() - started
    helper = sys.modules.get('fnit.synthmorph._cpu_raw_sampler')
    route = helper.backend_info() if helper else {'backend': None, 'reason': 'helper not imported'}
    files = {}
    for name in ('models', 'pipeline', 'spatial', '_cpu_preprocessing', '_cpu_raw_sampler', '_cpu_eigen'):
        module = sys.modules.get('fnit.synthmorph.' + name)
        if module:
            files[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    mask_restored = (helper.numba.get_num_threads() == route['previous_threads']
                     if helper and route.get('backend') == 'numba' else None)
    import torch
    report = {'scope': __doc__, 'hostname': socket.gethostname(),
              'affinity': sorted(os.sched_getaffinity(0)), 'cli_body_seconds': cli_seconds,
              'torch_threads': torch.get_num_threads(), 'route': route,
              'numba_mask_restored': mask_restored, 'source_sha256': files,
              'numba_cache_directory': os.environ.get('NUMBA_CACHE_DIR'),
              'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(args.record).write_text(json.dumps(report, indent=2) + '\n')
    if args.expected_backend and (route['backend'] != args.expected_backend
                                  or route.get('requested_threads') != 8 or not mask_restored):
        raise RuntimeError('CPU CLI did not use the required bounded sampling route')


if __name__ == '__main__':
    main()
