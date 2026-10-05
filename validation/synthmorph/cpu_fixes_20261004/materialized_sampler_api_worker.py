"""Delegate the materialized-object API gate and record its actual CPU sampler."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialized-worker', required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [args.materialized_worker, *remaining]
    runpy.run_path(args.materialized_worker, run_name='__main__')
    from fnit.synthmorph import _cpu_raw_sampler as helper
    route = helper.backend_info()
    restored = helper.numba.get_num_threads() == route.get('previous_threads')
    output = Path(remaining[remaining.index('--output') + 1])
    report = {'scope': __doc__, 'actual_backend': route, 'numba_mask_restored': restored,
              'helper_sha256': hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),
              'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'sampler_route.private.json').write_text(json.dumps(report, indent=2) + '\n')
    if route.get('backend') != 'numba' or route.get('requested_threads') != 8 or not restored:
        raise RuntimeError('materialized CPU API did not use the bounded NumBa sampler')


if __name__ == '__main__':
    main()
