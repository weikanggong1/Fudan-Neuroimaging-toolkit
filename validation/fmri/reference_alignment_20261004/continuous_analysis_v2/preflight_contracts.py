"""Real saved metadata preflight and refusal contracts; no MRI/endpoint calculations."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import time

import continuous_closure as closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    initial = closure.Audit()
    bindings = closure.load(initial.check({'path': str(args.bindings), 'sha256': closure.BINDINGS_SHA}, 'original_bindings'))
    runs = args.runs.resolve(strict=True)
    queue = closure.load(runs / 'queue.private.json')
    results = []
    for variant in ('middle', 'robust'):
        audit = closure.Audit()
        item = closure.checked_case('CON01', variant, runs, bindings, queue, audit)
        closure.require(item is not None and item['status'] == 'ready', 'actual CON01 variant not ready')
        before, after = audit.finish()
        results.append({'test': 'actual_CON01_' + variant, 'passed': before == after,
                        'original_report_sha256': closure.sha256(item['report_path']),
                        'guarded_files': len(before), 'runtime_reconstruction_files': item['late_runtime_reconstruction_files'],
                        'consumed_resources': item['late_resource_files']})
    original_load = closure.load
    report_path = runs / 'CON01/middle/report/report.public.json'
    report = original_load(report_path)
    # These change only in-memory copies of real saved metadata. No producer,
    # MRI file or original JSON is edited; the checks are not MRI benchmarks.
    mutations = [
        ('source_guard_false', 'source_guards_equal', False),
        ('input_guard_false', 'input_guards_equal', False),
        ('wrong_variant', 'variant', 'robust'),
        ('wrong_driver', 'driver_sha256', '0' * 64),
        ('wrong_source', 'source_sha256', '0' * 64),
        ('missing_frame', 'frames', 179),
        ('invalid_API_clock', 'api_seconds', -1),
        ('wrong_manifest_SHA', 'manifest_sha256', '0' * 64),
    ]
    for label, field, value in mutations:
        altered = copy.deepcopy(report)
        altered[field] = value
        closure.load = lambda p, a=altered: copy.deepcopy(a) if Path(p).resolve() == report_path.resolve() else original_load(p)
        rejected = False
        try:
            closure.checked_case('CON01', 'middle', runs, bindings, queue, closure.Audit())
        except ValueError:
            rejected = True
        finally:
            closure.load = original_load
        closure.require(rejected, 'metadata refusal contract failed: ' + label)
        results.append({'test': label, 'passed': rejected})
    protected = [Path(bindings['raw_bids_root']), runs]
    for label, target in [('inside_raw', protected[0] / 'refusal-test-never-created'),
                          ('contains_raw', protected[0].parent),
                          ('inside_MRI_run', runs / 'refusal-test-never-created')]:
        rejected = False
        try:
            closure.fresh_output(target, protected)
        except ValueError:
            rejected = True
        closure.require(rejected, 'output refusal contract failed: ' + label)
        results.append({'test': label, 'passed': rejected})
    with tempfile.TemporaryDirectory(prefix='fnit-late-provenance-contracts-') as directory:
        root = Path(directory)
        target = root / 'new'
        closure.require(closure.fresh_output(target, protected) == target and not target.exists(), 'fresh safe output contract failed')
        results.append({'test': 'fresh_safe_path_no_mkdir', 'passed': True})
        link = root / 'raw_link'
        link.symlink_to(protected[0], target_is_directory=True)
        rejected = False
        try:
            closure.fresh_output(link / 'refusal-test-never-created', protected)
        except ValueError:
            rejected = True
        closure.require(rejected, 'symlink-to-input contract failed')
        results.append({'test': 'symlink_inside_raw', 'passed': rejected})
    initial.finish()
    print(json.dumps({'schema': 'fnit.continuous_late_contracts.v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'scope': 'real_saved_metadata_SHA_preflight_and_in_memory_refusal_contracts_no_MRI_no_endpoint_metrics',
                      'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'closure_helper_sha256': hashlib.sha256(Path(closure.__file__).read_bytes()).hexdigest(),
                      'passed': all(x['passed'] for x in results), 'checks': results,
                      'elapsed_seconds': time.perf_counter() - start}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
