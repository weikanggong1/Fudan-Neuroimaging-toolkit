"""Collect completed fixed-identity CBIG comparisons and matched CPU clocks.

No individual labels or private paths enter this aggregate. The guarded
comparison must first finish both CPU budgets successfully.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from compare_official import CORE_SHA256, COMMON_MODULE_SHA256


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clock_values(path):
    text = Path(path).read_text()
    fields = {}
    for key in ('User time (seconds)', 'System time (seconds)',
                'Maximum resident set size (kbytes)', 'Exit status'):
        match = re.search(r'^\s*' + re.escape(key) + r':\s*([^\n]+)', text, re.MULTILINE)
        if match is None:
            raise ValueError('Incomplete GNU time receipt: ' + key)
        fields[key] = float(match.group(1))
    elapsed = re.search(r'^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)', text, re.MULTILINE)
    if elapsed is None or fields['Exit status'] != 0:
        raise ValueError('Successful complete process clock required')
    parts = elapsed.group(1).strip().split(':')
    if len(parts) not in (2, 3):
        raise ValueError('Unknown GNU time wall-clock format')
    wall = sum(float(part) * 60 ** index for index, part in enumerate(reversed(parts)))
    return {'full_process_wall_seconds': wall,
            'user_seconds': fields['User time (seconds)'],
            'system_seconds': fields['System time (seconds)'],
            'peak_RSS_bytes': int(fields['Maximum resident set size (kbytes)']) * 1024,
            'exit_status': 0, 'GNU_time_receipt_sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--candidate-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier official aggregate')
    queue = json.loads((args.run_root / 'queue_status.public.json').read_text())
    expected = [('cbig', 1), ('fnit', 1), ('fnit_candidate', 1),
                ('cbig', 8), ('fnit', 8), ('fnit_candidate', 8)]
    if (queue.get('state') != 'complete' or queue.get('execution_hostname') != 'nodecw10'
            or [(row['name'], row['threads']) for row in queue['jobs']] !=
            [(name + '_cpu' + str(threads), threads) for name, threads in expected]
            or any(row.get('state') != 'complete' or row.get('returncode') != 0
                   for row in queue['jobs'])):
        raise ValueError('All six actual same-node original/frozen/candidate jobs must finish')
    current = dict(COMMON_MODULE_SHA256, **{'core.py': CORE_SHA256['fnit_candidate']})
    for name, expected_sha in current.items():
        if sha(args.candidate_source / 'src/fnit/mshbm' / name) != expected_sha:
            raise ValueError('Current candidate source differs from matched benchmark')
    reports = {}
    for threads in (1, 8):
        path = args.run_root / ('official_cbig_cpu' + str(threads) + '_precision_v5.public.json')
        value = json.loads(path.read_text())
        if (not value.get('coordinator_and_input_gates_passed')
                or value.get('network_identities_permuted') is not False
                or value.get('full_vertices') != 64984
                or value.get('input_sha256') != queue['input_sha256']
                or value.get('reader_values_compared') != 29111880
                or len(value.get('comparisons', [])) != 2):
            raise ValueError('Full fixed-identity official precision gate required')
        seen = set()
        for comparison in value['comparisons']:
            core = comparison['actual_source_sha256']['src/fnit/mshbm/core.py']
            kind = next((name for name, source in CORE_SHA256.items() if source == core), None)
            if kind is None or comparison['threads'] != threads or kind in seen:
                raise ValueError('Actual frozen/candidate source or budget differs')
            seen.add(kind)
            for hemisphere in ('left', 'right', 'both'):
                precision = comparison['precision'][hemisphere]
                expected_vertices = 64984 if hemisphere == 'both' else 32492
                if (precision['vertices'] != expected_vertices or
                        [row['network'] for row in precision['per_network']] != list(range(1, 18))):
                    raise ValueError('No cropped vertices or relabelled networks permitted')
        if seen != set(CORE_SHA256):
            raise ValueError('Both actual FNIT implementations required')
        reports[str(threads)] = {'precision_receipt_sha256': sha(path), 'report': value}
    records = []
    for job in queue['jobs']:
        name = job['name']
        implementation = name.rsplit('_cpu', 1)[0]
        report_path = args.run_root / name / 'report.public.json'
        report = json.loads(report_path.read_text())
        row = {'implementation': implementation, 'threads': job['threads'],
               'affinity': job['affinity'], 'hostname': 'nodecw10',
               'load_before': job['load_before'], 'load_after': job['load_after'],
               'report_sha256': sha(report_path),
               **clock_values(args.run_root / (name + '.clock.private.txt'))}
        if implementation == 'cbig':
            row.update({'function_chain_seconds': report['function_seconds'],
                        'API_clock_scope': 'Original CBIG full CIFTI wrapper: original reader, profiles, complete inference and upstream normal save; isolated benchmark label-MAT export after this clock',
                        'reference_commit': queue['reference_commit'],
                        'matlab_version': report['matlab_version']})
        else:
            row.update({'function_chain_seconds': report['function_chain_seconds'],
                        'stages_seconds': report['stages_seconds'],
                        'API_clock_scope': 'Complete FNIT surface call, including normal label/network-timeseries/FC/provenance output save; benchmark validation and report outside this clock',
                        'actual_source_sha256': report['source_sha256']})
        records.append(row)
    aggregate = {'status': 'complete', 'scope': 'Complete real490-frame matched CPU1/8 original CBIG/frozen FNIT/current candidate; fixed17 identities on all64984 vertices',
                 'baseline_commit': queue['baseline_commit'], 'reference_commit': queue['reference_commit'],
                 'input_sha256': queue['input_sha256'], 'current_candidate_modules_sha256': current,
                 'complete_frames': 490, 'complete_fsLR_vertices': 64984,
                 'network_identities_permuted': False,
                 'timing_protocol': 'One complete fresh-process observation per implementation and thread budget; no warmup or stable speedup inference',
                 'full_process_clock_scope': 'GNU time around actual reference/FNIT process; includes interpreter/imports, validation, complete function and report saving; excludes CPU lease wait and preparation',
                 'records': records, 'precision_by_CPU_threads': reports,
                 'privacy': 'Aggregate metrics and source receipts only; no individual label arrays, brain maps or private paths'}
    args.output.write_text(json.dumps(aggregate, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'complete', 'process_observations': 6,
                      'fixed_networks': 17, 'full_vertices': 64984,
                      'output_sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
