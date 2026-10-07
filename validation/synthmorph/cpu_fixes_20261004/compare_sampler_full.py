"""Bind each new complete CPU sampler run to the accepted same-input outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from compare_cpu_branch import image_comparison
from build_joint_precision_report import gate


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def selected_run(folder):
    record = load(folder / 'record.json')
    if record['status'] != 'complete' or record['returncode'] != 0:
        raise RuntimeError('complete CPU run is unavailable: ' + folder.name)
    expected_threads = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'NUMBA_NUM_THREADS')
    if (record['hostname'] != 'nodecw7' or record['max_cpu_threads'] != 8
            or record['cpu_affinity'] != '2,6,10,14,18,22,26,30'
            or any(record['environment'][name] != '8' for name in expected_threads)
            or record['environment']['CUDA_VISIBLE_DEVICES'] != ''):
        raise RuntimeError('CPU budget or same-node condition changed')
    if hashlib.sha256(json.dumps(record['job'], sort_keys=True).encode()).hexdigest() != record['job_sha256']:
        raise RuntimeError('recorded job identity changed')
    actual_command = ['taskset', '-c', record['cpu_affinity'], '/usr/bin/time',
                      '-v', '-o', str(folder / 'time.txt'), *record['job']['argv']]
    if record['argv'] != actual_command:
        raise RuntimeError('actual command differs from the recorded job')
    resources = {line.strip().rsplit(': ', 1)[0]: line.strip().rsplit(': ', 1)[1]
                 for line in (folder / 'time.txt').read_text().splitlines()
                 if ': ' in line and not line.lstrip().startswith('Command being timed:')}
    return record, {name: record[name] for name in (
        'status', 'returncode', 'wall_seconds', 'job_sha256', 'hostname', 'max_cpu_threads',
        'cpu_affinity', 'started_utc', 'finished_utc', 'load_before', 'load_after',
        'maximum_sampled_tree_rss_bytes', 'maximum_sampled_tree_threads', 'resource_samples')}, resources


def verified_source(folder, expected_manifest_sha256):
    manifest = folder / 'source_manifest.json'
    if digest(manifest) != expected_manifest_sha256:
        raise RuntimeError('source manifest differs from the declared final freeze')
    record = load(manifest)
    for name, expected in record['files'].items():
        if digest(folder / name) != expected:
            raise RuntimeError('frozen source changed: ' + name)
    return {'manifest_sha256': expected_manifest_sha256, 'files_verified': len(record['files']),
            'base_commit': record['base_commit'],
            'runtime_delta_from_v30_guard': record['runtime_delta_from_v30_guard'],
            'files': {name: sha for name, sha in record['files'].items()
                      if name.startswith('src/fnit/synthmorph/') or name in (
                          'src/fnit/_nib.py', 'src/fnit/_transforms.py', 'src/fnit/_world_resampling.py')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs-root', 'accepted-root', 'baseline-source', 'candidate-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sources = {'baseline': verified_source(args.baseline_source,
        '358b80ca61f5f5842ff72d0281c2bc86280fd996d07506dd206d2ea455a74111'),
        'candidate': verified_source(args.candidate_source,
        '27151ecb3e7edd14fe191ebbbee71abcf5f798b5273ef503f416db4ac48940f3')}
    report = {'scope': __doc__, 'sources': sources,
              'reference_scope': 'accepted v29 complete CLI and its previously scored official outputs; official CNN not repeated',
              'same_input_preflight': load(args.runs_root / 'resource_preflight.public.json'),
              'CPU_order': ['A1_v34', 'C1_v35', 'C2_v35', 'A2_v34'],
              'cold_scope': 'C1 first use includes empty NumBa disk cache JIT; C2 new process reuses that disk cache; Eigen binary already built',
              'OS_page_cache_flushed': False, 'shared_node': True, 'arms': {},
              'worker_sha256': digest(__file__),
              'scorer_import_dependency_sha256': {name: digest(Path(__file__).parent / name)
                  for name in ('compare_cpu_branch.py', 'build_joint_precision_report.py', 'build_report.py')}}
    all_passed = report['same_input_preflight']['all_gates_passed']
    for arm in ('A1_v34', 'C1_v35', 'C2_v35', 'A2_v34', 'joint_192'):
        folder = args.runs_root / arm
        record, run, resources = selected_run(folder)
        role = 'candidate' if arm.startswith('C') or arm == 'joint_192' else 'baseline'
        expected_source = args.candidate_source if role == 'candidate' else args.baseline_source
        if Path(record['job']['source_manifest']).resolve() != (expected_source / 'source_manifest.json').resolve():
            raise RuntimeError('job used a different source freeze')
        if record['environment']['PYTHONPATH'] != str(expected_source / 'src'):
            raise RuntimeError('actual source import root changed')
        reference = args.accepted_root / ('joint_192' if arm == 'joint_192' else 'joint')
        outputs = {name: image_comparison(reference / (name + '.nii.gz'), folder / (name + '.nii.gz'))
                   for name in ('forward', 'inverse', 'moved', 'fixed_moved')}
        same = all(all(value for key, value in row.items() if key.endswith('_equal'))
                   for row in outputs.values())
        comparison = load(reference / 'nodecw7_reference_comparison.private.json')
        for category in ('images', 'transforms'):
            for name, row in comparison[category].items():
                if digest(reference / (name + '.nii.gz')) != row['candidate_sha256']:
                    raise RuntimeError('accepted CLI no longer matches its official comparison')
        official = gate(comparison, 'joint')
        route = load(folder / 'sampler_route.private.json')
        if route['worker_sha256'] != digest(record['job']['argv'][1]):
            raise RuntimeError('CLI observer source changed')
        for name, actual in route['source_sha256'].items():
            if actual != sources[role]['files']['src/fnit/synthmorph/' + name + '.py']:
                raise RuntimeError('actual imported module differs from its declared freeze: ' + name)
        if role == 'candidate':
            backend = (route['route'].get('backend') == 'numba'
                       and route['route'].get('requested_threads') == 8
                       and route['numba_mask_restored'] is True)
        else:
            backend = route['route']['backend'] is None
        passed = same and official['all_gates_passed'] and backend
        all_passed &= passed
        report['arms'][arm] = {'run': run, 'GNU_time_resources': resources, 'role': role,
            'outputs': outputs, 'exact_accepted_outputs_preserved': same,
            'actual_sampler': route, 'reference_comparison_sha256': digest(reference / 'nodecw7_reference_comparison.private.json'),
            'accepted_official_comparison': comparison, 'fixed_official_gate_transferred': official if same else None,
            'all_gates_passed': passed}
    api = args.runs_root / 'joint_object_api'
    record, run, resources = selected_run(api)
    if Path(record['job']['source_manifest']).resolve() != (args.candidate_source / 'source_manifest.json').resolve():
        raise RuntimeError('materialized API used a different source freeze')
    if record['environment']['PYTHONPATH'] != str(args.candidate_source / 'src'):
        raise RuntimeError('materialized API source import root changed')
    contract = load(api / 'artifacts/materialized.private.json')
    route = load(api / 'artifacts/sampler_route.private.json')
    observer = load(api / 'artifacts/report.private.json')
    observer.pop('source_root', None)
    for name, expected in sources['candidate']['files'].items():
        relative = name.removeprefix('src/fnit/')
        if relative in observer['source_sha256'] and observer['source_sha256'][relative] != expected:
            raise RuntimeError('materialized API imported source changed: ' + relative)
    if (route['worker_sha256'] != digest(record['job']['argv'][1])
            or route['helper_sha256'] != sources['candidate']['files']['src/fnit/synthmorph/_cpu_raw_sampler.py']
            or observer['input_sha256'] != report['same_input_preflight']['data']['input_sha256']):
        raise RuntimeError('materialized API worker, sampler or input changed')
    api_passed = (contract['all_gates_passed'] and route['actual_backend']['backend'] == 'numba'
                  and route['actual_backend']['requested_threads'] == 8 and route['numba_mask_restored'])
    all_passed &= api_passed
    report['materialized_API'] = {'run': run, 'GNU_time_resources': resources,
        'contract': contract, 'actual_sampler': route, 'observer': observer, 'all_gates_passed': api_passed}
    baseline = statistics.median(report['arms'][name]['run']['wall_seconds'] for name in ('A1_v34', 'A2_v34'))
    candidate = statistics.median(report['arms'][name]['run']['wall_seconds'] for name in ('C1_v35', 'C2_v35'))
    report['current_complete_CLI_medians_seconds'] = {'baseline': baseline, 'candidate': candidate,
        'observed_baseline_over_candidate': baseline / candidate,
        'limitation': 'one shared-node ABBA pair, including cold JIT in C1; not a general throughput guarantee'}
    report['all_gates_passed'] = bool(all_passed)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    if not all_passed:
        raise RuntimeError('new complete CPU sampler preservation gate failed')


if __name__ == '__main__':
    main()
