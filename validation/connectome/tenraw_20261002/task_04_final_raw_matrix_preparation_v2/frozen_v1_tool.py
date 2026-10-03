"""CPU-only final raw matrix comparison after immutable v9 origins and 10 official cases exist."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(identity):
    path = Path(identity['path'])
    require(set(identity) == {'path', 'sha256'} and path.is_absolute() and path.is_file() and
            not path.is_symlink() and sha(path) == identity['sha256'], 'immutable JSON identity changed: ' + str(path))
    return json.loads(path.read_bytes())


def create_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def readiness(config):
    for identity in config['source_files']:
        require(sha(identity['path']) == identity['sha256'], 'frozen source bytes changed')
    manifest = bound(config['raw_manifest'])
    mixed = bound(config['mixed_configuration'])
    origin_path = Path(mixed['GPU_origin_bindings'])
    missing = [case['case_id'] for case in manifest['cases']
               if not (Path(config['official_view']) / case['case_id'] / 'reference_manifest.json').is_file()]
    if missing or not origin_path.is_file():
        return {'status': 'waiting_actual_final_origins_or_official', 'missing_official_cases': missing,
                'final_origins_file_available': origin_path.is_file(), 'matrix_comparison_started': False}
    # All receipt/driver/source/output/memory/anatomy guards come from the actual frozen v9 reader.
    sys.path.insert(0, mixed['helper_directory'])
    compare = importlib.import_module('benchmark_connectome_cohort_compare')
    require(Path(compare.__file__).resolve() == Path(mixed['comparison_tool']).resolve(), 'wrong v9 comparator imported')
    from types import SimpleNamespace
    options = SimpleNamespace(**{key: Path(value) for key, value in mixed['comparison_inputs'].items()})
    cases = compare.manifest_cases(manifest)
    bindings, identity = compare.gpu_origins.load_bindings(origin_path, cases, options)
    expected = {(key.split('/')[0], key.split('/')[1]) for key in mixed['selected_pairs']}
    require(set(bindings) == expected and len(bindings) == 10, 'final ten override origins incomplete or duplicated')
    receipt = bound(config['v3_subset_receipt'])
    require(receipt['completed_case_keys'] == mixed['selected_attempts'][0]['completed_case_keys'] and
            receipt['failed_case_keys'] == ['candidate/sub-CON08'],
            'actual failed-v3 subset receipt differs from final declaration')
    atlases = json.loads((options.baseline_root / 'cohort_config.json').read_text())['atlases']
    chains = []
    for case in cases:
        official = Path(config['official_view']) / case['case_id'] / 'reference_manifest.json'
        reference = json.loads(official.read_bytes())
        require(reference.get('state') == 'completed' and reference.get('execution_completed') is True and
                reference.get('seeds') == [0, 1, 2, 3, 4], 'official five-seed execution is incomplete')
        for arm in ('baseline', 'candidate'):
            root, driver, binding = compare.gpu_origins.selected_origin(options, arm, case['case_id'], bindings)
            job = root / arm / case['case_id']
            wall = json.loads((job / 'raw_bids_wall.json').read_bytes())
            subject = compare.gpu_origins.selected_proof(binding, case)[0]['anatomy_subject_dir'] if binding else wall['selected_inputs']['freesurfer_subject_dir']
            _, evidence = compare.validate_gpu_run(root, driver, arm, case, subject, atlases, {}, GPU_origin_binding=binding)
            require(evidence['actual_source']['source_fingerprint'] == config['source_fingerprints'][arm],
                    'actual scientific source fingerprint differs')
            chains.append({'arm': arm, 'case_id': case['case_id'], 'job': str(job), 'GPU_report': evidence['gpu_report'],
                           'wall_report': evidence['wall_report'], 'official_manifest': {'path': str(official), 'sha256': sha(official)},
                           'qualified_origin': evidence})
    require(len(chains) == 20 and len({(item['arm'], item['case_id']) for item in chains}) == 20, 'twenty unique chains required')
    require(sha(origin_path) == identity['sha256'], 'final origin map changed during qualification')
    return {'status': 'ready_twenty_actual_qualified_chains', 'GPU_origin_bindings': identity, 'chains': chains,
            'matrix_comparison_started': False}


def execute(config, ready, output):
    sys.path.insert(0, str(Path(config['raw_tool']).parent))
    raw = importlib.import_module('benchmark_connectome_raw_cohort_envelope')
    require(Path(raw.__file__).resolve() == Path(config['raw_tool']).resolve(), 'wrong raw envelope module imported')
    summaries = []
    for chain in ready['chains']:
        target = output / (chain['arm'] + '_' + chain['case_id'])
        target.mkdir()
        create_json(target / 'actual_inputs.json', chain)
        started = time.monotonic()
        job = Path(chain['job'])
        report = raw.compare(Path(config['official_view']) / chain['case_id'], [job / 'connectome'],
                             [Path(chain['GPU_report']['path'])], Path(config['raw_manifest']['path']),
                             config['raw_manifest']['sha256'], chain['case_id'], fnit_seeds=[0])
        require(len(report['profiles']) == 8 and report['fnit_reproducibility_status'] == 'not_assessed',
                'single actual FNIT seed must remain not_assessed for self')
        create_json(target / 'envelope.json', report)
        result = {'arm': chain['arm'], 'case_id': chain['case_id'], 'wall_seconds': time.monotonic() - started,
                  'matrix_envelope_status': report['matrix_envelope_status'], 'FNIT_self': report['fnit_reproducibility_status'],
                  'accepted': sum(v['accepted_count'] for p in report['profiles'].values() for v in p['ranges'].values()),
                  'total': sum(len(v['comparison_accepted']) for p in report['profiles'].values() for v in p['ranges'].values()),
                  'envelope_sha256': sha(target / 'envelope.json')}
        require(result['total'] == 240, 'expected eight atlases, six fields, five cross pairs')
        create_json(target / 'execution.json', result)
        summaries.append(result)
    for chain in ready['chains']:
        for name in ('GPU_report', 'wall_report', 'official_manifest'):
            require(sha(chain[name]['path']) == chain[name]['sha256'], 'bound report changed during final comparison')
    require(sha(ready['GPU_origin_bindings']['path']) == ready['GPU_origin_bindings']['sha256'], 'origin map changed')
    return {'status': 'completed_twenty_actual_raw_matrix_comparisons', 'runs': summaries, 'full_ten_scientific_match':
            all(item['matrix_envelope_status'] == 'passed' for item in summaries), 'FNIT_self': 'not_assessed',
            'population': 'not_assessed', 'MRI_or_GPU_started': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--timeout-hours', type=float, default=72)
    args = parser.parse_args(argv)
    require(not os.environ.get('CUDA_VISIBLE_DEVICES'), 'CPU-only empty CUDA_VISIBLE_DEVICES required')
    require(1 <= args.poll_seconds <= 60 and 0 < args.timeout_hours <= 168, 'invalid observation interval')
    config = bound({'path': str(args.config), 'sha256': args.config_sha256})
    output = args.output_root
    require(output.is_absolute() and not output.exists() and not output.is_symlink(), 'fresh immutable output namespace required')
    protected = [Path(config['official_view']), Path(config['raw_manifest']['path']).parent,
                 Path(config['mixed_configuration']['path']).parent, *[Path(x['path']).parent for x in config['source_files']]]
    mixed = bound(config['mixed_configuration'])
    protected.extend(Path(mixed['comparison_inputs'][key]) for key in ('baseline_root', 'candidate_root', 'baseline_anatomy_root'))
    for path in protected:
        require(not output.resolve().is_relative_to(path.resolve()) and not path.resolve().is_relative_to(output.resolve()),
                'output overlaps protected input/source namespace')
    output.mkdir(parents=True)
    create_json(output / 'configuration.json', config)
    started = time.monotonic()
    try:
        count = 0
        while True:
            ready = readiness(config)
            create_json(output / f'observation_{count:06d}.json', ready)
            count += 1
            if ready['status'] == 'ready_twenty_actual_qualified_chains':
                if not args.check_only:
                    create_json(output / 'final_report.json', execute(config, ready, output))
                break
            if args.check_only or not args.watch or time.monotonic() - started >= args.timeout_hours * 3600:
                break
            time.sleep(args.poll_seconds)
        print(json.dumps({'status': ready['status'], 'matrix_comparison_started':
                         ready['status'] == 'ready_twenty_actual_qualified_chains' and not args.check_only}))
    except Exception as error:
        create_json(output / 'failure.json', {'type': type(error).__name__, 'message': str(error),
                    'elapsed_wall_seconds': time.monotonic() - started, 'MRI_or_GPU_started': False})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
