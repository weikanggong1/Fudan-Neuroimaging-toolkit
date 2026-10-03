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


def protect_namespace(output, protected):
    output = Path(output).resolve()
    for item in protected:
        item = Path(item).resolve()
        require(not output.is_relative_to(item) and not item.is_relative_to(output),
                'output overlaps protected namespace: ' + str(item))


def initial_protected_paths(config, mixed, canonical):
    paths = [Path(config['official_view']), Path(config['raw_manifest']['path']).parent,
             Path(config['mixed_configuration']['path']).parent]
    paths += [Path(item['path']).resolve().parent for case in canonical['cases'] for item in case['input_files']]
    paths += [Path(item['path']).resolve().parent for item in config['source_files']]
    paths += [Path(mixed['comparison_inputs'][key]) for key in ('baseline_root', 'candidate_root', 'baseline_anatomy_root')]
    paths += [Path(item) for item in config.get('selected_protected_roots', [])]
    for key in ('official_origin_binding', 'official_mapper'):
        if key in config:
            paths.append(Path(config[key]['path']).resolve().parent)
    if 'official_origin_binding' in config:
        routing = bound(config['official_origin_binding'])
        for item in routing.get('cases', {}).values():
            for key in ('actual_case_root', 'actual_planned_case_root'):
                if item.get(key):
                    actual = Path(item[key]).resolve()
                    paths += [actual, actual.parent]
            for record in item.get('producer_contracts', {}).values():
                directory = Path(record['path']).resolve().parent
                paths += [directory, directory.parent]
        for launch in routing.get('launch_bindings', {}).values():
            if launch.get('configuration'):
                paths.append(Path(launch['configuration']['output_root']).resolve())
            if launch.get('configuration_identity'):
                paths.append(Path(launch['configuration_identity']['path']).resolve().parent)
    return paths


def immutable_snapshot(config, ready):
    """Bind original/static reports, complete configs and numerical source files."""
    records = {}
    historical_observations = set()
    if config.get('historical_observation_JSONs'):
        mixed = bound(config['mixed_configuration'])
        declared = {item['path']: item['sha256'] for item in mixed['static_JSON_bindings']}
        for item in config['historical_observation_JSONs']:
            require(set(item) == {'path', 'sha256', 'role'} and
                    item['role'] == 'historical_prior_comparison_observation' and
                    declared.get(item['path']) == item['sha256'],
                    'historical observation must retain exact frozen mixed declaration')
            historical_observations.add(str(Path(item['path']).resolve()))
    def add(path, digest):
        path = str(Path(path).resolve())
        require(path not in records or records[path] == digest, 'conflicting immutable file identity: ' + path)
        if path in records:
            return
        require(Path(path).is_file() and sha(path) == digest, 'bound file changed before matrix writes: ' + path)
        records[path] = digest
    def walk(value):
        if isinstance(value, dict):
            # Historical mutable controller snapshots are preserved in the map,
            # while the audit binds their actual current observations separately.
            if 'content' in value and 'policy' in value and 'path' in value:
                return
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                add(value['path'], value['sha256'])
            for key, child in value.items():
                if isinstance(key, str) and key.startswith('/') and isinstance(child, str) and len(child) == 64:
                    add(key, child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(config)
    walk(ready)
    source_inventory = set()
    for chain in ready['chains']:
        source = chain['qualified_origin']['actual_source']
        for relative, digest in source['source_sha256'].items():
            original = Path(source['directory']) / relative
            add(original, digest)
            source_inventory.add(str(original.resolve()))
    # Expand bound configs/reports so their own resource/source/JSON ledgers are
    # also fixed. No MRI arrays, solver or GPU is loaded by this metadata walk.
    visited = set()
    while True:
        # The frozen scientific tree includes repository inventory manifests.
        # Hash those source bytes fully, but do not reinterpret inventory paths
        # (e.g. relative license paths from packaging) as consumed runtime files.
        # Actual launch/resource/config/report ledgers outside that inventory
        # continue to expand recursively.
        pending = [path for path in records if path not in visited and path not in source_inventory and
                   path not in historical_observations and
                   Path(path).suffix == '.json']
        if not pending:
            break
        for path in pending:
            visited.add(path)
            try:
                walk(json.loads(Path(path).read_bytes()))
            except ValueError as error:
                raise ValueError(str(error) + ' while expanding bound JSON ' + path) from error
    return records


def ready_protected_paths(ready, snapshot):
    # Protect every bound file itself. Protect actual code and config/report
    # directories as namespaces; a single shared-root archive/resource file
    # does not turn its whole storage root into a scientific producer tree.
    paths = [Path(path) for path in snapshot]
    names = ('config', 'manifest', 'report', 'status', 'receipt', 'binding', 'origin')
    paths += [Path(path).parent for path in snapshot if Path(path).suffix == '.py' or
              (Path(path).suffix == '.json' and any(name in Path(path).name for name in names))]
    for chain in ready['chains']:
        job = Path(chain['job'])
        paths += [job, job.parent.parent, Path(chain['qualified_origin']['actual_GPU_root']),
                  Path(chain['qualified_origin']['actual_source']['directory'])]
        # reference_manifest resolves through a legitimate mapper routing link
        # to the original producer directory, which remains protected.
        actual = Path(chain['official_manifest']['path']).resolve().parent
        paths += [actual, actual.parent]
    for item in ready['official_origin_audit']['cases'].values():
        actual = Path(item['actual_case_root'])
        paths += [actual, actual.parent]
        for contract in item['producer_contracts'].values():
            directory = Path(contract['path']).resolve().parent
            paths += [directory, directory.parent]
    return paths


def verify_snapshot(snapshot):
    for path, digest in snapshot.items():
        require(Path(path).is_file() and sha(path) == digest, 'immutable snapshot changed at final verification: ' + path)


def verify_controller_observations(ready):
    """Recheck selected rows and routing; historical waiting controllers may poll."""
    audit = ready['official_origin_audit']
    routing = audit['explicit_origin_map']
    owners = {case: item['controller_binding'] for case, item in routing['cases'].items()}
    fixed = ('scope', 'raw_manifest_sha256', 'dataset', 'snapshot', 'official_dwi_root',
             'official_anatomy_root', 'workers', 'downstream_threads', 'tracking_threads',
             'controller_sha256', 'worker_sha256', 'helper_sha256', 'matrix_helper_sha256',
             'verified_reference_manifest_sha256')
    observations = []
    for observation in audit['current_controller_observations']:
        current_bytes = Path(observation['path']).read_bytes()
        current = json.loads(current_bytes)
        before = observation['content']
        require(all(current.get(key) == before.get(key) for key in fixed) and
                set(current['cases']) == set(before['cases']), 'official controller workload changed')
        for case, row in current['cases'].items():
            if owners.get(case) == observation['controller_binding']:
                require(row == before['cases'][case] and row['state'] == 'completed',
                        'selected completed official controller record changed at final verification')
            else:
                require(row['state'] not in ('running', 'completed'),
                        'another official controller started explicitly reassigned case')
        observations.append({'path': observation['path'], 'before_sha256': observation.get('sha256'),
                             'after_sha256': hashlib.sha256(current_bytes).hexdigest(),
                             'controller_binding': observation['controller_binding'],
                             'selected_completed_rows_equal': True, 'reassigned_case_not_dispatched': True})
    return observations


def audit_official_origin_map(config, canonical):
    """Re-audit the frozen explicit mapper result; never publish or change links."""
    require('official_origin_binding' in config and 'official_mapper' in config,
            'final explicit per-case official origin map and mapper identities are required')
    identity = config['official_origin_binding']
    routing = bound(identity)
    require(routing.get('schema_version') == 1 and
            routing.get('scope') == 'explicit actual per-case official source hand-off only' and
            routing.get('state') == 'completed' and routing.get('execution_completed') is True and
            routing.get('scientific_parity') == 'not_assessed',
            'completed explicit official origin map required; old group view cannot replace it')
    ids = {case['case_id'] for case in canonical['cases']}
    require(len(ids) == 10 and set(routing['cases']) == ids, 'official routing must cover exact ten canonical cases')
    mapper_file = Path(config['official_mapper']['path'])
    require(mapper_file.name == 'bind_connectome_raw_reference_case_map.py' and
            sha(mapper_file) == config['official_mapper']['sha256'] and not mapper_file.is_symlink(),
            'pinned actual per-case mapper changed')
    recorded_sources = routing['source_identity']
    mapper_binding = recorded_sources.get('binding_source:' + str(mapper_file.resolve()))
    require(mapper_binding and mapper_binding['sha256'] == config['official_mapper']['sha256'],
            'routing was produced by another mapper')
    sys.path.insert(0, str(mapper_file.parent))
    mapper = importlib.import_module('bind_connectome_raw_reference_case_map')
    require(Path(mapper.__file__).resolve() == mapper_file.resolve(), 'wrong official mapper imported')
    actual_config = recorded_sources['configuration']
    specification, raw, launches, identities = mapper.configuration(
        Path(actual_config['path']), actual_config['sha256'])
    require(identities == recorded_sources and raw == canonical and
            identities['raw_manifest']['sha256'] == config['raw_manifest']['sha256'],
            'official mapper config/source/canonical identities differ')
    root = Path(config['official_view'])
    require(Path(identity['path']).parent.resolve() == root.resolve(), 'origin map belongs to another controlled view')
    selected11 = launches[specification['case_bindings']['sub-CON11']]
    require(selected11['state'] == 'configured' and selected11['configuration']['case_ids'] == ['sub-CON11'],
            'CON11 must come from explicit new CON11-only launch, not old group waiting11')
    audits = {}
    current_controllers = []
    for name, launch in launches.items():
        require(launch['state'] == 'configured', 'all final launch origins must be configured')
        stored = routing['launch_bindings'][name]
        for key in ('configuration', 'runtime_sources', 'controller_status_path', 'mode', 'configuration_identity'):
            require(stored[key] == launch[key], 'official launch contract changed')
        controller_path = Path(launch['controller_status_path'])
        current_bytes = controller_path.read_bytes()
        current = json.loads(current_bytes)
        current_controllers.append({'path': str(controller_path), 'sha256': hashlib.sha256(current_bytes).hexdigest(),
                                    'controller_binding': name, 'content': current,
                                    'policy': 'mutable controller observation; selected completed rows and routing rechecked at completion'})
        original = stored['controller_snapshot']['content']
        for case, row in current['cases'].items():
            require(specification['case_bindings'].get(case) == name or row['state'] not in ('running', 'completed'),
                    'another official controller executes explicitly reassigned case')
            if specification['case_bindings'].get(case) == name:
                require(row == original['cases'][case] and row['state'] == 'completed',
                        'selected completed official controller record changed')
    for case in sorted(ids):
        item = routing['cases'][case]
        require(item['state'] == 'completed' and item['controller_binding'] == specification['case_bindings'][case],
                'official case is not actually completed by its explicit controller')
        launch = launches[item['controller_binding']]
        target = Path(item['actual_case_root'])
        link = root / case
        require(target == Path(launch['configuration']['output_root']) / case and
                not target.is_symlink() and target.absolute() == target.resolve() and link.is_symlink() and
                link.resolve() == target.resolve() and item['controlled_link']['path'] == str(link) and
                Path(item['controlled_link']['actual_target']).resolve() == target.resolve(),
                'controlled symlink route differs from original actual producer directory')
        controller = json.loads(Path(launch['controller_status_path']).read_bytes())
        verified = mapper.completed_case(case, controller['cases'][case],
            Path(launch['configuration']['output_root']), specification, launch['runtime_sources'])
        require(all(item[key] == value for key, value in verified.items()), 'official manifest/producer binding changed')
        manifest = json.loads(Path(item['reference_manifest']['path']).read_bytes())
        mapper.verify_manifest_contract(manifest, raw, specification, identities, launch)
        require(len(manifest['commands']) == len(manifest['completed_commands']) == 198 and
                all(all(executed.get(key) == value for key, value in planned.items()) and executed['returncode'] == 0
                    for planned, executed in zip(manifest['commands'], manifest['completed_commands'])),
                'official exact planned/executed commands changed')
        manifest['_actual_manifest_path'] = item['reference_manifest']['path']
        files = mapper.referenced_files(manifest)
        require(files == item['file_verification'], 'full official raw/producer/FS/reader/TCK/scalar/matrix audit changed')
        audits[case] = {'actual_case_root': str(target), 'reference_manifest': item['reference_manifest'],
                       'producer_contracts': item['producer_contracts'], 'file_verification': files}
    require(sha(identity['path']) == identity['sha256'], 'official origin map changed during audit')
    return {'origin_binding': identity, 'source_identity': identities,
            'explicit_origin_map': routing, 'cases': audits, 'current_controller_observations': current_controllers,
            'routing_is_metadata_only': True}


def readiness(config):
    for identity in config['source_files']:
        require(sha(identity['path']) == identity['sha256'], 'frozen source bytes changed')
    manifest = bound(config['raw_manifest'])
    mixed = bound(config['mixed_configuration'])
    require('official_origin_binding' in config and 'official_mapper' in config,
            'final comparison requires explicit per-case mapper handoff; old nine-case config cannot launch')
    origin_path = Path(mixed['GPU_origin_bindings'])
    missing = [case['case_id'] for case in manifest['cases']
               if not (Path(config['official_view']) / case['case_id'] / 'reference_manifest.json').is_file()]
    official_map_available = Path(config['official_origin_binding']['path']).is_file()
    if missing or not origin_path.is_file() or not official_map_available:
        return {'status': 'waiting_actual_final_origins_or_official', 'missing_official_cases': missing,
                'final_origins_file_available': origin_path.is_file(), 'official_origin_map_available': official_map_available,
                'matrix_comparison_started': False}
    official_audit = audit_official_origin_map(config, manifest)
    # All receipt/driver/source/output/memory/anatomy guards come from the actual frozen v9 reader.
    sys.path.insert(0, mixed['helper_directory'])
    compare = importlib.import_module('benchmark_connectome_cohort_compare')
    require(Path(compare.__file__).resolve() == Path(mixed['comparison_tool']).resolve(), 'wrong v9 comparator imported')
    from types import SimpleNamespace
    options = SimpleNamespace(**{key: Path(value) for key, value in mixed['comparison_inputs'].items()})
    cases = compare.manifest_cases(manifest)
    # v9 load_bindings reuses its identity variable while checking historical
    # output files. Bind the actual map independently before calling it, rather
    # than treating that helper's returned last-file identity as the map.
    _, identity = compare.safe_json(origin_path)
    bindings, _helper_returned_identity = compare.gpu_origins.load_bindings(origin_path, cases, options)
    expected = {(key.split('/')[0], key.split('/')[1]) for key in mixed['selected_pairs']}
    require(set(bindings) == expected and len(bindings) == 10, 'final ten override origins incomplete or duplicated')
    receipt = bound(config['v3_subset_receipt'])
    require(receipt['completed_case_keys'] == mixed['selected_attempts'][0]['completed_case_keys'] and
            receipt['failed_case_keys'] == ['candidate/sub-CON08'],
            'actual failed-v3 subset receipt differs from final declaration')
    atlases = json.loads((options.baseline_root / 'cohort_config.json').read_text())['atlases']
    chains = []
    required_file_sha256 = {}
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
            touched = {}
            _, evidence = compare.validate_gpu_run(root, driver, arm, case, subject, atlases, touched, GPU_origin_binding=binding)
            for path, digest in touched.items():
                require(path not in required_file_sha256 or required_file_sha256[path] == digest,
                        'conflicting actual required output identity across chains')
                required_file_sha256[path] = digest
            require(evidence['actual_source']['source_fingerprint'] == config['source_fingerprints'][arm],
                    'actual scientific source fingerprint differs')
            chains.append({'arm': arm, 'case_id': case['case_id'], 'job': str(job), 'GPU_report': evidence['gpu_report'],
                           'wall_report': evidence['wall_report'], 'official_manifest': {'path': str(official), 'sha256': sha(official)},
                           'qualified_origin': evidence, 'required_file_sha256': touched})
    require(len(chains) == 20 and len({(item['arm'], item['case_id']) for item in chains}) == 20, 'twenty unique chains required')
    require(sha(origin_path) == identity['sha256'], 'final origin map changed during qualification')
    return {'status': 'ready_twenty_actual_qualified_chains', 'GPU_origin_bindings': identity, 'chains': chains,
            'official_origin_audit': official_audit, 'required_file_sha256': required_file_sha256,
            'matrix_comparison_started': False}


def execute(config, ready, output):
    snapshot = immutable_snapshot(config, ready)
    protect_namespace(output, ready_protected_paths(ready, snapshot))
    create_json(output / 'immutable_input_snapshot.json', snapshot)
    sys.path.insert(0, str(Path(config['raw_tool']).parent))
    raw = importlib.import_module('benchmark_connectome_raw_cohort_envelope')
    require(Path(raw.__file__).resolve() == Path(config['raw_tool']).resolve(), 'wrong raw envelope module imported')
    summaries = []
    for chain in ready['chains']:
        target = output / (chain['arm'] + '_' + chain['case_id'])
        protect_namespace(output, ready_protected_paths(ready, snapshot))
        target.mkdir()
        protect_namespace(output, ready_protected_paths(ready, snapshot))
        create_json(target / 'actual_inputs.json', chain)
        started = time.monotonic()
        job = Path(chain['job'])
        report = raw.compare(Path(config['official_view']) / chain['case_id'], [job / 'connectome'],
                             [Path(chain['GPU_report']['path'])], Path(config['raw_manifest']['path']),
                             config['raw_manifest']['sha256'], chain['case_id'], fnit_seeds=[0])
        require(len(report['profiles']) == 8 and report['fnit_reproducibility_status'] == 'not_assessed',
                'single actual FNIT seed must remain not_assessed for self')
        protect_namespace(output, ready_protected_paths(ready, snapshot))
        create_json(target / 'envelope.json', report)
        result = {'arm': chain['arm'], 'case_id': chain['case_id'], 'wall_seconds': time.monotonic() - started,
                  'matrix_envelope_status': report['matrix_envelope_status'], 'FNIT_self': report['fnit_reproducibility_status'],
                  'accepted': sum(v['accepted_count'] for p in report['profiles'].values() for v in p['ranges'].values()),
                  'total': sum(len(v['comparison_accepted']) for p in report['profiles'].values() for v in p['ranges'].values()),
                  'envelope_sha256': sha(target / 'envelope.json')}
        require(result['total'] == 240, 'expected eight atlases, six fields, five cross pairs')
        create_json(target / 'execution.json', result)
        summaries.append(result)
    require(sha(ready['official_origin_audit']['origin_binding']['path']) ==
            ready['official_origin_audit']['origin_binding']['sha256'], 'official origin map changed during comparison')
    for chain in ready['chains']:
        for name in ('GPU_report', 'wall_report', 'official_manifest'):
            require(sha(chain[name]['path']) == chain[name]['sha256'], 'bound report changed during final comparison')
    require(sha(ready['GPU_origin_bindings']['path']) == ready['GPU_origin_bindings']['sha256'], 'origin map changed')
    verify_snapshot(snapshot)
    verify_controller_observations(ready)
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
    require('official_origin_binding' in config and 'official_mapper' in config and
            isinstance(config.get('selected_protected_roots'), list) and config['selected_protected_roots'] and
            all(Path(root).is_absolute() for root in config['selected_protected_roots']),
            'v2 requires actual explicit official mapper identities and selected actual roots')
    output = args.output_root
    require(output.is_absolute() and not output.exists() and not output.is_symlink(), 'fresh immutable output namespace required')
    mixed = bound(config['mixed_configuration'])
    canonical = bound(config['raw_manifest'])
    protect_namespace(output, initial_protected_paths(config, mixed, canonical))
    output.mkdir(parents=True)
    create_json(output / 'configuration.json', config)
    started = time.monotonic()
    try:
        count = 0
        while True:
            ready = readiness(config)
            ready['configuration_identity'] = {'path': str(args.config), 'sha256': args.config_sha256}
            if ready['status'] == 'ready_twenty_actual_qualified_chains':
                snapshot = immutable_snapshot(config, ready)
                protect_namespace(output, ready_protected_paths(ready, snapshot))
            create_json(output / f'observation_{count:06d}.json', ready)
            count += 1
            if ready['status'] == 'ready_twenty_actual_qualified_chains':
                if args.check_only:
                    verify_snapshot(snapshot)
                if not args.check_only:
                    result = execute(config, ready, output)
                    verify_snapshot(json.loads((output / 'immutable_input_snapshot.json').read_bytes()))
                    result['official_controller_end_observations'] = verify_controller_observations(ready)
                    create_json(output / 'final_report.json', result)
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
