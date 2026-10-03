"""逐病例绑定真实官方参考出处：旧病例复用、新 namespace 明确交接；不运行科学计算。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

try:
    from .bind_connectome_raw_reference_origins import checked_record, completed_case, require, sha256, atomic
except ImportError:
    from bind_connectome_raw_reference_origins import checked_record, completed_case, require, sha256, atomic

RUNTIME_NAMES = {
    'controller_sha256': 'run_connectome_raw_official_cohort.py',
    'worker_sha256': 'benchmark_connectome_raw_official.py',
    'helper_sha256': 'benchmark_connectome_repeats_official.py',
    'matrix_helper_sha256': 'connectome_repeat_common.py',
}
MATRIX_NAMES = ('count', 'sift2_fbc', 'mean_length', 'mean_fa')
ATLASES = {'fs-aparc', 'aparc+tian-s1', 'aparc.a2009s+tian-s1', 'glasser+tian-s1',
           'glasser+tian-s4', 'schaefer200+tian-s1', 'schaefer500+tian-s4', 'schaefer1000+tian-s4'}
CONTRACT_SCOPES = {'anatomy_contract': 'official_self_produced_fresh_fs_anatomy_and_raw_dwi_atlases',
                   'official_dwi_contract': 'official_self_produced_raw_dwi_chain'}


def configuration(path, digest):
    identity = checked_record({'path': str(path), 'sha256': digest})
    config = json.loads(Path(path).read_text())
    raw_record = checked_record(config['raw_manifest'])
    raw = json.loads(Path(raw_record['path']).read_text())
    canonical = [case['case_id'] for case in raw['cases']]
    require(len(canonical) == len(set(canonical)) == 10 and set(config['case_bindings']) == set(canonical),
            'explicit map must cover the exact ten canonical cases')
    require(config['seeds'] == [0, 1, 2, 3, 4] and config['n_seeds'] == 100000 and
            config['expected_eddy_solver'] == 'cpu', 'frozen five-repeat CPU scientific workload required')
    binding_sources = {filename: checked_record({'path': filename, 'sha256': value})
                       for filename, value in config['binding_source_files'].items()}
    require({str(Path(__file__).resolve()), str(Path(require.__code__.co_filename).resolve())} ==
            set(binding_sources), 'actual mapper and imported origin helper must be frozen')
    reference = checked_record(config['verified_reference_manifest'])
    used = set(config['case_bindings'].values())
    require(used == set(config['launch_bindings']), 'all controller bindings must be explicitly used')
    require(config['expected_parameters']['n_seed_attempts'] == config['n_seeds'] and
            config['expected_parameters']['tracking_threads'] == 0, 'explicit scientific parameters required')
    launches, common_runtime, output_roots = {}, None, set()
    for name, entry in config['launch_bindings'].items():
        if entry.get('state') == 'pending_configuration':
            require(set(entry) == {'state'}, 'pending producer binding cannot invent output paths or hashes')
            launches[name] = {'state': 'pending_configuration'}
            continue
        launch_identity = checked_record(entry['launch_configuration'])
        launch = json.loads(Path(entry['launch_configuration']['path']).read_text())
        require(Path(launch['raw_manifest']).resolve() == Path(raw_record['path']).resolve() and
                Path(launch['verified_reference_manifest']).resolve() == Path(reference['path']).resolve() and
                launch['raw_manifest_sha256'] == raw_record['sha256'] and
                launch['verified_reference_manifest_sha256'] == reference['sha256'] and
                launch['n_seeds'] == config['n_seeds'] and launch['seeds'] == config['seeds'],
                'actual explicit controller configuration differs from raw/scientific workload')
        sources = {}
        for filename, value in launch['source_files'].items():
            checked_record({'path': filename, 'sha256': value})
            if Path(filename).name in RUNTIME_NAMES.values():
                require(Path(filename).name not in sources, 'runtime source names must be unambiguous')
                sources[Path(filename).name] = value
        require(set(sources) == set(RUNTIME_NAMES.values()), 'actual controller/worker/helpers required')
        require(sources == config['expected_runtime_sources'] and
                (common_runtime is None or common_runtime == sources), 'per-case launches must keep the same frozen scientific runtime')
        for filename in launch['source_files']:
            if Path(filename).name in RUNTIME_NAMES.values():
                require(Path(filename).absolute() == Path(filename).resolve(), 'frozen runtime source cannot be a substituted alias')
        common_runtime = sources
        output = Path(launch['output_root'])
        require(output.absolute() == output.resolve() and str(output.resolve()) not in output_roots, 'actual controller roots must be independent and not aliases')
        output_roots.add(str(output.resolve()))
        assigned = {case for case, owner in config['case_bindings'].items() if owner == name}
        require(assigned.issubset(set(launch['case_ids'])), 'explicit assigned case absent from controller launch')
        status = Path(launch['output_root']) / 'cohort_status.json'
        require(Path(entry['controller_status_path']).resolve() == status.resolve(),
                'explicit controller status is not the actual launch output')
        launches[name] = {'state': 'configured', 'configuration': launch, 'runtime_sources': sources,
                          'controller_status_path': str(status), 'mode': entry['mode'],
                          'configuration_identity': launch_identity}
    return config, raw, launches, {'configuration': identity, 'raw_manifest': raw_record,
                                    'verified_reference_manifest': reference,
                                    **{'binding_source:' + key: value for key, value in binding_sources.items()}}


def referenced_files(manifest):
    """Recheck consumed source and saved output bytes; no tensor/metric computation."""
    records = {}
    def add_record(item):
        if isinstance(item, dict):
            if 'path' in item and 'sha256' in item:
                add(item['path'], item['sha256'])
            for value in item.values():
                add_record(value)
        elif isinstance(item, list):
            for value in item:
                add_record(value)
    def add(path, digest):
        resolved = str(Path(path).resolve())
        require(resolved not in records or records[resolved] == digest, 'conflicting actual file identities')
        records[resolved] = digest
    for item in manifest['raw_case_binding']['files']:
        add(item['path'], item['sha256'])
    add(manifest['raw_case_binding']['official_rawprep_report'],
        manifest['raw_case_binding']['official_rawprep_report_sha256'])
    source = manifest['source']
    for name in ('anatomy_contract', 'official_dwi_contract'):
        item = source[name]; add(item['path'], item['sha256'])
        contract = json.loads(Path(item['path']).read_text())
        require(contract.get('state') == 'completed' and contract.get('scope') == CONTRACT_SCOPES[name] and
                contract.get('case_id') == manifest['case_id'],
                'actual producer contract no longer completed/same case')
    anatomy = json.loads(Path(source['anatomy_contract']['path']).read_text())
    require(Path(anatomy['official_dwi_contract']['path']).resolve() ==
            Path(source['official_dwi_contract']['path']).resolve() and
            anatomy['official_dwi_contract']['sha256'] == source['official_dwi_contract']['sha256'],
            'anatomy must reference this exact actual DWI producer contract')
    for key in ('prepared_report', 'official_anatomy_report'):
        add_record(anatomy[key])
    add_record(source.get('raw_t1w'))
    add_record(source.get('fresh_fs_origin'))
    for item in source['images'].values():
        add(item['path'], item['sha256'])
    for profile in source['profiles'].values():
        add(profile['image']['path'], profile['image']['sha256'])
        add(profile['nodes_path'], profile['nodes_sha256'])
    for item in manifest['programs'].values():
        add(item['path'], item['sha256'])
    for item in manifest['completed_commands']:
        require(Path(item['log']).is_file() and Path(item['time_file']).is_file(), 'actual command log/time missing')
        if item['stage'] == 'input_readback':
            add(item['json'], manifest['input_readbacks'][item['input']]['mrinfo_json_sha256'])
    root = Path(manifest['_actual_manifest_path']).parent
    for seed in manifest['seeds']:
        directory = root / f'seed-{seed}'
        output = manifest['outputs'][str(seed)]
        add(directory / 'tracks.tck', output['tracks_sha256'])
        for name, item in output['scalars'].items():
            add(directory / name, item['sha256'])
        require(set(output['profiles']) == set(source['profiles']), 'actual output atlas set differs')
        for name, profile in output['profiles'].items():
            actual = directory / 'atlases' / name
            require(Path(profile['directory']).resolve() == actual.resolve(), 'matrix belongs to another actual case/seed')
            require(set(profile['matrix_sha256']) == set(MATRIX_NAMES), 'all four actual matrices required')
            original_profile = source['profiles'][name]
            require(profile['nodes'] == original_profile['nodes'] and
                    profile['node_rows'] == original_profile['node_rows'] and
                    profile['atlas_sha256'] == original_profile['image']['sha256'],
                    'saved atlas/node semantics differ from the actual source')
            count_file, atlas_file = actual / 'nodes.txt', actual / 'atlas.sha256'
            if count_file.is_file():
                require(int(count_file.read_text().strip()) == profile['nodes'], 'saved node count differs')
                add(count_file, sha256(count_file))
            if atlas_file.is_file():
                require(atlas_file.read_text().split()[0] == profile['atlas_sha256'], 'saved atlas metadata differs')
                add(atlas_file, sha256(atlas_file))
            alignment = profile.get('canonical_matrix_alignment')
            if alignment is not None:
                alignment_file = actual / 'canonical_matrix_alignment.json'
                require(json.loads(alignment_file.read_text()) == alignment and
                        alignment['canonical_matrix_sha256'] == profile['matrix_sha256'], 'actual canonical alignment metadata differs')
                add(alignment_file, sha256(alignment_file))
                for matrix, digest in alignment['raw_matrix_sha256'].items():
                    add(actual / (matrix + '.csv'), digest)
            prefix = 'connectome_' if alignment is not None else ''
            for matrix, digest in profile['matrix_sha256'].items():
                add(actual / (prefix + matrix + '.csv'), digest)
            add(actual / 'nodes.tsv', profile['nodes_tsv_sha256'])
    for path, digest in records.items():
        checked_record({'path': path, 'sha256': digest})
    return {'actual_unique_files_verified': len(records),
            'scope': 'consumed raw/source/FS/reader and saved TCK/scalar/matrix/node file bytes',
            'verified_file_sha256': records}


def verify_manifest_contract(manifest, raw, config, identities, launch):
    require(manifest['scope'] == 'independent official raw DWI plus fresh FS anatomy tracking/downstream' and
            manifest['parameters'] == config['expected_parameters'] and
            set(manifest['source']['profiles']) == ATLASES, 'actual independent raw scientific profile differs')
    canonical = next(item for item in raw['cases'] if item['case_id'] == manifest['case_id'])
    binding = manifest['raw_case_binding']
    t1 = manifest['source']['raw_t1w']
    require(any(item['kind'] == 'raw_t1w' and Path(item['path']).resolve() == Path(t1['path']).resolve() and
                item['sha256'] == t1['sha256'] for item in canonical['input_files']),
            'official anatomy raw T1 differs from this canonical case')
    require(binding['files'] == canonical['input_files'] and
            Path(binding['manifest_path']).resolve() == Path(identities['raw_manifest']['path']).resolve() and
            all(binding[key] == raw[key] for key in ('dataset', 'snapshot', 'license')),
            'actual original raw case file identities differ from canonical')
    actual = {str(Path(path).resolve()): digest for path, digest in
              binding['rawprep_producer_identity']['input_sha256'].items()}
    expected = {str(Path(item['path']).resolve()): item['sha256'] for item in canonical['input_files']
                if item['kind'] != 'dataset_description'}
    require(actual == expected and binding['rawprep_producer_identity']['subject'] == canonical['subject'] and
            binding['rawprep_producer_identity']['session'] == canonical['session'],
            'official rawprep raw subject/session/input identities differ')
    original = json.loads(Path(binding['official_rawprep_report']).read_text())
    require(original == binding['rawprep_producer_identity'] and original['completed'] is True and
            not original.get('error') and all(row['returncode'] == 0 for row in original['commands']),
            'official rawprep actual report changed or incomplete')
    reference = json.loads(Path(identities['verified_reference_manifest']['path']).read_text())
    require(reference['execution_completed'] is True and manifest['programs'] == reference['programs'] and
            manifest['mrtrix_version'] == reference['mrtrix_version'] and
            manifest['reference_command_helper_sha256'] == reference['script_sha256'] and
            manifest['matrix_helper_sha256'] == reference['helper_sha256'],
            'actual official programs/version/helpers differ from pinned reference identity')
    require(Path(manifest['verified_reference_identity']['path']).resolve() ==
            Path(identities['verified_reference_manifest']['path']).resolve() and
            manifest['verified_reference_identity']['sha256'] == identities['verified_reference_manifest']['sha256'],
            'actual pinned reference manifest identity differs')
    for name, entry in manifest['programs'].items():
        require(Path(entry['invoked_path']) == Path(launch['configuration']['mrtrix_bin']) / name,
                'actual invoked official binary path differs')


def publish(path, digest, output_root):
    started = time.perf_counter()
    config, raw, launches, identities = configuration(path, digest)
    root = Path(output_root)
    protected = [Path(launch['configuration']['output_root']).resolve()
                 for launch in launches.values() if launch['state'] == 'configured']
    protected += [Path(launch['configuration'][key]).resolve() for launch in launches.values()
                  if launch['state'] == 'configured' for key in ('official_dwi_root', 'official_anatomy_root')]
    protected += [Path(path).resolve().parent for launch in launches.values()
                  if launch['state'] == 'configured' for path in launch['configuration']['source_files']
                  if Path(path).suffix == '.py']
    protected += [Path(launch['configuration_identity']['path']).parent for launch in launches.values()
                  if launch['state'] == 'configured']
    protected.append(Path(identities['configuration']['path']).parent)
    protected += [Path(filename).resolve().parent for filename in config['binding_source_files']]
    protected += [Path(item['path']).resolve().parent for case in raw['cases'] for item in case['input_files']]
    require(not any(root.resolve() == item or root.resolve().is_relative_to(item) or
                    item.is_relative_to(root.resolve()) for item in protected),
            'controlled view must remain outside actual source/producer/controller roots')
    require(not root.exists(), 'fresh explicit origin map view required')
    states, cases = {}, {}
    for name, launch in launches.items():
        if launch['state'] != 'configured':
            continue
        status = Path(launch['controller_status_path'])
        if not status.is_file():
            states[name] = None
            continue
        snapshot = status.read_bytes(); state = json.loads(snapshot)
        plan = launch['configuration']
        require(set(state['cases']) == set(plan['case_ids']) and state['workers'] == 1 and
                state['tracking_threads'] == 0 and state['downstream_threads'] == 8 and
                state['raw_manifest_sha256'] == config['raw_manifest']['sha256'] and
                state['verified_reference_manifest_sha256'] == config['verified_reference_manifest']['sha256'],
                'actual controller workload differs')
        for key in ('official_dwi_root', 'official_anatomy_root'):
            require(Path(state[key]).resolve() == Path(plan[key]).resolve(), 'actual controller producer root differs')
        for key, filename in RUNTIME_NAMES.items():
            require(state[key] == launch['runtime_sources'][filename], 'actual controller runtime source differs')
        for case, row in state['cases'].items():
            require(config['case_bindings'].get(case) == name or row['state'] not in ('running', 'completed'),
                    'another bound controller already executes this explicit case')
        states[name] = state
        launch['controller_snapshot'] = {'path': str(status), 'sha256': hashlib.sha256(snapshot).hexdigest(),
                                         'policy': 'exact status snapshot; other case progress may advance independently',
                                         'content': state}
    for case, name in config['case_bindings'].items():
        launch = launches[name]
        item = {'controller_binding': name, 'state': 'pending_configuration', 'controlled_link': None}
        if launch['state'] == 'configured':
            plan, state = launch['configuration'], states[name]
            row = state['cases'][case] if state else {'state': 'not_launched'}
            item.update(state=row['state'], origin_mode=launch['mode'],
                        actual_planned_case_root=str(Path(plan['output_root']) / case),
                        expected_dwi_contract=str(Path(plan['official_dwi_root']) / case / 'consumer_contract.json'),
                        expected_anatomy_contract=str(Path(plan['official_anatomy_root']) / case / 'consumer_contract.json'))
            if row['state'] == 'completed':
                actual = Path(plan['output_root']) / case
                require(not actual.is_symlink(), 'actual case root cannot be a substituted alias')
                verified = completed_case(case, row, Path(plan['output_root']), config, launch['runtime_sources'])
                for key, expected in (('official_dwi_contract', item['expected_dwi_contract']),
                                      ('anatomy_contract', item['expected_anatomy_contract'])):
                    require(Path(verified['producer_contracts'][key]['path']).resolve() == Path(expected).resolve(),
                            'actual case consumed another explicit producer path')
                manifest = json.loads(Path(verified['reference_manifest']['path']).read_text())
                verify_manifest_contract(manifest, raw, config, identities, launches[name])
                require(len(manifest['commands']) == len(manifest['completed_commands']) == 198 and
                        all(all(executed.get(key) == value for key, value in planned.items()) and
                            executed['returncode'] == 0 for planned, executed in zip(manifest['commands'], manifest['completed_commands'])),
                        'actual planned/executed command sequence differs')
                manifest['_actual_manifest_path'] = verified['reference_manifest']['path']
                item.update(verified, file_verification=referenced_files(manifest))
        cases[case] = item
    # Mutable controllers may advance other cases; selected completed rows and
    # duplicate-case guards must still match after the source byte checks.
    for name, state in states.items():
        if state is None:
            continue
        current = json.loads(Path(launches[name]['controller_status_path']).read_text())
        for case, row in current['cases'].items():
            require(config['case_bindings'].get(case) == name or row['state'] not in ('running', 'completed'),
                    'another bound controller started the explicitly reassigned case during audit')
            if config['case_bindings'].get(case) == name and state['cases'][case]['state'] == 'completed':
                require(row == state['cases'][case], 'selected completed controller row changed during audit')
    for identity in identities.values():
        checked_record(identity)
    for launch in launches.values():
        if launch['state'] == 'configured':
            checked_record(launch['configuration_identity'])
            for filename, value in launch['configuration']['source_files'].items():
                checked_record({'path': filename, 'sha256': value})
    for item in cases.values():
        if item['state'] == 'completed':
            checked_record(item['reference_manifest'])
    # All bytes verified before any new view or link is created.
    root.mkdir(parents=True)
    for case, item in cases.items():
        if item['state'] == 'completed':
            link, target = root / case, Path(item['actual_case_root'])
            link.symlink_to(target, target_is_directory=True)
            item['controlled_link'] = {'path': str(link), 'actual_target': str(target), 'policy': 'explicit actual origin symlink only; no copy'}
    # A failed historical controller can contain genuinely completed cases;
    # this new map reports the selected case origins and retains its full state.
    failed = any(item['state'] == 'failed' for item in cases.values())
    complete = all(item['state'] == 'completed' for item in cases.values())
    result = {'schema_version': 1, 'scope': 'explicit actual per-case official source hand-off only',
        'state': 'completed' if complete else 'failed' if failed else 'partial_or_waiting', 'execution_completed': complete,
        'scientific_parity': 'not_assessed', 'source_identity': identities, 'launch_bindings': launches,
        'cases': cases, 'utc': datetime.now(timezone.utc).isoformat(), 'readonly_binding_seconds': time.perf_counter() - started}
    atomic(root / 'case_origin_binding.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    result = publish(args.config, args.config_sha256, args.output_root)
    print(json.dumps({'state': result['state'], 'completed_cases': [case for case, row in result['cases'].items() if row['state'] == 'completed'],
                      'scientific_parity': result['scientific_parity']}))


if __name__ == '__main__':
    main()
