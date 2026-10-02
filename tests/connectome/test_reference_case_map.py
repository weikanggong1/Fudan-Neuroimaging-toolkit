"""逐病例 source hand-off 的 CPU 文件协议回归；不是 MRI benchmark。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOL = Path(__file__).parents[2] / 'tools/reference/bind_connectome_raw_reference_case_map.py'
sys.path.insert(0, str(TOOL.parent))
SPEC = importlib.util.spec_from_file_location('explicit_reference_case_map', TOOL)
binder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binder)


class ExplicitOrigins(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cases = [f'sub-C{index:02d}' for index in range(10)]
        self.sources = {}
        for name in binder.RUNTIME_NAMES.values():
            path = self.bytes('tools/' + name, '# protocol fixture; no scientific execution\n')
            self.sources[str(path)] = binder.sha256(path)
        self.hashes = {Path(path).name: value for path, value in self.sources.items()}
        raw = {'dataset': 'fixture', 'snapshot': 'fixture', 'license': 'fixture', 'cases': []}
        for case in self.cases:
            image = self.bytes('raw/' + case + '/T1', 'raw fixture ' + case)
            raw['cases'].append({'case_id': case, 'subject': case, 'session': 'ses-test',
                                 'input_files': [{**self.record(image), 'kind': 'raw_t1w'}]})
        self.raw = raw
        raw_path = self.json('raw_manifest.json', raw)
        self.programs = {}
        for name in ('mrinfo', 'tckgen', 'tckinfo', 'tcksift2', 'tckstats', 'tcksample', 'tck2connectome'):
            path = self.bytes('bin/' + name, 'official program protocol fixture')
            self.programs[name] = {**self.record(path), 'invoked_path': str(path)}
        reference = {'execution_completed': True, 'programs': self.programs, 'mrtrix_version': 'fixture version',
                     'script_sha256': self.hashes['benchmark_connectome_repeats_official.py'],
                     'helper_sha256': self.hashes['connectome_repeat_common.py']}
        reference_path = self.json('reference.json', reference)
        self.config = {'raw_manifest': self.record(raw_path), 'verified_reference_manifest': self.record(reference_path),
                       'seeds': [0, 1, 2, 3, 4], 'n_seeds': 100000, 'expected_eddy_solver': 'cpu',
                       'expected_parameters': {'n_seed_attempts': 100000, 'tracking_threads': 0},
                       'expected_runtime_sources': self.hashes,
                       'binding_source_files': {str(TOOL): binder.sha256(TOOL),
                           str(Path(binder.require.__code__.co_filename)): binder.sha256(binder.require.__code__.co_filename)},
                       'case_bindings': {case: 'A' if index % 2 == 0 else 'B' for index, case in enumerate(self.cases)},
                       'launch_bindings': {}}
        self.config['case_bindings'][self.cases[-1]] = 'tail'
        self.config['launch_bindings']['tail'] = {'state': 'pending_configuration'}
        for group, cases in [('A', self.cases[::2]), ('B', self.cases[1::2])]:
            self.launch(group, cases)
        self.config_path = self.json('new-freeze/config.json', self.config)
        self.view = self.root / 'new-view'

    def bytes(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

    def json(self, name, value):
        return self.bytes(name, json.dumps(value))

    def record(self, path):
        return {'path': str(path), 'sha256': binder.sha256(path)}

    def launch(self, group, cases, roots='old'):
        plan = {'raw_manifest': self.config['raw_manifest']['path'],
                'raw_manifest_sha256': self.config['raw_manifest']['sha256'],
                'verified_reference_manifest': self.config['verified_reference_manifest']['path'],
                'verified_reference_manifest_sha256': self.config['verified_reference_manifest']['sha256'],
                'n_seeds': 100000, 'seeds': [0, 1, 2, 3, 4], 'source_files': self.sources,
                'case_ids': cases, 'output_root': str(self.root / ('actual-' + group)),
                'official_dwi_root': str(self.root / (roots + '-dwi')),
                'official_anatomy_root': str(self.root / (roots + '-anatomy')),
                'mrtrix_bin': str(self.root / 'bin')}
        launch = self.json('launches/' + group + '.json', plan)
        self.config['launch_bindings'][group] = {'launch_configuration': self.record(launch),
            'controller_status_path': str(Path(plan['output_root']) / 'cohort_status.json'),
            'mode': 'reuse actual reference' if roots == 'old' else 'new explicit producer namespace'}
        return plan, launch

    def config_update(self):
        self.config_path.write_text(json.dumps(self.config))

    def publish(self):
        self.config_update()
        return binder.publish(self.config_path, binder.sha256(self.config_path), self.view)

    def status(self, group, changed=None):
        plan = json.loads(Path(self.config['launch_bindings'][group]['launch_configuration']['path']).read_text())
        state = {'cases': {case: {'state': 'waiting'} for case in plan['case_ids']}, 'workers': 1,
            'tracking_threads': 0, 'downstream_threads': 8, 'raw_manifest_sha256': self.config['raw_manifest']['sha256'],
            'verified_reference_manifest_sha256': self.config['verified_reference_manifest']['sha256'],
            'official_dwi_root': plan['official_dwi_root'], 'official_anatomy_root': plan['official_anatomy_root'],
            'state': 'waiting', **{key: self.hashes[value] for key, value in binder.RUNTIME_NAMES.items()}}
        if changed:
            state['cases'].update(changed)
        return self.json('actual-' + group + '/cohort_status.json', state)

    def completed(self, group='A', case=None, aligned=False):
        case = case or self.cases[0]
        plan = json.loads(Path(self.config['launch_bindings'][group]['launch_configuration']['path']).read_text())
        canonical = next(item for item in self.raw['cases'] if item['case_id'] == case)
        rawprep = {'completed': True, 'subject': case, 'session': 'ses-test', 'commands': [{'returncode': 0}],
                   'input_sha256': {item['path']: item['sha256'] for item in canonical['input_files']}}
        rawprep_path = self.json('rawprep/' + case + '.json', rawprep)
        dwi = self.json(Path(plan['official_dwi_root']).relative_to(self.root).as_posix() + '/' + case + '/consumer_contract.json',
                        {'state': 'completed', 'case_id': case, 'scope': binder.CONTRACT_SCOPES['official_dwi_contract']})
        aux = self.json('aux/' + case + '.json', {'protocol_fixture': True})
        anatomy = self.json(Path(plan['official_anatomy_root']).relative_to(self.root).as_posix() + '/' + case + '/consumer_contract.json',
            {'state': 'completed', 'case_id': case, 'scope': binder.CONTRACT_SCOPES['anatomy_contract'],
             'official_dwi_contract': self.record(dwi), 'prepared_report': self.record(aux), 'official_anatomy_report': self.record(aux)})
        image = self.bytes('images/' + case, 'consumed image fixture')
        nodes = self.bytes('nodes/' + case, 'nodes fixture')
        source = {'official_dwi_contract': self.record(dwi), 'anatomy_contract': self.record(anatomy),
            'raw_t1w': {key: value for key, value in canonical['input_files'][0].items() if key != 'kind'},
            'images': {name: self.record(image) for name in ('wm_fod', 'fa', 'five_tissue_act', 'five_tissue_sift2', 'gmwmi')},
            'profiles': {name: {'image': self.record(image), 'nodes_path': str(nodes), 'nodes_sha256': binder.sha256(nodes), 'nodes': 1,
                              'node_rows': [['1', '7', 'L', 'fixture']]}
                         for name in binder.ATLASES}}
        base = Path(plan['output_root']) / case
        commands = []
        for index in range(198):
            log = self.bytes((base.relative_to(self.root) / f'log-{index}').as_posix(), 'log fixture')
            commands.append({'stage': 'protocol_fixture', 'ordinal': index, 'log': str(log)})
        outputs = {}
        for seed in self.config['seeds']:
            directory = base / f'seed-{seed}'
            track = self.bytes((directory.relative_to(self.root) / 'tracks.tck').as_posix(), 'TCK fixture')
            vectors = {name: self.record(self.bytes((directory.relative_to(self.root) / name).as_posix(), '1'))
                       for name in ('sift2_weights.txt', 'lengths.txt', 'mean_fa.txt')}
            profiles = {}
            for name in binder.ATLASES:
                target = directory / 'atlases' / name
                hashes = {}
                raw_hashes = {}
                for matrix in binder.MATRIX_NAMES:
                    raw_file = self.bytes((target.relative_to(self.root) / (matrix + '.csv')).as_posix(), '1')
                    raw_hashes[matrix] = binder.sha256(raw_file)
                    canonical_file = self.bytes((target.relative_to(self.root) / ('connectome_' + matrix + '.csv')).as_posix(), '1,0\n0,0') if aligned else raw_file
                    hashes[matrix] = binder.sha256(canonical_file)
                node_file = self.bytes((target.relative_to(self.root) / 'nodes.tsv').as_posix(), 'node fixture')
                alignment = {'canonical_matrix_sha256': hashes, 'raw_matrix_sha256': raw_hashes} if aligned else None
                if aligned:
                    self.json((target.relative_to(self.root) / 'canonical_matrix_alignment.json').as_posix(), alignment)
                profiles[name] = {'directory': str(target), 'matrix_sha256': hashes,
                                  'nodes_tsv_sha256': binder.sha256(node_file), 'canonical_matrix_alignment': alignment,
                                  'nodes': 1, 'node_rows': [['1', '7', 'L', 'fixture']], 'atlas_sha256': binder.sha256(image)}
            outputs[str(seed)] = {'tracks_sha256': binder.sha256(track), 'scalars': vectors, 'profiles': profiles}
        report = {'scope': 'independent official raw DWI plus fresh FS anatomy tracking/downstream',
            'execution_completed': True, 'state': 'completed', 'case_id': case, 'seeds': self.config['seeds'],
            'parameters': self.config['expected_parameters'], 'downstream_threads': 8, 'source': source,
            'script_sha256': self.hashes['benchmark_connectome_raw_official.py'],
            'reference_command_helper_sha256': self.hashes['benchmark_connectome_repeats_official.py'],
            'matrix_helper_sha256': self.hashes['connectome_repeat_common.py'],
            'verified_reference_identity': self.config['verified_reference_manifest'], 'programs': self.programs,
            'mrtrix_version': 'fixture version', 'commands': commands,
            'completed_commands': [{**row, 'returncode': 0, 'time_file': row['log']} for row in commands], 'outputs': outputs,
            'raw_case_binding': {'manifest_path': self.config['raw_manifest']['path'], 'manifest_sha256': self.config['raw_manifest']['sha256'],
                'case_id': case, 'dataset': self.raw['dataset'], 'snapshot': self.raw['snapshot'], 'license': self.raw['license'],
                'files': canonical['input_files'], 'official_eddy_solver': {'mode': 'cpu'}, 'raw_b0_selection': {},
                'official_rawprep_report': str(rawprep_path), 'official_rawprep_report_sha256': binder.sha256(rawprep_path),
                'rawprep_producer_identity': rawprep}}
        manifest = self.json((base.relative_to(self.root) / 'reference_manifest.json').as_posix(), report)
        row = {'state': 'completed', 'returncode': 0, 'reference_manifest_sha256': binder.sha256(manifest),
               'anatomy_contract_sha256': binder.sha256(anatomy), 'dwi_contract_sha256': binder.sha256(dwi)}
        status = self.status(group, {case: row})
        return manifest, status

    def report_update(self, manifest, status, edit):
        report = json.loads(manifest.read_text())
        edit(report)
        manifest.write_text(json.dumps(report))
        snapshot = json.loads(status.read_text())
        snapshot['cases'][report['case_id']]['reference_manifest_sha256'] = binder.sha256(manifest)
        status.write_text(json.dumps(snapshot))

    def test_pending_tail_has_no_invented_path_or_link(self):
        result = self.publish()
        row = result['cases'][self.cases[-1]]
        self.assertEqual(row, {'controller_binding': 'tail', 'state': 'pending_configuration', 'controlled_link': None})
        self.assertEqual(result['scientific_parity'], 'not_assessed')
        self.assertFalse((self.view / self.cases[-1]).exists())

    def test_new_actual_producer_namespace_can_differ_without_runtime_change(self):
        self.launch('tail', [self.cases[-1]], roots='fresh')
        manifest, _ = self.completed('tail', self.cases[-1])
        result = self.publish()
        row = result['cases'][self.cases[-1]]
        self.assertEqual(Path(row['actual_case_root']), manifest.parent)
        self.assertIn('fresh-dwi', row['producer_contracts']['official_dwi_contract']['path'])
        self.assertTrue((self.view / self.cases[-1]).is_symlink())

    def test_canonical_role_and_case_raw_input_changes_rejected(self):
        manifest, status = self.completed()
        self.report_update(manifest, status, lambda report: report['raw_case_binding']['files'][0].update(kind='raw_ap_dwi'))
        with self.assertRaisesRegex(ValueError, 'canonical'):
            self.publish()
        self.assertFalse(self.view.exists())

    def test_aligned_and_original_matrix_hashes_both_verified(self):
        manifest, _ = self.completed(aligned=True)
        result = self.publish()
        row = result['cases'][self.cases[0]]
        verified = row['file_verification']['verified_file_sha256']
        self.assertTrue(any(Path(name).name == 'connectome_count.csv' for name in verified))
        self.assertTrue(any(Path(name).name == 'count.csv' for name in verified))
        self.assertEqual(Path(row['actual_case_root']), manifest.parent)

    def test_auxiliary_producer_report_change_rejected(self):
        self.completed()
        (self.root / 'aux' / (self.cases[0] + '.json')).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.publish()
        self.assertFalse(self.view.exists())

    def test_original_matrix_change_rejected_even_when_canonical_unmodified(self):
        manifest, _ = self.completed(aligned=True)
        (manifest.parent / 'seed-0/atlases/fs-aparc/count.csv').write_text('2')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.publish()

    def test_pending_configuration_cannot_fabricate_paths(self):
        self.config['launch_bindings']['tail']['output_root'] = '/fake'
        with self.assertRaisesRegex(ValueError, 'cannot invent'):
            self.publish()

    def test_runtime_change_between_launches_rejected(self):
        _, launch = self.launch('tail', [self.cases[-1]], roots='fresh')
        data = json.loads(launch.read_text())
        name = 'benchmark_connectome_raw_official.py'
        alternate = self.bytes('different-tools/' + name, '# changed protocol fixture')
        data['source_files'] = {path: value for path, value in self.sources.items() if Path(path).name != name}
        data['source_files'][str(alternate)] = binder.sha256(alternate)
        launch.write_text(json.dumps(data))
        self.config['launch_bindings']['tail']['launch_configuration'] = self.record(launch)
        with self.assertRaisesRegex(ValueError, 'same frozen'):
            self.publish()

    def test_old_controller_may_not_execute_reassigned_tail(self):
        self.status('B', {self.cases[-1]: {'state': 'running'}})
        with self.assertRaisesRegex(ValueError, 'another bound controller'):
            self.publish()

    def test_complete_row_changed_during_byte_check_rejected(self):
        _, status = self.completed()
        actual_check = binder.referenced_files
        def mutate(report):
            result = actual_check(report)
            current = json.loads(status.read_text())
            current['cases'][self.cases[0]]['returncode'] = 4
            status.write_text(json.dumps(current))
            return result
        with patch.object(binder, 'referenced_files', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'completed controller row changed'):
                self.publish()
        self.assertFalse(self.view.exists())

    def test_pinned_program_version_and_scientific_parameters_unchanged(self):
        for field, replacement in [('mrtrix_version', 'changed version'),
                                   ('parameters', {'n_seed_attempts': 100000, 'tracking_threads': 1})]:
            with self.subTest(field=field):
                manifest, status = self.completed()
                self.report_update(manifest, status, lambda report: report.update({field: replacement}))
                with self.assertRaisesRegex(ValueError, 'configuration|profile|version'):
                    self.publish()
                self.assertFalse(self.view.exists())

    def test_case_map_must_cover_exact_canonical_ten(self):
        del self.config['case_bindings'][self.cases[0]]
        with self.assertRaisesRegex(ValueError, 'exact ten'):
            self.publish()

    def test_view_cannot_be_parent_of_actual_controller_or_in_sources(self):
        for target in (self.root, self.root / 'tools/view', self.root / 'old-dwi/view'):
            self.view = target
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'outside'):
                self.publish()

    def test_failed_historical_controller_remains_visible_for_completed_origin(self):
        _, status = self.completed()
        snapshot = json.loads(status.read_text()); snapshot['state'] = 'failed'; snapshot['error'] = 'preserved fixture failure'
        status.write_text(json.dumps(snapshot))
        result = self.publish()
        self.assertEqual(result['cases'][self.cases[0]]['state'], 'completed')
        self.assertEqual(result['launch_bindings']['A']['controller_snapshot']['content']['error'], 'preserved fixture failure')
        self.assertEqual(result['state'], 'partial_or_waiting')


if __name__ == '__main__':
    unittest.main()
