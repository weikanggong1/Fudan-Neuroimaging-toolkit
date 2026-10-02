"""两组来源绑定的 CPU 文件协议回归；不是 MRI 精度或性能 benchmark。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('reference_case_origin_fixture',
    Path(__file__).parents[2] / 'tools/reference/bind_connectome_raw_reference_origins.py')
binder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binder)


class CaseOrigins(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        names = ('run_connectome_raw_official_cohort.py', 'benchmark_connectome_raw_official.py',
                 'benchmark_connectome_repeats_official.py', 'connectome_repeat_common.py')
        self.sources = {}
        for name in names:
            path = self.root / name
            path.write_text('# CPU protocol fixture only\n')
            self.sources[str(path)] = binder.sha256(path)
        self.source_hashes = {Path(path).name: value for path, value in self.sources.items()}
        raw = self.write('raw.json', {'cases': [{'case_id': 'sub-a'}, {'case_id': 'sub-b'}]})
        reference = self.write('verified.json', {'fixture_only': True})
        self.config = {'raw_manifest': self.record(raw), 'verified_reference_manifest': self.record(reference),
                       'source_files': self.sources, 'seeds': [0, 1, 2, 3, 4], 'n_seeds': 100000,
                       'expected_eddy_solver': 'cpu', 'groups': {}}
        for group, case in [('A', 'sub-a'), ('B', 'sub-b')]:
            output = self.root / ('group-' + group)
            launch = {'case_ids': [case], 'output_root': str(output), 'seeds': self.config['seeds'],
                      'n_seeds': self.config['n_seeds'], 'raw_manifest_sha256': binder.sha256(raw),
                      'verified_reference_manifest_sha256': binder.sha256(reference),
                      'official_dwi_root': str(self.root / 'dwi'),
                      'official_anatomy_root': str(self.root / 'anatomy')}
            launch_path = self.write('launch-' + group + '.json', launch)
            self.config['groups'][group] = {'case_ids': [case], 'output_root': str(output),
                                           'launch_configuration': self.record(launch_path)}
        self.config_path = self.write('configuration.json', self.config)
        self.view = self.root / 'controlled-view'

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def record(self, path):
        return {'path': str(path), 'sha256': binder.sha256(path)}

    def update_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def publish(self):
        return binder.publish(self.config_path, binder.sha256(self.config_path), self.view)

    def completed(self, group='A', case='sub-a'):
        group_root = Path(self.config['groups'][group]['output_root'])
        group_root.mkdir()
        anatomy = self.write('anatomy/' + case + '/consumer_contract.json', {'fixture_only': True})
        dwi = self.write('dwi/' + case + '/consumer_contract.json', {'fixture_only': True})
        result = {'execution_completed': True, 'state': 'completed', 'case_id': case,
                  'seeds': self.config['seeds'], 'parameters': {'n_seed_attempts': 100000, 'tracking_threads': 0},
                  'downstream_threads': 8, 'script_sha256': self.source_hashes['benchmark_connectome_raw_official.py'],
                  'reference_command_helper_sha256': self.source_hashes['benchmark_connectome_repeats_official.py'],
                  'matrix_helper_sha256': self.source_hashes['connectome_repeat_common.py'],
                  'raw_case_binding': {'manifest_sha256': self.config['raw_manifest']['sha256'], 'case_id': case,
                                       'official_eddy_solver': {'mode': 'cpu'}, 'raw_b0_selection': {'fixture_only': True}},
                  'commands': [{}], 'completed_commands': [{'returncode': 0}],
                  'source': {'anatomy_contract': self.record(anatomy), 'official_dwi_contract': self.record(dwi)}}
        reference = self.write('group-' + group + '/' + case + '/reference_manifest.json', result)
        case_row = {'state': 'completed', 'returncode': 0, 'reference_manifest_sha256': binder.sha256(reference),
                    'anatomy_contract_sha256': binder.sha256(anatomy), 'dwi_contract_sha256': binder.sha256(dwi)}
        launch = json.loads(Path(self.config['groups'][group]['launch_configuration']['path']).read_text())
        status = {'cases': {case: case_row}, 'workers': 1, 'tracking_threads': 0, 'downstream_threads': 8,
                  'raw_manifest_sha256': self.config['raw_manifest']['sha256'],
                  'verified_reference_manifest_sha256': self.config['verified_reference_manifest']['sha256'],
                  'controller_sha256': self.source_hashes['run_connectome_raw_official_cohort.py'],
                  'worker_sha256': self.source_hashes['benchmark_connectome_raw_official.py'],
                  'helper_sha256': self.source_hashes['benchmark_connectome_repeats_official.py'],
                  'matrix_helper_sha256': self.source_hashes['connectome_repeat_common.py'],
                  'official_dwi_root': launch['official_dwi_root'], 'official_anatomy_root': launch['official_anatomy_root'],
                  'state': 'completed', 'controller_pid': 123}
        status_path = self.write('group-' + group + '/cohort_status.json', status)
        return reference, status_path

    def test_waiting_contracts_create_no_case_links(self):
        result = self.publish()
        self.assertFalse(result['execution_completed'])
        self.assertEqual(result['cases']['sub-a']['state'], 'not_launched')
        self.assertFalse((self.view / 'sub-a').exists())

    def test_completed_case_links_actual_group_output_without_copy(self):
        reference, _ = self.completed()
        result = self.publish()
        self.assertFalse(result['execution_completed'])
        self.assertTrue((self.view / 'sub-a').is_symlink())
        self.assertEqual((self.view / 'sub-a').resolve(), reference.parent)
        self.assertEqual(result['cases']['sub-a']['reference_manifest']['sha256'], binder.sha256(reference))
        self.assertEqual(self.publish()['cases']['sub-a']['actual_case_root'], str(reference.parent))

    def test_groups_cannot_duplicate_a_case_or_diverge_inputs(self):
        self.config['groups']['B']['case_ids'] = ['sub-a']
        self.update_config()
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            self.publish()
        self.config['groups']['B']['case_ids'] = ['sub-b']
        launch = Path(self.config['groups']['B']['launch_configuration']['path'])
        values = json.loads(launch.read_text()); values['official_dwi_root'] = str(self.root / 'different')
        launch.write_text(json.dumps(values))
        self.config['groups']['B']['launch_configuration'] = self.record(launch)
        self.update_config()
        with self.assertRaisesRegex(ValueError, 'identical frozen'):
            self.publish()

    def test_failed_worker_is_not_published_as_complete(self):
        _, status = self.completed()
        row = json.loads(status.read_text()); row['cases']['sub-a']['returncode'] = 9
        status.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'nonzero worker'):
            self.publish()
        self.assertFalse((self.view / 'sub-a').exists())

    def test_actual_contract_or_runtime_hash_change_is_rejected(self):
        reference, _ = self.completed()
        report = json.loads(reference.read_text())
        Path(report['source']['official_dwi_contract']['path']).write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'actual source/config/contract changed'):
            self.publish()

    def test_existing_fake_directory_or_different_link_is_rejected(self):
        self.completed()
        self.view.mkdir()
        (self.view / 'sub-a').mkdir()
        with self.assertRaisesRegex(ValueError, 'preexisting real directory'):
            self.publish()
        (self.view / 'sub-a').rmdir()
        (self.view / 'sub-a').symlink_to(self.root / 'wrong')
        with self.assertRaisesRegex(ValueError, 'another actual output'):
            self.publish()

    def test_failed_controller_is_reported_without_false_completion(self):
        _, status = self.completed()
        row = json.loads(status.read_text()); row['state'] = 'failed'
        row['cases']['sub-a'] = {'state': 'waiting'}
        status.write_text(json.dumps(row))
        result = self.publish()
        self.assertEqual(result['state'], 'failed')
        self.assertFalse(result['execution_completed'])
        self.assertFalse((self.view / 'sub-a').exists())

    def test_failed_controller_after_all_cases_does_not_claim_success(self):
        _, status = self.completed(); self.completed('B', 'sub-b')
        row = json.loads(status.read_text()); row['state'] = 'failed'
        status.write_text(json.dumps(row))
        result = self.publish()
        self.assertEqual(result['state'], 'failed')
        self.assertFalse(result['execution_completed'])

    def test_all_completed_groups_produce_complete_origin_binding(self):
        self.completed(); self.completed('B', 'sub-b')
        result = self.publish()
        self.assertTrue(result['execution_completed'])
        self.assertEqual(result['scientific_parity'], 'not_assessed')


if __name__ == '__main__':
    unittest.main()
