"""CPU metadata fixtures test correction gates; no real comparison or GPU execution."""
import copy
import json
from pathlib import Path
import unittest

from audit_pair_binding_correction import audit, digest
import test_evaluate_pair_input_binding as fixtures


class CorrectionAuditTests(unittest.TestCase):
    setUp = fixtures.PairInputBindingTests.setUp
    run_verify_phase = fixtures.PairInputBindingTests.run_verify_phase

    def prepare(self, role='startup_only_candidate'):
        if role == 'baseline':
            resources_path = Path(self.config['baseline_resources'])
            resources = json.loads(resources_path.read_text())
            self.resource_files = {}
            for category in ('weights', 'assets', 'binaries'):
                path = self.put(category + '/fixture.dat', 'CPU metadata resource ' + category)
                self.resource_files[category] = path
                resources[category] = {'fixture': {'resolved_path': str(path), 'sha256': digest(path)}}
            resources_path.write_text(json.dumps(resources))
        self.run_verify_phase(startup=role != 'baseline', verify_only=True, role=role)
        verified_root = Path(self.config['output'])
        verified_binding = json.loads((verified_root/'execution_binding.json').read_text())
        original_root = self.root/'original_comparison'
        original_config = dict(self.config, output=str(original_root))
        original_path = self.put('original_config.json', original_config)
        wrong_binding = copy.deepcopy(verified_binding)
        wrong_binding['input_sha256'] = digest(self.programs[-1])
        original_binding = self.put('original_comparison/execution_binding.json', wrong_binding)
        original_state = json.loads((verified_root/'checkpoint.json').read_text())
        original_state.update(status='complete', config_sha256=digest(original_path))
        original_checkpoint = self.put('original_comparison/checkpoint.json', original_state)
        receipt_paths = {'original_config':original_path, 'original_checkpoint':original_checkpoint,
            'original_binding':original_binding, 'verification_config':self.config_path,
            'verification_checkpoint':verified_root/'checkpoint.json',
            'verification_binding':verified_root/'execution_binding.json'}
        self.audit_config = {key:{'path':str(path),'sha256':digest(path)} for key,path in receipt_paths.items()}
        self.audit_config['output'] = str(self.root/'correction.json')
        self.audit_path = self.put('audit_config.json', self.audit_config)
        return receipt_paths

    def refresh(self, name):
        self.audit_config[name]['sha256'] = digest(self.audit_config[name]['path'])
        self.audit_path.write_text(json.dumps(self.audit_config))

    def test_only_input_metadata_corrected_and_originals_unchanged(self):
        paths = self.prepare(); original_hashes = {key:digest(path) for key,path in paths.items()}
        result = audit(self.audit_path)
        self.assertEqual(result['changed_fields'], ['input_sha256'])
        self.assertEqual(result['corrected_input_sha256'], self.input_sha)
        self.assertFalse(result['numerical_comparison_executed_by_verification'])
        self.assertEqual(original_hashes, {key:digest(path) for key,path in paths.items()})
        with self.assertRaises(FileExistsError): audit(self.audit_path)

    def test_any_other_binding_drift_rejected(self):
        paths = self.prepare(); row = json.loads(paths['verification_binding'].read_text())
        row['declared_gpu_uuid'] = 'another GPU'; paths['verification_binding'].write_text(json.dumps(row))
        self.refresh('verification_binding')
        with self.assertRaisesRegex(ValueError, 'beyond input_sha256'): audit(self.audit_path)
        self.assertFalse(Path(self.audit_config['output']).exists())

    def test_wrong_old_value_and_frozen_receipt_drift_rejected(self):
        paths = self.prepare(); row = json.loads(paths['original_binding'].read_text())
        row['input_sha256'] = digest(self.programs[0]); paths['original_binding'].write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'frozen receipt SHA'): audit(self.audit_path)
        self.refresh('original_binding')
        with self.assertRaisesRegex(ValueError, 'not the last official program'): audit(self.audit_path)

    def test_verification_cannot_claim_numerical_completion(self):
        paths = self.prepare(); row = json.loads(paths['verification_checkpoint'].read_text())
        row['status'] = 'complete'; paths['verification_checkpoint'].write_text(json.dumps(row))
        self.refresh('verification_checkpoint')
        with self.assertRaisesRegex(ValueError, 'did not stop'): audit(self.audit_path)

    def test_baseline_unique_correction_rechecks_real_resource_files(self):
        paths = self.prepare(role='baseline')
        hashes = {key: digest(path) for key, path in paths.items()}
        checkpoint = json.loads(paths['verification_checkpoint'].read_text())
        self.assertNotIn('role_helper_sha256', checkpoint)
        result = audit(self.audit_path)
        self.assertEqual(result['evaluated_role'], 'baseline')
        self.assertEqual(result['changed_fields'], ['input_sha256'])
        self.assertEqual(result['corrected_input_sha256'], self.input_sha)
        for path in self.resource_files.values():
            self.assertEqual(result['read_file_sha256'][str(path.resolve())], digest(path))
        self.assertEqual(hashes, {key: digest(path) for key, path in paths.items()})

    def test_baseline_resource_drift_rejected_before_correction(self):
        self.prepare(role='baseline')
        self.resource_files['weights'].write_text('drift after actual verify-only')
        with self.assertRaisesRegex(ValueError, 'baseline resource changed'):
            audit(self.audit_path)
        self.assertFalse(Path(self.audit_config['output']).exists())

    def test_baseline_frozen_manifest_drift_rejected(self):
        self.prepare(role='baseline')
        path = Path(self.config['baseline_resources'])
        resources = json.loads(path.read_text()); resources['code_commit'] = 'another commit'
        path.write_text(json.dumps(resources))
        with self.assertRaisesRegex(ValueError, 'frozen receipt SHA'):
            audit(self.audit_path)

    def test_baseline_missing_actual_launch_input_declaration_rejected(self):
        launch = self.root/'official_diag/launch.json'
        row = json.loads(launch.read_text()); del row['input_sha256']
        launch.write_text(json.dumps(row))
        self.prepare(role='baseline')
        with self.assertRaisesRegex(ValueError, 'raw input SHA declaration missing'):
            audit(self.audit_path)

    def test_candidate_resource_receipt_cannot_be_missing(self):
        paths = self.prepare(role='precision_candidate')
        for name in ('original_binding', 'verification_binding'):
            row = json.loads(paths[name].read_text())
            del row['precision_resource_verification']
            paths[name].write_text(json.dumps(row)); self.refresh(name)
        with self.assertRaisesRegex(ValueError, 'resource verification receipt missing'):
            audit(self.audit_path)

    def test_precision_role_receipt_and_helper_are_required(self):
        paths = self.prepare(role='precision_candidate')
        result = audit(self.audit_path)
        self.assertEqual(result['evaluated_role'], 'precision_candidate')
        Path(self.audit_config['output']).unlink()
        state = json.loads(paths['verification_checkpoint'].read_text())
        del state['role_helper_sha256']; paths['verification_checkpoint'].write_text(json.dumps(state))
        self.refresh('verification_checkpoint')
        with self.assertRaisesRegex(ValueError, 'helper SHA'):
            audit(self.audit_path)


if __name__ == '__main__': unittest.main()
