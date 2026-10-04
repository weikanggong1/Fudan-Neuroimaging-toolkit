"""Exercise actual verify phase with distinct raw-input/program hashes; CPU metadata only."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate_pair


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StopBeforeNumericalComparison(RuntimeError):
    pass


class PairInputBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        def put(name, value):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) if isinstance(value, dict) else value)
            return path
        self.put = put
        raw = put('raw.nii', 'distinct original input')
        self.input_sha = digest(raw)
        source = put('source/src/fnit/recon_all/native_free.py', '# frozen algorithm')
        archive = put('source.tar', 'frozen source archive')
        self.programs = [put('program_a', 'official program A'), put('program_b', 'official program B')]
        programs = put('program_manifest.json', {'programs': {str(p): digest(p) for p in self.programs}, 'resources': {}})
        cohort = put('cohort.json', {'cases': [{'id': 'case', 'sha256': self.input_sha}]})
        resources = put('resources.json', {'code_commit': '816', 'mismatches': [], 'weights': {}, 'assets': {}, 'binaries': {}})
        driver = put('driver.py', '# test driver')
        scripts = self.root / 'scripts'; scripts.mkdir()
        common = {'input': str(raw), 'input_sha256': self.input_sha, 'threads': 4, 'gpu_uuid': 'GPU-test'}
        baseline = dict(common, output=str(self.root / 'baseline'), diagnostic_root=str(self.root / 'baseline_diag'),
                        code_root=str(self.root / 'source'), code_commit='816', source_archive_sha256=digest(archive))
        official = dict(common, output=str(self.root / 'official'), diagnostic_root=str(self.root / 'official_diag'),
                        program_manifest=str(programs))
        baseline_path = put('baseline.json', baseline); official_path = put('official.json', official)
        log = put('official/scripts/recon-all.log', 'actual log'); done = put('official/scripts/recon-all.done', 'actual done')
        put('baseline_diag/completion.json', {'execution_status': 'complete', 'exit_code': 0, 'code_commit': '816',
            'source_archive_sha256': digest(archive), 'output_validation': {'expected': 138, 'present': 138}})
        put('official_diag/completion.json', {'execution_status': 'complete', 'exit_code': 0,
            'code_version': 'FreeSurfer 8.2.0 d932c45', 'program_manifest_sha256': digest(programs),
            'log_sha256': digest(log), 'done_sha256': digest(done), 'input_sha256': self.input_sha})
        for name, config_path in (('baseline', baseline_path), ('official', official_path)):
            put(name + '_diag/launch.json', {'host': socket.gethostname(), 'config_sha256': digest(config_path),
                'candidate_native_free_sha256': digest(source), 'input_sha256': self.input_sha})
        self.config = {'case': 'case', 'baseline_config': str(baseline_path), 'official_config': str(official_path),
            'baseline_commit': '816', 'baseline_resources': str(resources), 'cohort_manifest': str(cohort),
            'source_archive': str(archive), 'gpu_uuid': 'GPU-test', 'code_root': str(self.root / 'source'),
            'whole_driver': str(driver), 'scripts_dir': str(scripts), 'output': str(self.root / 'comparison'),
            'lock': str(self.root / 'lock')}
        self.config_path = put('comparison_config.json', self.config)

    def run_verify_phase(self, startup=False, verify_only=False, role=None):
        if startup:
            self.config.update(evaluated_role=role or 'startup_only_candidate', evaluated_config=self.config['baseline_config'],
                               evaluated_commit='816', evaluated_resources=self.config['baseline_resources'],
                               evaluated_resources_kind='admission_inventory')
            self.config_path.write_text(json.dumps(self.config))
        def write(path, value):
            Path(path).write_text(json.dumps(value))
        def stop(*args):
            raise StopBeforeNumericalComparison('metadata verification finished; no numerical comparison')
        driver = SimpleNamespace(digest=digest, write=write, configure_runtime=lambda: None,
            common_lock=lambda path: contextlib.nullcontext(), tool=lambda name, scripts: SimpleNamespace(compare=stop))
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        argv = ['evaluate_pair', '--config', str(self.config_path)] + (['--verify-only'] if verify_only else [])
        with patch.dict(os.environ), patch.object(sys, 'argv', argv), \
             patch.object(sys, 'path', list(sys.path)), \
             patch.object(evaluate_pair.importlib.util, 'spec_from_file_location', return_value=spec), \
             patch.object(evaluate_pair.importlib.util, 'module_from_spec', return_value=driver), \
             patch.object(evaluate_pair, 'verify_startup_binding', return_value={'verified_by_existing_helper': True}), \
             patch.object(evaluate_pair, 'verify_admitted_candidate_binding', return_value={'verified_by_existing_helper': True}):
            evaluate_pair.main()

    def assert_input_receipt(self):
        binding = json.loads((self.root / 'comparison/execution_binding.json').read_text())
        self.assertEqual(binding['input_sha256'], self.input_sha)
        self.assertTrue(all(binding['input_sha256'] != digest(p) for p in self.programs))
        checkpoint = json.loads((self.root / 'comparison/checkpoint.json').read_text())
        self.assertEqual(checkpoint['phases']['verify_binding']['status'], 'complete')
        return binding

    def test_baseline_input_sha_survives_multiple_program_checks(self):
        with self.assertRaises(StopBeforeNumericalComparison): self.run_verify_phase()
        self.assert_input_receipt()

    def test_startup_input_sha_survives_same_verification_loop(self):
        with self.assertRaises(StopBeforeNumericalComparison): self.run_verify_phase(startup=True)
        binding = self.assert_input_receipt()
        self.assertIn('startup_only_candidate', binding)
        self.assertIn('startup_resource_verification', binding)

    def test_verify_only_stops_before_all_numerical_phases(self):
        self.run_verify_phase(startup=True, verify_only=True)
        self.assert_input_receipt()
        checkpoint = json.loads((self.root / 'comparison/checkpoint.json').read_text())
        self.assertEqual(checkpoint['status'], 'binding_verified_only')
        self.assertEqual(set(checkpoint['phases']), {'verify_binding'})
        self.assertFalse(checkpoint['numerical_comparison_executed'])
        self.assertNotIn('strict_138', checkpoint)
        self.assertEqual(checkpoint['overall_metric_equivalence'], 'not_assessed')
        self.assertEqual(set(p.name for p in (self.root / 'comparison').iterdir()),
                         {'checkpoint.json', 'execution_binding.json'})
        with self.assertRaises(FileExistsError):
            self.run_verify_phase(startup=True, verify_only=True)

    def test_program_drift_still_rejected_before_binding_publication(self):
        self.programs[-1].write_text('changed official program')
        with self.assertRaisesRegex(ValueError, 'official program changed'): self.run_verify_phase()
        self.assertFalse((self.root / 'comparison/execution_binding.json').exists())


if __name__ == '__main__':
    unittest.main()
