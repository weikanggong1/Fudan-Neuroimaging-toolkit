"""安全出口的来源/产物/类型拒绝门；小文件用于合同，不是 MRI benchmark。"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from export_recon_topology_genetic_distance_wiring import export_completed, sha, NPZ_FILES, CONTROL_FILES, ROUTE_MODULES


class DistanceExportContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.children, self.diagnostic, self.out = (self.root / n for n in ('source', 'children', 'diagnostic', 'publication'))
        for p in (self.source, self.children, self.diagnostic):
            p.mkdir()
        public = json.loads((ROOT / 'validation/recon_all/optimizations/20261011_topology_genetic_distance_wiring/metrics.json').read_text())
        self.report = copy.deepcopy(public)
        self.report['original_parent_failure'] = 'checker_import_setup'
        self.report['original_parent_exit_code'] = 1
        self.report['algorithm_rerun'] = False
        self.report['successful_original_pair_cli_seconds'] = None
        self.write(self.children / 'EXIT_CODE', '1', raw=True)
        self.report['original_parent_exit_sha256'] = sha(self.children / 'EXIT_CODE')
        payload, modules = {}, {}
        for name in ROUTE_MODULES:
            rel = 'src/' + name.replace('.', '/') + '.py'
            p = self.source / rel
            self.write(p, '# fixed test module\n', raw=True)
            payload[rel] = dict(sha256=sha(p), size=p.stat().st_size)
            modules[name] = sha(p)
        rel = 'tools/benchmark_recon_topology_genetic_optimization.py'
        self.write(self.source / rel, '# fixed test comparator\n', raw=True)
        payload[rel] = dict(sha256=sha(self.source / rel), size=(self.source / rel).stat().st_size)
        self.write(self.source / 'SOURCE_MANIFEST.private.json', dict(payload=payload, expected_module_sha256=modules))
        self.manifest_sha = sha(self.source / 'SOURCE_MANIFEST.private.json')
        self.report['source_manifest_sha256'] = self.manifest_sha
        self.report['comparator_sha256'] = payload[rel]['sha256']
        base = self.report['records'][0]['report']
        self.protocol = dict(base['policy'], status='frozen_explicit_distance_short_pair',
            source_manifest_sha256=self.manifest_sha, input_sha256=base['input_sha256'],
            cpu_affinity=[84, 85], distance_backends=['serial', 'parallel'],
            fresh_processes=True, global_function_replacement=False)
        self.write(self.source / 'protocol.private.json', self.protocol)
        self.protocol_sha = sha(self.source / 'protocol.private.json')
        self.report['comparisons'] = {n: copy.deepcopy(public['comparisons'][n.replace('.private', '')]) for n in NPZ_FILES}
        self.report['controls'] = {n: True for n in CONTROL_FILES}
        for row in self.report['records']:
            label, v = row['label'], row['report']
            p = self.children / 'pair' / label
            p.mkdir(parents=True)
            v['status'] = 'complete_six_score_production_API_stage'
            v['source_manifest_sha256'] = self.manifest_sha
            v['global_function_replacement'] = False
            v['actual_loaded_module_sha256'] = modules.copy()
            v.pop('output_sha256', None)
            for name in NPZ_FILES:
                with zipfile.ZipFile(p / name, 'w') as archive:
                    for field in self.report['comparisons'][name]['fields']:
                        archive.writestr(field + '.npy', b'fixture array payload')
            self.write(p / CONTROL_FILES[0], dict(rounds=40, accepted=True))
            self.write(p / CONTROL_FILES[1], dict(random_draws=v['random_draws'], observed_random_draws=v['random_draws'],
                random_sequence_sha256=v['random_sequence_sha256'], strict=True))
            v['checkpoint_sha256'] = {n: sha(p / n) for n in NPZ_FILES + CONTROL_FILES}
            row['process'] = dict(exit_code=0, fresh_cli_seconds=row['fresh_cli_seconds'])
        self.bind()

    @staticmethod
    def write(path, value, raw=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if raw else json.dumps(value, indent=2) + '\n')

    def bind(self):
        """重新封装外层SHA，使每项负向确实到达目标语义门。"""
        for row in self.report['records']:
            p = self.children / 'pair' / row['label']
            self.write(p / 'report.private.json', row['report'])
            self.write(p / 'COMPLETE.json', dict(status=row['report']['status'], report_sha256=sha(p / 'report.private.json')))
            process = self.children / 'pair' / (row['label'] + '.EXIT.json')
            self.write(process, row['process'])
            row.update(actual_report_sha256=sha(p / 'report.private.json'),
                       actual_complete_sha256=sha(p / 'COMPLETE.json'), actual_process_sha256=sha(process))
        self.write(self.diagnostic / 'report.private.json', self.report)
        self.write(self.diagnostic / 'COMPLETE.json', dict(status=self.report['status'], report_sha256=sha(self.diagnostic / 'report.private.json')))
        self.write(self.diagnostic / 'EXIT.json', dict(exit_code=0, complete_present=True))

    def export(self):
        return export_completed(input_run=self.diagnostic, child_run=self.children, frozen_source=self.source,
            output_directory=self.out, expected_source_manifest_sha256=self.manifest_sha,
            expected_protocol_sha256=self.protocol_sha, expected_report_sha256=sha(self.diagnostic / 'report.private.json'),
            expected_complete_sha256=sha(self.diagnostic / 'COMPLETE.json'), expected_exit_sha256=sha(self.diagnostic / 'EXIT.json'))

    def reject(self):
        self.bind()
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.out.exists())

    def test_complete_positive_preserves_original_failure(self):
        receipt = self.export()
        v = json.loads((self.out / 'metrics.json').read_text())
        self.assertEqual(v['compared_numeric_fields'], 81)
        self.assertEqual(v['original_parent_exit_code'], 1)
        self.assertIsNone(v['successful_original_pair_cli_seconds'])
        self.assertEqual(receipt['protocol_sha256'], self.protocol_sha)

    def test_empty_comparison_rejected(self):
        self.report['comparisons'] = {}
        self.reject()

    def test_missing_control_rejected(self):
        self.report['controls'].pop(CONTROL_FILES[0])
        self.reject()

    def test_missing_checkpoint_rejected(self):
        self.report['records'][0]['report']['checkpoint_sha256'].pop(NPZ_FILES[0])
        self.reject()

    def test_missing_actual_module_rejected(self):
        self.report['records'][0]['report']['actual_loaded_module_sha256'].pop(ROUTE_MODULES[0])
        self.reject()

    def test_report_boolean_truthy_string_rejected(self):
        self.report['comparisons'][NPZ_FILES[0]]['complete_file_sha_equal'] = 'false'
        self.reject()

    def test_control_truthy_integer_rejected(self):
        self.report['controls'][CONTROL_FILES[0]] = 1
        self.reject()

    def test_boolean_or_float_count_rejected(self):
        self.report['records'][0]['report']['scores'] = 6.0
        self.reject()

    def test_protocol_input_binding_rejected(self):
        for row in self.report['records']:
            row['report']['input_sha256']['brain'] = 'e' * 64
        self.reject()

    def test_actual_rng_control_binding_rejected(self):
        row = self.report['records'][0]
        p = self.children / 'pair' / row['label'] / CONTROL_FILES[1]
        value = json.loads(p.read_text()); value['observed_random_draws'] += 1
        self.write(p, value); row['report']['checkpoint_sha256'][CONTROL_FILES[1]] = sha(p)
        self.reject()

    def test_type_sensitive_actual_control_rejected(self):
        row = self.report['records'][1]
        p = self.children / 'pair' / row['label'] / CONTROL_FILES[0]
        self.write(p, dict(rounds=40, accepted=1)); row['report']['checkpoint_sha256'][CONTROL_FILES[0]] = sha(p)
        self.reject()

    def test_actual_archive_difference_not_hidden_by_passed_flags(self):
        row = self.report['records'][1]
        p = self.children / 'pair' / row['label'] / NPZ_FILES[0]
        with zipfile.ZipFile(p, 'a') as archive:
            archive.writestr('extra.npy', b'different complete output')
        row['report']['checkpoint_sha256'][NPZ_FILES[0]] = sha(p)
        self.reject()

    def test_private_nested_string_rejected_before_publication(self):
        self.report['records'][0]['report']['stages']['private_path'] = '/private/example'
        self.reject()

    def test_comparison_field_private_string_rejected(self):
        field = next(iter(self.report['comparisons'][NPZ_FILES[0]]['fields'].values()))
        field['private_path'] = '/private/example'
        self.reject()

    def test_unrecognized_complete_status_rejected(self):
        self.report['records'][0]['report']['status'] = 'other_complete'
        self.reject()


if __name__ == '__main__':
    unittest.main()
