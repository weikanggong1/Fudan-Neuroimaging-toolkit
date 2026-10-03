"""只读 collector 来源协议回归；JSON fixture 不作为 MRI benchmark。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).parents[2] / 'tools/reference/run_connectome_raw_readonly_audit_cohort.py'
spec = importlib.util.spec_from_file_location('readonly_audit_collector_fixture', path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class AuditOriginFixture(unittest.TestCase):
    def test_actual_manifest_record_with_size_matches_audit_path_and_sha(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audit.json'
            config = {'producer_worker': '/fixture/worker.py', 'audit_worker': '/fixture/audit.py',
                      'source_files': {'/fixture/worker.py': 'a' * 64, '/fixture/audit.py': 'b' * 64},
                      'raw_manifest': {'sha256': 'c' * 64}}
            source = {'reference_manifest': {'path': '/fixture/actual_case/reference_manifest.json',
                                            'sha256': 'd' * 64, 'size_bytes': 999}}
            report = {'case_id': 'sub-fixture', 'contract_status': 'passed', 'scientific_parity': 'not_assessed',
                      'manifest': {key: source['reference_manifest'][key] for key in ('path', 'sha256')},
                      'actual_worker': {'sha256': 'a' * 64},
                      'raw_case_binding': {'raw_manifest_sha256': 'c' * 64}, 'audit_script_sha256': 'b' * 64}
            path.write_text(json.dumps(report))
            self.assertEqual(tool.validated_audit(path, 'sub-fixture', source, config), report)
            for group, key, wrong in [('manifest', 'sha256', 'e' * 64),
                                       ('manifest', 'path', '/fixture/another_case/reference_manifest.json'),
                                       ('actual_worker', 'sha256', 'e' * 64),
                                       ('raw_case_binding', 'raw_manifest_sha256', 'e' * 64)]:
                changed = deepcopy(report); changed[group][key] = wrong
                path.write_text(json.dumps(changed))
                with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                    tool.validated_audit(path, 'sub-fixture', source, config)
            for key, wrong in [('scientific_parity', 'passed'), ('audit_script_sha256', 'e' * 64),
                               ('case_id', 'sub-another')]:
                changed = deepcopy(report); changed[key] = wrong
                path.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    tool.validated_audit(path, 'sub-fixture', source, config)

    def test_atomic_status_records_exact_snapshot_and_refuses_changed_source(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            tool.atomic(path, {'scope': 'fixture_only', 'state': 'waiting'})
            record = tool.record(path)
            self.assertEqual(tool.checked(record), record)
            tool.atomic(path, {'scope': 'fixture_only', 'state': 'completed'})
            with self.assertRaises(ValueError):
                tool.checked(record)


if __name__ == '__main__':
    unittest.main()
