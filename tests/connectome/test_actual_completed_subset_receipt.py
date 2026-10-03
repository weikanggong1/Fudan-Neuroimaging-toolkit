"""Explicit failed-controller subset receipts; CPU metadata fixtures, not MRI."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

source = Path(__file__).resolve().parents[2] / 'tools/reference'
sys.path.insert(0, str(source))
import connectome_actual_gpu_completed_subset as subset
v9_file = source / 'connectome_actual_gpu_origins_v9.py'
if not v9_file.exists(): v9_file = source / 'connectome_actual_gpu_origins.py'
spec = importlib.util.spec_from_file_location('connectome_actual_gpu_origins', v9_file)
origins = importlib.util.module_from_spec(spec); spec.loader.exec_module(origins)
sys.modules['connectome_actual_gpu_origins'] = origins
# Reuse mature project fixtures, read-only. On integration this is repo-local;
# isolated task checkout falls back to the actual existing integration fixture.
fixtures_path = Path(__file__).with_name('test_actual_selected_gpu_origin_reader.py')
if not fixtures_path.exists():
    fixtures_path = Path('/tmp/fnit-connectome-tenraw-20261002/integration/tests/connectome/test_actual_selected_gpu_origin_reader.py')
spec = importlib.util.spec_from_file_location('selected_subset_mature_fixtures', fixtures_path)
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
fixtures.origins = origins; fixtures.compare.gpu_origins = origins


class ReceiptTests(fixtures.SelectedOriginFixture):
    def setUp(self):
        super().setUp()
        self.keys = ['candidate/' + case['case_id'] for case in self.cases]
        self.state.update(status='selected_recovery_failed_or_ineligible', selected_pairs=self.keys,
                          selected_attempted=7, selected_completed=6)
        self.state['cases'] = {self.keys[0]: self.record}
        self.rows = [self.declaration]
        for index in range(1, 7):
            case_id = self.cases[index]['case_id']; job = self.new_root / 'candidate' / case_id; job.mkdir(parents=True)
            record = copy.deepcopy(self.record); record['case_id'] = case_id
            response = record['response']; response['job_root'] = str(job)
            response['gpu_result']['case_id'] = case_id
            if index == 6:
                record['status'] = 'failed_or_ineligible'; response['gpu_result'].update(status='failed', exit_code=1)
                response['eligibility'].update(status='not_eligible', execution_status='failed', memory_budget=None)
            for filename, field in (('gpu_report.json', 'gpu_result'), ('raw_bids_wall.json', None),
                                    ('recovery_eligibility.json', 'eligibility'), ('recovery_config.json', 'new_config'),
                                    ('recovery_binding.json', 'binding')):
                self.write(job / filename, response[field] if field else {'status': 'failed' if index == 6 else 'completed'})
            response.update(GPU_report=self.identity(job / 'gpu_report.json'), wall_report=self.identity(job / 'raw_bids_wall.json'),
                            configuration=self.identity(job / 'recovery_config.json'))
            self.state['cases'][self.keys[index]] = record
            row = copy.deepcopy(self.declaration); row['case_id'] = case_id; row['replacement'].update(root=str(job), configuration=response['configuration'])
            self.rows.append(row)
        self.producer = self.identity(source / 'connectome_actual_gpu_completed_subset.py')
        self.receipt_path = self.root / 'receipt.json'
        self.origins_path = self.root / 'actual-seven-origins.json'
        self.persist_all()

    def persist_all(self):
        self.write(self.driver_path, self.state)
        for row in self.rows: row['replacement']['driver_status'] = self.identity(self.driver_path)
        self.write(self.origins_path, {'schema_version': 1, 'scope': 'explicit_actual_GPU_monitor_recovery', 'bindings': self.rows})
        self.declaration = copy.deepcopy(self.rows[0]); self.persist_declaration()

    def receipt(self):
        return subset.build_receipt(self.identity(self.driver_path), self.identity(self.origins_path), self.keys[:6], self.producer)

    def attach(self):
        self.write(self.receipt_path, self.receipt())
        self.declaration['completed_case_subset_receipt'] = self.identity(self.receipt_path)
        self.persist_declaration()

    def test_explicit_receipt_normalizes_without_rewriting_failed_driver(self):
        self.attach(); before = self.driver_path.read_bytes()
        normalized, selected = origins.normalize_selected_declaration(self.declaration)
        self.assertNotIn('completed_case_subset_receipt', normalized)
        self.assertEqual(selected['completed_case_subset_receipt'], self.identity(self.receipt_path))
        self.assertEqual(self.driver_path.read_bytes(), before)
        self.assertEqual(json.loads(before)['status'], 'selected_recovery_failed_or_ineligible')

    def test_failed_driver_without_receipt_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'completed immutable driver'):
            origins.normalize_selected_declaration(self.declaration)

    def test_failure_pending_and_all_ten_history_preserved(self):
        receipt = self.receipt()
        self.assertEqual(receipt['failed_records'], {self.keys[6]: self.state['cases'][self.keys[6]]})
        self.assertEqual(receipt['not_dispatched_case_keys'], self.keys[7:])
        self.assertEqual(len(receipt['case_ledger']), 10)
        self.assertEqual(receipt['completed_case_keys'], self.keys[:6])
        self.assertFalse(receipt['full_ten_complete'])

    def test_failed_case_cannot_use_receipt(self):
        self.attach(); row = copy.deepcopy(self.rows[6]); row['completed_case_subset_receipt'] = self.identity(self.receipt_path)
        with self.assertRaisesRegex(ValueError, 'failed/pending case'):
            origins.normalize_selected_declaration(row)

    def test_unknown_running_or_non_scientific_failure_rejected(self):
        for status in ('running', 'preflighting', 'unknown_failure', 'completed_selected_subset'):
            self.state['status'] = status; self.persist_all()
            with self.subTest(status=status), self.assertRaises(ValueError): self.receipt()
        self.state['status'] = 'selected_recovery_failed_or_ineligible'
        self.state['cases'][self.keys[6]]['response']['gpu_result']['exit_code'] = 0; self.persist_all()
        with self.assertRaises(ValueError): self.receipt()

    def test_changed_driver_report_or_origin_binding_rejected(self):
        self.attach(); receipt_identity = self.identity(self.receipt_path)
        self.driver_path.write_bytes(self.driver_path.read_bytes() + b'\n')
        with self.assertRaises(ValueError): subset.verify_receipt(receipt_identity, self.declaration, self.state)
        self.persist_all(); self.attach()
        self.gpu_path.write_text('{}')
        with self.assertRaises(ValueError): origins.normalize_selected_declaration(self.declaration)

    def test_missing_failure_history_or_duplicate_completed_key_rejected(self):
        with self.assertRaises(ValueError):
            subset.build_receipt(self.identity(self.driver_path), self.identity(self.origins_path), self.keys[:5] + [self.keys[0]], self.producer)
        self.attach(); value=json.loads(self.receipt_path.read_text()); value['failed_records'] = {}
        self.write(self.receipt_path,value); self.declaration['completed_case_subset_receipt']=self.identity(self.receipt_path)
        with self.assertRaises(ValueError): origins.normalize_selected_declaration(self.declaration)

    def test_original_complete_proof_and_memory_guards_still_run(self):
        self.attach(); binding = self.load()
        evidence = origins.verify_replacement(binding, self.GPU, self.wall, self.case, self.subject_dir)
        self.assertEqual(evidence['status'], 'actual_eligible_replacement_verified')
        GPU = copy.deepcopy(self.GPU); GPU['memory_budget']['monitor_issues'] = ['errors']
        with self.assertRaises(ValueError): origins.verify_replacement(binding, GPU, self.wall, self.case, self.subject_dir)
        wall = copy.deepcopy(self.wall); wall['cli_arguments'] = ['changed science']
        with self.assertRaises(ValueError): origins.verify_replacement(binding, self.GPU, wall, self.case, self.subject_dir)



class MixedAttemptGateTests(unittest.TestCase):
    """Exercise pending/failure and explicit origin map routing; no comparison."""
    def setUp(self):
        fixture = ReceiptTests(); fixture.setUp(); fixture.attach(); self.addCleanup(fixture.doCleanups)
        for name in ("root", "keys", "driver_path", "receipt_path", "rows", "fixture", "write", "identity"):
            setattr(self, name, getattr(fixture, name))
        sys.path.insert(0, str(source.parent))
        import watch_connectome_actual_mixed_completion as mixed
        self.mixed = mixed
        self.prior_path = self.root / 'prior_v2.json'
        self.write(self.prior_path, {'status':'failed_actual_comparison','end_utc':'fixture end',
            'cases':{'sub-CON01':{'status':'completed_comparison'},'sub-CON03':{'status':'completed_comparison'}}})
        self.four_keys = self.keys[6:]; self.four_driver = self.root / 'v4-driver.json'; self.four_origins = self.root / 'v4-origins.json'
        self.config = {'static_JSON_bindings': [], 'prior_v2': self.identity(self.prior_path), 'prior_v2_pairs': {},
            'selected_attempts': [{'driver_status':str(self.driver_path),'completed_case_keys':self.keys[:6],
                'completed_case_subset_receipt':self.identity(self.receipt_path)},
                {'driver_status':str(self.four_driver),'completed_case_keys':self.four_keys,'recovery_origins':str(self.four_origins)}],
            'selected_pairs':self.keys,'GPU_origin_bindings':str(self.root / 'final-ten-origins.json'), 'comparison_inputs':{}}
        self.comparison = self.fixture.options  # overwritten by a minimal pure metadata namespace
        from types import SimpleNamespace
        self.comparison = SimpleNamespace(gpu_origins=origins)

    def write_four(self, status='preflighting', terminal=False):
        state={'mode':origins.SELECTED_MODE,'status':status,'selected_pairs':self.four_keys,'full_ten_complete':False,
               'cases':{self.four_keys[0]:{'status':'completed'}}}
        if terminal:
            state.update(selected_attempted=4,selected_completed=4,cases={k:{'status':'completed'} for k in self.four_keys})
        self.write(self.four_driver,state)

    def test_v3_receipt_only_pending_until_actual_v4_exists(self):
        result=self.mixed.metadata_gate(self.config,self.comparison)
        self.assertEqual(result['status'],'pending_actual_v4_driver')
        self.assertEqual(result['v3_completed_subset_preserved'],6)
        self.assertFalse(Path(self.config['GPU_origin_bindings']).exists())

    def test_running_v4_does_not_fake_ten_completed(self):
        self.write_four()
        result=self.mixed.metadata_gate(self.config,self.comparison)
        self.assertEqual(result['status'],'pending_actual_v4_completed_subset')
        self.assertEqual(result['v4_completed_observed'],1)
        self.assertFalse(result['final_CPU_comparison_started'])

    def test_actual_v4_failure_rejected_instead_of_waiting(self):
        self.write_four(status='selected_recovery_failed_or_ineligible')
        with self.assertRaisesRegex(ValueError,'v4 ended'):
            self.mixed.metadata_gate(self.config,self.comparison)

    def test_explicit_v3_receipt_required_even_with_v4(self):
        del self.config['selected_attempts'][0]['completed_case_subset_receipt']
        with self.assertRaisesRegex(ValueError,'explicit failed-v3'):
            self.mixed.metadata_gate(self.config,self.comparison)

    def test_final_map_uses_only_six_v3_and_four_v4_then_full_guard(self):
        self.write_four(status='completed_selected_subset',terminal=True)
        rows=[]
        for key in self.four_keys:
            arm,case_id=key.split('/');row=copy.deepcopy(self.rows[0]);row.update(arm=arm,case_id=case_id)
            row['replacement'].update(driver_status=self.identity(self.four_driver),root=str(self.root/'fresh-v4'/arm/case_id))
            rows.append(row)
        self.write(self.four_origins,{'schema_version':1,'scope':'explicit_actual_GPU_monitor_recovery','bindings':rows})
        manifest=self.root/'manifest.json';self.write(manifest,{'cases':[]});self.config['comparison_inputs']={'manifest':str(manifest)}
        def reject(*args):raise ValueError('original full guard reached; no bypass')
        from unittest.mock import patch
        with patch.object(origins,'load_bindings',side_effect=reject):
            self.comparison.manifest_cases=lambda value:[]
            with self.assertRaisesRegex(ValueError,'original full guard reached'):
                self.mixed.metadata_gate(self.config,self.comparison)
        actual=json.loads(Path(self.config['GPU_origin_bindings']).read_text())['bindings']
        self.assertEqual(len(actual),10)
        self.assertEqual(sum('completed_case_subset_receipt' in row for row in actual),6)
        failed_key=self.keys[6]
        replacement=[row for row in actual if row['arm']+'/'+row['case_id']==failed_key]
        self.assertEqual(len(replacement),1)
        self.assertEqual(replacement[0]['replacement']['driver_status'],self.identity(self.four_driver))
        self.assertNotIn('completed_case_subset_receipt',replacement[0])


if __name__ == '__main__': unittest.main()
