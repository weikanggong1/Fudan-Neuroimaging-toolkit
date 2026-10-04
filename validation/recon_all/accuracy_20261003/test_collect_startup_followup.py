"""CPU fixtures validate metadata collection; never a real benchmark."""
import json
from pathlib import Path
import tempfile
import unittest
from collect_startup_followup import collect


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.output = self.root / 'snapshot'

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, relative, value):
        path = self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_pending_guard_exit_is_not_pass(self):
        self.put('guard/state.json', {'status': 'exited', 'exit_code': 0})
        result = collect(self.run, self.output)
        self.assertEqual(result['completion_state'], 'pending')
        self.assertEqual(result['run_state'], 'pending')
        self.assertFalse(result['passed'])

    def test_failed_preflight_without_completion_remains_pending(self):
        self.put('guard/preflight.json', {'status': 'failed_preflight', 'exit_code': 1})
        result = collect(self.run, self.output)
        self.assertEqual(result['completion_state'], 'pending')
        self.assertEqual(result['run_state'], 'failed_preflight_or_guard')
        self.assertFalse(result['passed'])

    def complete(self, mesh=None):
        self.put('attempt_01/diagnostics/completion.json', {'execution_status': 'complete', 'exit_code': 0, 'pipeline_status': 'complete'})
        self.put('admission.json', {'status': 'complete', 'exit_code': 0, 'finished_utc': '2026-10-04T00:00:00Z'})
        pipeline = {'status': 'complete', 'output_validation': {'status': 'passed', 'expected': 138, 'present': 138, 'missing': []}}
        if mesh is not None:
            pipeline['mesh_validation'] = mesh
        self.put('attempt_01/subject/fnit-native-free-run.json', pipeline)

    def test_exit_zero_missing_mesh_cannot_pass(self):
        self.complete()
        result = collect(self.run, self.output)
        self.assertFalse(result['passed'])
        self.assertEqual(result['run_state'], 'completion_recorded_integrity_missing_or_failed')

    def test_complete_bound_metadata_can_pass(self):
        self.complete({'status': 'passed'})
        result = collect(self.run, self.output)
        self.assertTrue(result['passed'])
        self.assertEqual(result['run_state'], 'complete_integrity_passed')

    def test_real_admission_and_guard_failed_schema(self):
        self.put('guard/state.json', {'status': 'whole_case_exited', 'whole_case_exit_code': 1})
        self.put('admission.json', {'status': 'query_or_validation_failed', 'error': 'FileNotFoundError'})
        result = collect(self.run, self.output)
        self.assertEqual(result['run_state'], 'failed_preflight_or_guard')
        self.assertEqual(result['completion_state'], 'pending')
        self.assertEqual(len(result['guard_failure_records']), 2)
        self.assertFalse(result['passed'])

    def test_missing_admission_cannot_pass(self):
        self.complete({'status': 'passed'})
        (self.run / 'admission.json').unlink()
        result = collect(self.run, self.output)
        self.assertFalse(result['passed'])
        self.assertFalse(result['gates']['admission_finished_successfully'])

    def test_unfinished_cleanup_cannot_pass(self):
        self.complete({'status': 'passed'})
        self.put('admission.json', {'status': 'complete', 'exit_code': 0,
                 'finished_utc': '2026-10-04T00:00:00Z', 'cleanup_status': 'term_waiting_lock_held'})
        result = collect(self.run, self.output)
        self.assertFalse(result['passed'])

    def test_admission_needs_exit_zero_and_finished_timestamp(self):
        self.complete({'status': 'passed'})
        for index, admission in enumerate(({'status': 'complete', 'exit_code': 0},
                {'status': 'complete', 'exit_code': 1, 'finished_utc': '2026-10-04T00:00:00Z'})):
            with self.subTest(admission=admission):
                self.put('admission.json', admission)
                result = collect(self.run, self.root / ('admission_snapshot_' + str(index)))
                self.assertFalse(result['passed'])
                self.assertFalse(result['gates']['admission_finished_successfully'])

    def test_zero_expected_fake_pass_cannot_pass(self):
        self.complete({'status': 'passed'})
        self.put('attempt_01/subject/fnit-native-free-run.json', {'status': 'complete',
                 'mesh_validation': {'status': 'passed'},
                 'output_validation': {'status': 'passed', 'expected': 0, 'present': 0, 'missing': []}})
        result = collect(self.run, self.output)
        self.assertFalse(result['passed'])
        self.assertFalse(result['gates']['output138_ready'])

    def test_existing_monitor_must_finish(self):
        self.complete({'status': 'passed'})
        self.put('attempt_01/monitor/monitor.json', {'exit_code': 0, 'monitor_thread_finished': False})
        result = collect(self.run, self.output)
        self.assertFalse(result['passed'])
        self.assertFalse(result['gates']['existing_monitor_finished_successfully'])

    def test_actual_startup_wait_field_and_legacy_alias(self):
        self.put('attempt_01/subject/scripts/annotation.hemisphere-group.json',
                 {'startup_wait_actual_seconds': 3.25})
        result = collect(self.run, self.output)
        self.assertEqual(result['groups'][0]['startup_wait_actual_seconds'], 3.25)
        self.assertEqual(result['groups'][0]['startup_wait_source_field'], 'startup_wait_actual_seconds')
        self.put('attempt_01/subject/scripts/annotation.hemisphere-group.json',
                 {'startup_wait_seconds_actual': 2.5})
        legacy = collect(self.run, self.root / 'legacy_snapshot')
        self.assertEqual(legacy['groups'][0]['startup_wait_actual_seconds'], 2.5)
        self.assertEqual(legacy['groups'][0]['startup_wait_source_field'], 'startup_wait_seconds_actual')

    def test_safe_scope_does_not_follow_files_directories_or_resource_paths(self):
        secret = self.root / 'external.json'
        secret.write_text('{"do_not_read":true}')
        self.put('config.json', {'code_root': str(secret), 'weights': str(secret), 'fs_license': str(secret)})
        self.put('attempt_01/subject/mri/forbidden.json', {'forbidden': True})
        self.put('attempt_01/subject/scripts/unrelated.json', {'forbidden': True})
        self.put('guard/deeper/not_allowed.json', {'forbidden': True})
        self.put('guard/license.json', {'forbidden': True})
        (self.run / 'admission.json').symlink_to(secret)
        # Directory symlink must not even be enumerated outside the root.
        (self.run / 'attempt_01/diagnostics').symlink_to(self.root, target_is_directory=True)
        result = collect(self.run, self.output)
        self.assertEqual([x['relative'] for x in result['files']], ['config.json'])
        self.assertEqual(len(result['rejected_paths']), 3)
        self.assertFalse(result['declared_source_bindings'][0]['external_paths_read'])
        self.assertFalse(any('forbidden' in x['relative'] for x in result['files']))
        with self.assertRaises(FileExistsError):
            collect(self.run, self.output)

    def test_incomplete_json_bytes_preserved(self):
        path = self.run / 'admission.json'
        original = b'{"status":'
        path.write_bytes(original)
        result = collect(self.run, self.output)
        self.assertEqual(result['invalid_json_files'], ['admission.json'])
        self.assertEqual((self.output / 'snapshot/admission.json').read_bytes(), original)
        self.assertEqual(result['run_state'], 'invalid_or_incomplete_snapshot')
        self.assertFalse(result['passed'])

    def test_startup_attempts_do_not_equal_algorithm_calls(self):
        self.put('attempt_01/subject/scripts/annotation.hemisphere-group.json',
                 {'operation': 'annotation', 'startup_attempts': {'lh': [
                     {'worker_report': {'operation_entered': False, 'status': 'failed'}},
                     {'worker_report': {'operation_entered': True, 'status': 'complete'}}]}})
        result = collect(self.run, self.output)
        workers = result['groups'][0]['workers']
        self.assertEqual(workers['lh']['startup_attempt_count'], 2)
        self.assertEqual(workers['lh']['operation_entered_each_attempt'], [False, True])
        self.assertEqual(workers['lh']['known_operation_entries'], 1)
        self.assertIsNone(workers['rh']['startup_attempt_count'])
        self.assertFalse(workers['rh']['operation_entry_count_known'])


if __name__ == '__main__':
    unittest.main()
