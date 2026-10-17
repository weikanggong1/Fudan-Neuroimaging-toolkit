"""外层监测metadata契约，不模拟影像或GPU，不运行科学命令。"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'validation/recon_all/optimizations/20261017_sphere_normals/monitor_science_pair.py'
SPEC = importlib.util.spec_from_file_location('fnit_outer_memory_monitor', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MonitorContractTest(unittest.TestCase):
    def test_existing_source_manifest_binds_commit_and_two_modules_without_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {}
            for name in ('profiling.py', 'nvml_process_memory.py'):
                path = root / 'src/fnit/recon_all' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# existing test source metadata')
                files[str(path.relative_to(root))] = MODULE.file_sha(path)
            manifest = root / 'source_manifest.json'
            manifest.write_text(json.dumps({'schema': 'fnit-sphere-source-v1', 'commit': 'a' * 40, 'files': files}))
            inputs = dict(source_root=root, source_manifest=manifest, source_manifest_sha256=MODULE.file_sha(manifest),
                source_commit='a' * 40, profiling_sha256=files['src/fnit/recon_all/profiling.py'],
                nvml_sha256=files['src/fnit/recon_all/nvml_process_memory.py'])
            value = MODULE.source_manifest_identity(**inputs)
            self.assertEqual(value['source_manifest_declared_files'], 2)
            self.assertFalse((root / 'SOURCE_COMMIT').exists())
            with self.assertRaisesRegex(ValueError, 'commit'):
                MODULE.source_manifest_identity(**{**inputs, 'source_commit': 'b' * 40})
            with self.assertRaisesRegex(ValueError, 'module SHA'):
                MODULE.source_manifest_identity(**{**inputs, 'profiling_sha256': 'c' * 64})

    def test_unknown_and_failed_frames_never_become_zero_or_pass(self):
        total = MODULE.empty_totals()
        MODULE.update_totals(total, {'capture_started_monotonic': 1., 'sampler_row': None})
        MODULE.update_totals(total, {'capture_started_monotonic': 2., 'sampler_row': {
            'ownership': 'unresolved', 'tree_total_bytes': None, 'known_tree_bytes': 20,
            'target_device_used_bytes': 500, 'target_compute_process_sum_bytes': 450}})
        self.assertIsNone(total['owned_simultaneous_sampled_peak_bytes'])
        self.assertEqual(total['partial_attributed_same_snapshot_peak_bytes'], 20)
        self.assertEqual(total['unknown_or_failed_frames'], 2)
        self.assertEqual(total['maximum_observed_interval_seconds'], 1.)

    def test_peak_is_maximum_of_same_snapshot_sums_not_sum_of_pid_peaks(self):
        total = MODULE.empty_totals()
        for tick, value in ((1., 100), (2., 150), (3., 80)):
            MODULE.update_totals(total, {'capture_started_monotonic': tick, 'sampler_row': {
                'ownership': 'resolved', 'tree_total_bytes': value, 'known_tree_bytes': value}})
        self.assertEqual(total['owned_simultaneous_sampled_peak_bytes'], 150)
        MODULE.update_totals(total, {'capture_started_monotonic': 4., 'sampler_row': None})
        self.assertIsNone(total['owned_simultaneous_sampled_peak_bytes'])

    def test_missing_or_incomplete_pair_never_assigns_wrapper_gap_to_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pair.json'
            observer = MODULE.RunObserver(path)
            self.assertEqual(observer.poll(set(), lambda pid: {}), {'status': 'not_observed'})
            path.write_text(json.dumps({'schema_version': 1, 'scope': MODULE.PAIR_SCOPE, 'runs': []}))
            self.assertEqual(observer.poll(set(), lambda pid: {})['status'], 'no_verified_live_cli')
            self.assertFalse(observer.run_metadata)

    def test_explicit_empty_command_is_rejected_without_resource_or_process_access(self):
        with self.assertRaisesRegex(ValueError, 'explicit command'):
            MODULE.monitor(SimpleNamespace(), ['--'])

    def test_sample_is_bound_only_to_the_same_live_identity_across_both_queries(self):
        class Sampler:
            samples, errors = [], []
            uuid = 'GPU-test'
            trusted_host_pid_map = {}
            def _tree(self, pid): return {pid}
            def _process_identity(self, pid): return {}
            def _validate_pid_map(self, mapping): return mapping
        class Query:
            def snapshot(self, uuid): return [], 123, None
        class Helper:
            def build_owned_sched_pid_map(self, query, uuid, tree, rows): return {}, []
            def sample_current_call(self, sampler):
                return {'ownership': 'unresolved', 'tree_total_bytes': None}, {'new_sample_count': 1}
        class Observer:
            def __init__(self, change): self.count, self.change = 0, change
            def poll(self, tree, identity):
                self.count += 1
                return {'status': 'verified_live_cli_metadata', 'run_id': 'A',
                        'cli_identity': {'starttime_ticks': self.count if self.change else 3}}
        same = MODULE.capture_frame(query=Query(), sampler=Sampler(), helper=Helper(),
            gpu_uuid='GPU-test', observer=Observer(False))
        self.assertEqual(same['run_binding']['status'], 'verified_live_cli_metadata')
        changed = MODULE.capture_frame(query=Query(), sampler=Sampler(), helper=Helper(),
            gpu_uuid='GPU-test', observer=Observer(True))
        self.assertEqual(changed['run_binding']['status'], 'not_bound_across_sample')

    def test_current_sampler_failure_cannot_reuse_historical_row(self):
        class Sampler:
            samples = [{'ownership': 'resolved', 'tree_total_bytes': 999}]
            errors = []
            trusted_host_pid_map = {'old': 'mapping'}
            uuid = 'GPU-test'
            def _tree(self, pid): return {pid}
            def _process_identity(self, pid): return {}
        class Query:
            def snapshot(self, uuid): raise RuntimeError('private query error')
        frame = MODULE.capture_frame(query=Query(), sampler=Sampler(), helper=None,
            gpu_uuid='GPU-test', observer=MODULE.RunObserver(None))
        self.assertIsNone(frame['sampler_row'])
        self.assertEqual(frame['status'], 'unknown_sample')
        self.assertNotIn('private query error', json.dumps(frame))


if __name__ == '__main__':
    unittest.main()
