"""无GPU的回执与取消契约测试；不运行影像计算。"""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load('run_resource_replay_queue')
verify = load('verify_replay_bindings')


class ReplayWrapperTests(unittest.TestCase):
    def run_queue(self, root, mode):
        script = root / 'admission.py'
        script.write_text('# mocked admission\n')
        plan = {'python': 'unused', 'admission_script': str(script),
                'admission_script_sha256': queue.digest(script), 'shared_lock': str(root / 'lock'),
                'cases': [{'id': 'case', 'original_config': str(root / 'config.json'),
                           'retry_root': str(root / 'retry'), 'admission_report': str(root / 'admission.json')}]}
        plan_path = root / 'plan.json'
        plan_path.write_text(json.dumps(plan))
        report_path = root / 'queue.json'
        handlers = {}
        class Child:
            pid = 123
            def __init__(self):
                self.signals = []
                self.alive = True
            def poll(self):
                return None if self.alive else 0
            def send_signal(self, sig):
                self.signals.append(sig)
            def wait(self):
                self.alive = False
                if mode == 'signal':
                    return 130
                return 0
        child = Child()
        def popen(*args, **kwargs):
            if mode == 'spawn_error':
                raise OSError('cannot spawn')
            if mode == 'signal':
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            if mode in ('success', 'bad_json'):
                (root / 'admission.json').write_text('{bad' if mode == 'bad_json' else json.dumps({'status': 'complete'}))
                target = root / 'retry/diagnostics'
                target.mkdir(parents=True)
                (target / 'completion.json').write_text(json.dumps({'execution_status': 'complete', 'exit_code': 0, 'pipeline_status': 'complete'}))
            return child
        def install(sig, handler):
            previous = handlers.get(sig, signal.SIG_DFL)
            handlers[sig] = handler
            return previous
        with patch.object(sys, 'argv', ['queue', '--plan', str(plan_path), '--report', str(report_path)]), \
             patch.object(queue.subprocess, 'Popen', side_effect=popen), \
             patch.object(queue.signal, 'signal', side_effect=install):
            code = queue.main()
        return code, json.loads(report_path.read_text()), child

    def test_success_requires_complete_receipts(self):
        for mode, expected in [('success', 0), ('no_receipts', 1), ('bad_json', 1), ('spawn_error', 1)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                code, report, _ = self.run_queue(Path(directory), mode)
                self.assertEqual(code, expected)
                self.assertTrue(report['queue_finished'])
                self.assertEqual(report['successful'], int(mode == 'success'))
                self.assertEqual(report['all_cases_succeeded'], mode == 'success')
                self.assertEqual(report['exit_code'], expected)

    def test_signal_in_spawn_window_cancels_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, child = self.run_queue(Path(directory), 'signal')
            self.assertEqual(code, 130)
            self.assertIn(signal.SIGTERM, child.signals)
            self.assertFalse(report['queue_finished'])
            self.assertEqual(report['status'], 'interrupted')
            self.assertFalse(report['all_cases_succeeded'])

    def test_validation_exception_is_json_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            with patch.object(verify, '_verify', side_effect=FileNotFoundError('missing binding')):
                result = verify.verify(cohort_root='unused', expected={}, report_path=path)
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['verification_complete'])
            self.assertEqual(json.loads(path.read_text()), result)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                verify.verify(cohort_root='unused', expected={}, report_path=path)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
