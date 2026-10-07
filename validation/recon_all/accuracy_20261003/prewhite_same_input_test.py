"""标准库合同测试；不启动原生程序，不构成 MRI benchmark。"""
import json
from pathlib import Path
import tempfile
import unittest
from prewhite_same_input import binding, command, digest, relative_path, run, source_binding, validate_output, official_repeat_binding


class ContractTests(unittest.TestCase):
    def test_binding_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input'
            path.write_bytes(b'first')
            expected = dict(path=str(path), size=5, sha256=digest(path))
            binding(expected)
            path.write_bytes(b'other')
            with self.assertRaises(ValueError):
                binding(expected)

    def test_source_hash_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'src'
            src.mkdir()
            (src / 'a.py').write_text('a = 1')
            before = source_binding(directory)
            (src / 'a.py').write_text('a = 2')
            self.assertNotEqual(before, source_binding(directory))

    def test_destination_and_command(self):
        for value in ('../surf/lh.orig', '/surf/lh.orig', 'surf/../mri/wm.mgz'):
            with self.assertRaises(ValueError):
                relative_path(value)
        argv = command('/absolute/program', Path('/absolute/subject'), 'rh')
        self.assertIn('--rh', argv)
        self.assertEqual(argv[argv.index('--threads') + 1], '4')
        self.assertEqual(argv.count('--restore-255'), 2)

    def test_nested_output_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / 'source'
            source.mkdir()
            plan = {'hemi': 'lh', 'source_root': str(source), 'inputs': [{'relative_path': 'surf/lh.orig', 'path': str(base / 'subject/surf/lh.orig')}], 'arms': [{'assets_dir': str(base / 'assets'), 'binary': {'path': str(base / 'bin/program')}}]}
            path = base / 'plan.json'
            path.write_text(json.dumps(plan))
            for output in (source / 'new', base / 'subject/surf/new', base / 'assets/new', base / 'bin/new'):
                with self.assertRaises(ValueError):
                    run(path, output)
                self.assertFalse(output.exists())
            self.assertEqual(list(source.iterdir()), [])

    def test_official_repeat_binding(self):
        arm = {'binary': {'path': '/program', 'sha256': 'a', 'size': 1}, 'assets_dir': '/assets', 'assets': [], 'license_path': '/license'}
        self.assertTrue(official_repeat_binding([dict(arm), dict(arm)]))
        changed = dict(arm, assets_dir='/other')
        self.assertFalse(official_repeat_binding([arm, changed]))

    def test_failure_receipt_and_output_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            plan = Path(directory) / 'plan.json'
            plan.write_text(json.dumps({'hemi': 'lh', 'source_root': str(Path(directory) / 'source'), 'inputs': [{'relative_path': 'surf/lh.orig', 'path': str(Path(directory) / 'subject/surf/lh.orig')}], 'arms': []}))
            report = run(plan, output)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(json.loads((output / 'receipt.json').read_text())['status'], 'failed')
            with self.assertRaises(FileExistsError):
                run(plan, output)


if __name__ == '__main__':
    unittest.main()
